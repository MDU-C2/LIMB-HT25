#include "motion_control.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"
#include "hal/gpio_ll.h"

#include "app_config.h"

static const char *TAG = "motion";

// Define the motion control context structure to store the motion control state
typedef struct {
    motion_control_config_t cfg;
    gptimer_handle_t timer;
    portMUX_TYPE spinlock;
    volatile bool pulse_active;
    volatile uint32_t pulse_tick_counter;
    volatile bool moving;
    volatile bool estop_active;
    volatile uint32_t current_interval_ticks;
    volatile uint32_t tick_accumulator;
    volatile int32_t steps_remaining;
    volatile int32_t steps_executed;
    volatile int32_t current_position_steps;
    int32_t target_position_steps;
    int32_t move_total_steps;
    int direction_sign;
    float steps_per_degree;
    float current_velocity_sps;
    float max_velocity_sps;
    float max_accel_sps2;
    float peak_velocity_sps;
    int32_t accel_steps;
    int32_t decel_steps;
    uint32_t base_period_ticks;
    uint32_t pulse_width_ticks;
} motion_ctx_t;

static motion_ctx_t s_motion = {0}; // Global motion control context

/*
 * @brief Set the level of a GPIO pin
 * @param gpio_num The GPIO pin to set the level of
 * @param level The level to set the GPIO pin to
 */
static inline void IRAM_ATTR step_gpio_set_level(gpio_num_t gpio_num, int level)
{
    if (gpio_num == GPIO_NUM_NC) {
        return;
    }
    gpio_ll_set_level(GPIO_LL_GET_HW(GPIO_PORT_0), gpio_num, level);
}

/*
 * @brief Timer interrupt service routine for the motion control
 * @param timer The timer handle
 * @param edata The alarm event data
 * @param user_ctx The user context
 * @return true if the interrupt was handled, false otherwise
 */
static bool IRAM_ATTR motion_timer_isr(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata, void *user_ctx)
{
    (void)timer; // Unused parameter
    (void)edata; // Unused parameter
    motion_ctx_t *ctx = (motion_ctx_t *)user_ctx; // Cast the user context to the motion control context

    // Enter the critical section to protect the motion control state
    portENTER_CRITICAL_ISR(&ctx->spinlock);

    // Check if the pulse is active
    if (ctx->pulse_active) {
        // Increment the pulse tick counter
        ctx->pulse_tick_counter += ctx->base_period_ticks;
        if (ctx->pulse_tick_counter >= ctx->pulse_width_ticks) {
            step_gpio_set_level(ctx->cfg.step_gpio, 0); // Set the step GPIO level to 0
            ctx->pulse_active = false; // Set the pulse active flag to false
        }
        portEXIT_CRITICAL_ISR(&ctx->spinlock); // Exit the critical section
        return false;
    }

    // Check if the ESTOP is active
    if (ctx->estop_active) {
        ctx->moving = false; // Set the moving flag to false
        ctx->tick_accumulator = 0; // Set the tick accumulator to 0
        ctx->current_interval_ticks = 0; // Set the current interval ticks to 0
        portEXIT_CRITICAL_ISR(&ctx->spinlock); // Exit the critical section
        return false;
    }

    // Increment the tick accumulator
    ctx->tick_accumulator += ctx->base_period_ticks;

    // Check if the motor is moving and the current interval ticks are greater than 0
    if (ctx->moving && ctx->current_interval_ticks > 0) {
        // Check if the tick accumulator is greater than or equal to the current interval ticks
        if (ctx->tick_accumulator >= ctx->current_interval_ticks) {
            ctx->tick_accumulator -= ctx->current_interval_ticks; // Subtract the current interval ticks from the tick accumulator
            step_gpio_set_level(ctx->cfg.step_gpio, 1); // Set the step GPIO level to 1
            ctx->pulse_active = true; // Set the pulse active flag to true
            ctx->pulse_tick_counter = 0; // Set the pulse tick counter to 0
            ctx->steps_remaining -= (ctx->steps_remaining > 0) ? 1 : 0; // Subtract 1 from the steps remaining
            ctx->steps_executed += 1; // Increment the steps executed
            ctx->current_position_steps += ctx->direction_sign; // Increment the current position steps by the direction sign
            if (ctx->steps_remaining <= 0) { // Check if the steps remaining are less than or equal to 0
                ctx->moving = false; // Set the moving flag to false
            }
        }
    }

    portEXIT_CRITICAL_ISR(&ctx->spinlock);
    return false;
}

/*
 * @brief Convert a velocity in steps per second to interval ticks
 * @param velocity_sps The velocity in steps per second
 * @return The interval ticks
 */
static uint32_t velocity_to_interval_ticks(float velocity_sps)
{
    // Check if the velocity is less than or equal to 0
    if (velocity_sps <= 0.0f) {
        return 0;
    }

    
    float interval_us = 1000000.0f / velocity_sps; // Calculate the interval in microseconds
    // Calculate the ticks
    float ticks = interval_us / (float)s_motion.base_period_ticks; // Calculate the ticks
    if (ticks < 1.0f) {
        ticks = 1.0f;
    }
    return (uint32_t)lroundf(ticks) * s_motion.base_period_ticks; // Return the interval ticks
}

/*
 * @brief Apply the direction to the motion control
 * @param direction The direction to apply
 */
static void apply_direction(int direction)
{
    s_motion.direction_sign = (direction >= 0) ? 1 : -1; // Set the direction sign to 1 if the direction is greater than or equal to 0, otherwise set it to -1
    if (s_motion.cfg.dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_motion.cfg.dir_gpio, (s_motion.direction_sign > 0) ? 1 : 0); // Set the direction GPIO level to 1 if the direction sign is greater than 0, otherwise set it to 0
    }
}

/*
 * @brief Initialize the motion control
 * @param config The motion control configuration
 * @return ESP_OK if the initialization was successful, otherwise an error code
 */
esp_err_t motion_control_init(const motion_control_config_t *config)
{
    // Check if the configuration is valid
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }

    s_motion = (motion_ctx_t){0}; // Initialize the motion control context
    s_motion.cfg = *config; // Set the configuration
    s_motion.spinlock = (portMUX_TYPE)portMUX_INITIALIZER_UNLOCKED; // Initialize the spinlock

    s_motion.steps_per_degree = (float)(config->steps_per_revolution * config->microstepping) * config->gear_ratio / 360.0f; // Calculate the steps per degree
    s_motion.base_period_ticks = (config->timer_base_period_us == 0) ? 50 : config->timer_base_period_us; // Calculate the base period ticks
    uint32_t raw_pulse = (config->pulse_width_us == 0) ? 5 : config->pulse_width_us; // Calculate the raw pulse
    if (raw_pulse < s_motion.base_period_ticks) {
        raw_pulse = s_motion.base_period_ticks; // Set the raw pulse to the base period ticks if the raw pulse is less than the base period ticks
    }
    s_motion.pulse_width_ticks = raw_pulse; // Set the pulse width ticks

    uint64_t pin_mask = (1ULL << config->step_gpio); // Set the pin mask to the step GPIO
    if (config->dir_gpio != GPIO_NUM_NC) {
        pin_mask |= (1ULL << config->dir_gpio); // Set the pin mask to the direction GPIO if the direction GPIO is not NC
    }
    if (config->enable_gpio != GPIO_NUM_NC) {
        pin_mask |= (1ULL << config->enable_gpio); // Set the pin mask to the enable GPIO if the enable GPIO is not NC
    }

    // Configure the GPIOs
    gpio_config_t io_conf = {
        .pin_bit_mask = pin_mask, // Set the pin bit mask to the pin mask
        .mode = GPIO_MODE_OUTPUT, // Set the mode to output
        .pull_up_en = GPIO_PULLUP_DISABLE, // Set the pull up enable to disable
        .pull_down_en = GPIO_PULLDOWN_DISABLE, // Set the pull down enable to disable
        .intr_type = GPIO_INTR_DISABLE, // Set the interrupt type to disable
    };
    ESP_RETURN_ON_ERROR(gpio_config(&io_conf), TAG, "Failed to configure GPIOs"); // Configure the GPIOs

    gpio_set_level(config->step_gpio, 0); // Set the step GPIO level to 0
    if (config->dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(config->dir_gpio, 0); // Set the direction GPIO level to 0 if the direction GPIO is not NC
    }
    if (config->enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(config->enable_gpio, 0); // Set the enable GPIO level to 0 if the enable GPIO is not NC
    }

    // Configure the GPTimer
    gptimer_config_t timer_config = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT, // Set the clock source to default
        .direction = GPTIMER_COUNT_UP, // Set the direction to up
        .resolution_hz = config->timer_resolution_hz, // Set the resolution to the timer resolution
    };

    ESP_RETURN_ON_ERROR(gptimer_new_timer(&timer_config, &s_motion.timer), TAG, "Failed to create GPTimer"); // Create the GPTimer

    gptimer_event_callbacks_t callbacks = {
        .on_alarm = motion_timer_isr, // Set the on alarm callback to the motion timer interrupt service routine
    };
    ESP_RETURN_ON_ERROR(gptimer_register_event_callbacks(s_motion.timer, &callbacks, &s_motion), TAG, "Failed to register GPTimer callbacks"); // Register the GPTimer callbacks

    ESP_RETURN_ON_ERROR(gptimer_enable(s_motion.timer), TAG, "Failed to enable GPTimer"); // Enable the GPTimer

    gptimer_alarm_config_t alarm_config = {
        .alarm_count = s_motion.base_period_ticks, // Set the alarm count to the base period ticks
        .reload_count = 0, // Set the reload count to 0
        .flags.auto_reload_on_alarm = true, // Set the auto reload on alarm to true
    };
    ESP_RETURN_ON_ERROR(gptimer_set_alarm_action(s_motion.timer, &alarm_config), TAG, "Failed to set GPTimer alarm"); // Set the GPTimer alarm
    ESP_RETURN_ON_ERROR(gptimer_start(s_motion.timer), TAG, "Failed to start GPTimer"); // Start the GPTimer

    ESP_LOGI(TAG, "Motion control initialized: %.3f steps/deg", s_motion.steps_per_degree); // Log the motion control initialized
    return ESP_OK;
}

/*
 * @brief Clamp the angle to the maximum and minimum joint angle
 * @param angle_deg The angle to clamp
 * @return The clamped angle
 */
static float clamp_angle(float angle_deg)
{
    if (angle_deg > MAX_JOINT_ANGLE_DEG) {
        return MAX_JOINT_ANGLE_DEG;
    }
    if (angle_deg < -MAX_JOINT_ANGLE_DEG) {
        return -MAX_JOINT_ANGLE_DEG;
    }
    return angle_deg;
}

/*
 * @brief Apply the motion command to the motion control
 * @param command The motion command to apply
 * @return ESP_OK if the motion command was applied successfully, otherwise an error code
 */
esp_err_t motion_control_apply_command(const arm_motion_command_t *command)
{
    if (!command || !command->has_command) { // Check if the command is valid and has a command
        return ESP_ERR_INVALID_ARG;
    }

    if (s_motion.estop_active) { // Check if the ESTOP is active
        ESP_LOGW(TAG, "Ignoring motion command while ESTOP active");
        return ESP_ERR_INVALID_STATE;
    }

    float target_angle = clamp_angle(command->target_angle_deg); // Clamp the target angle to the maximum and minimum joint angle
    int32_t target_steps = (int32_t)lroundf(target_angle * s_motion.steps_per_degree); // Calculate the target steps

    portENTER_CRITICAL(&s_motion.spinlock); // Enter the critical section
    int32_t current_steps = s_motion.current_position_steps;
    portEXIT_CRITICAL(&s_motion.spinlock); // Exit the critical section

    int32_t delta_steps = target_steps - current_steps; // Calculate the delta steps    
    if (delta_steps == 0) { // Check if the delta steps are 0
        ESP_LOGI(TAG, "Motion command already satisfied");
        s_motion.target_position_steps = target_steps; // Set the target position steps to the target steps
        s_motion.move_total_steps = 0; // Set the move total steps to 0
        return ESP_OK; // Return ESP_OK
    }

    float max_velocity_sps = fmaxf(command->max_velocity_dps, 5.0f) * s_motion.steps_per_degree; // Calculate the maximum velocity in steps per second (sps)
    float max_accel_sps2 = fmaxf(command->max_accel_dps2, 10.0f) * s_motion.steps_per_degree; // Calculate the maximum acceleration in steps per second squared (sps^2)

    float delta_steps_abs = (float)llabs((long long)delta_steps); // Calculate the absolute delta steps
    float accel_distance = (max_velocity_sps * max_velocity_sps) / (2.0f * max_accel_sps2);
    float peak_velocity = max_velocity_sps; // Set the peak velocity to the maximum velocity

    if (2.0f * accel_distance > delta_steps_abs) {
        peak_velocity = sqrtf(delta_steps_abs * max_accel_sps2); // Calculate the peak velocity
        accel_distance = delta_steps_abs / 2.0f;
    } // Check if the 2 times the acceleration distance is greater than the absolute delta steps

    s_motion.max_velocity_sps = max_velocity_sps; // Set the maximum velocity in steps per second to the maximum velocity
    s_motion.max_accel_sps2 = max_accel_sps2; // Set the maximum acceleration in steps per second squared to the maximum acceleration
    s_motion.peak_velocity_sps = peak_velocity;
    s_motion.accel_steps = (int32_t)lroundf(accel_distance); // Set the acceleration steps to the acceleration distance
    s_motion.decel_steps = s_motion.accel_steps; // Set the deceleration steps to the acceleration steps
    if (s_motion.accel_steps < 1) {
        s_motion.accel_steps = 1;
        s_motion.decel_steps = 1;
    } // Check if the acceleration steps are less than 1

    s_motion.target_position_steps = target_steps; // Set the target position steps to the target steps 
    s_motion.move_total_steps = (int32_t)llabs((long long)delta_steps); // Set the move total steps to the absolute delta steps (number of steps to move)

    apply_direction(delta_steps > 0 ? 1 : -1); // Apply the direction to the motion control (1 for positive, -1 for negative)

    portENTER_CRITICAL(&s_motion.spinlock); // Enter the critical section
    s_motion.steps_remaining = s_motion.move_total_steps; // Set the steps remaining to the move total steps
    s_motion.steps_executed = 0; // Set the steps executed to 0
    s_motion.tick_accumulator = 0; // Set the tick accumulator to 0
    s_motion.current_interval_ticks = velocity_to_interval_ticks(peak_velocity * 0.1f); // Set the current interval ticks to the velocity to interval ticks
    s_motion.current_velocity_sps = fmaxf(peak_velocity * 0.1f, 1.0f); // Set the current velocity in steps per second to the maximum of the peak velocity times 0.1 and 1.0
    if (s_motion.current_interval_ticks == 0) {
        s_motion.current_interval_ticks = velocity_to_interval_ticks(peak_velocity); // Set the current interval ticks to the velocity to interval ticks    
    } // Check if the current interval ticks are 0
    s_motion.moving = true;
    portEXIT_CRITICAL(&s_motion.spinlock); // Exit the critical section

    ESP_LOGI(TAG, "New motion command: target %.2f deg (%ld steps)", target_angle, (long)s_motion.move_total_steps);
    return ESP_OK;
}

/*
 * @brief Handle the ESTOP state
 * @param state The ESTOP state
 */
void motion_control_handle_estop(estop_state_t state)
{
    bool active = (state == ESTOP_STATE_ACTIVE); // Check if the ESTOP is active
    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.estop_active = active; // Set the ESTOP active flag to the active state
    if (active) {
        s_motion.moving = false;
        s_motion.steps_remaining = 0;
        s_motion.current_interval_ticks = 0;
        s_motion.tick_accumulator = 0;
        s_motion.pulse_active = false;
    }
    portEXIT_CRITICAL(&s_motion.spinlock);

    if (s_motion.cfg.enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_motion.cfg.enable_gpio, active ? 1 : 0); // Set the enable GPIO level to 1 if the ESTOP is active, otherwise set it to 0
    }
    if (active) {
        gpio_set_level(s_motion.cfg.step_gpio, 0); // Set the step GPIO level to 0 if the ESTOP is active
    }
}

/*
 * @brief Update the motion control
 * @param dt_seconds The time since the last update in seconds
 */
void motion_control_update(float dt_seconds)
{
    if (dt_seconds <= 0.0f) {
        return;
    }

    // Get the motion control state
    bool estop = false;
    bool moving = false;
    int32_t steps_executed = 0;
    int32_t total_steps = 0;

    portENTER_CRITICAL(&s_motion.spinlock); // Enter the critical section
    estop = s_motion.estop_active; // Set the ESTOP active flag to the ESTOP state
    moving = s_motion.moving;
    steps_executed = s_motion.steps_executed; // Set the steps executed to the steps executed
    total_steps = s_motion.move_total_steps; // Set the move total steps to the move total steps
    portEXIT_CRITICAL(&s_motion.spinlock); // Exit the critical section

    if (estop || !moving || total_steps == 0) {
        s_motion.current_velocity_sps = 0.0f; // Set the current velocity in steps per second to 0.0
        portENTER_CRITICAL(&s_motion.spinlock);
        if (!moving) {
            s_motion.current_interval_ticks = 0; // Set the current interval ticks to 0 if the moving flag is false
        }
        portEXIT_CRITICAL(&s_motion.spinlock); // Exit the critical section
        return;
    }

    float target_velocity = s_motion.current_velocity_sps; // Set the target velocity to the current velocity in steps per second
    if (steps_executed < s_motion.accel_steps) {
        target_velocity = fminf(s_motion.current_velocity_sps + s_motion.max_accel_sps2 * dt_seconds, s_motion.peak_velocity_sps); // Calculate the target velocity
    } else if (steps_executed >= (total_steps - s_motion.decel_steps)) {
        target_velocity = fmaxf(s_motion.current_velocity_sps - s_motion.max_accel_sps2 * dt_seconds, 0.0f); // Calculate the target velocity
    } else {
        target_velocity = s_motion.peak_velocity_sps; // Set the target velocity to the peak velocity
    }

    if (target_velocity < 1.0f) {
        target_velocity = fminf(s_motion.peak_velocity_sps, s_motion.max_accel_sps2 * dt_seconds * 2.0f); // Calculate the target velocity
        if (target_velocity < 1.0f) {
            target_velocity = 1.0f; // Set the target velocity to 1.0 if the target velocity is less than 1.0
        }
    }

    s_motion.current_velocity_sps = target_velocity; // Set the current velocity in steps per second to the target velocity
    uint32_t interval_ticks = velocity_to_interval_ticks(target_velocity); // Calculate the interval ticks

    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.current_interval_ticks = interval_ticks; // Set the current interval ticks to the interval ticks
    portEXIT_CRITICAL(&s_motion.spinlock);
}

/*
 * @brief Convert steps to angle
 * @param steps The steps to convert
 * @return The angle in degrees
 */
static float steps_to_angle(int32_t steps)
{
    return (float)steps / s_motion.steps_per_degree;
}

/*
 * @brief Get the current angle in degrees
 * @return The current angle in degrees
 */
float motion_control_get_current_angle_deg(void)
{
    int32_t steps = 0;
    portENTER_CRITICAL(&s_motion.spinlock);
    steps = s_motion.current_position_steps;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return steps_to_angle(steps);
}

/*
 * @brief Get the target angle in degrees
 * @return The target angle in degrees
 */
float motion_control_get_target_angle_deg(void)
{
    return steps_to_angle(s_motion.target_position_steps);
}

/*
 * @brief Get the error in degrees
 * @return The error in degrees
 */
float motion_control_get_error_deg(void)
{
    int32_t error_steps = 0;
    portENTER_CRITICAL(&s_motion.spinlock);
    error_steps = s_motion.target_position_steps - s_motion.current_position_steps;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return steps_to_angle(error_steps);
}

/*
 * @brief Get the status of the motion control
 * @param status The status to get
 */
void motion_control_get_status(arm_status_t *status)
{
    if (!status) {
        return;
    }

    status->angle_deg = motion_control_get_current_angle_deg(); // Set the angle to the current angle
    status->position_error_deg = motion_control_get_error_deg(); // Set the position error to the error
    status->estop_active = s_motion.estop_active; // Set the ESTOP active flag to the ESTOP state
}

