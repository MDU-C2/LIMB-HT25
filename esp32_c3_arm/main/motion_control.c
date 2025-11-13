#include "motion_control.h"

#include <math.h>
#include <string.h>

#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"

#include "app_config.h"

static const char *TAG = "motion";

// Motion control context
typedef struct {
    motion_control_config_t cfg;
    portMUX_TYPE spinlock;
    
    // LEDC configuration
    ledc_timer_t ledc_timer;
    ledc_channel_t ledc_channel;
    uint32_t duty_50_percent;
    
    // Motion state
    volatile bool estop_active;
    volatile bool is_moving;
    volatile float current_velocity_dps;      // Current velocity in degrees per second
    volatile float target_angle_deg;         // Target angle in degrees
    volatile float current_angle_deg;        // Current angle (from feedback or estimated)
    volatile bool use_position_feedback;      // Whether to use external position feedback
    
    // Calculated parameters
    float steps_per_degree;
    float max_velocity_sps;                  // Max velocity in steps per second
    float min_velocity_sps;                  // Min velocity in steps per second
    float max_accel_sps2;                    // Max acceleration in steps per second squared
} motion_ctx_t;

static motion_ctx_t s_motion = {0};

// Helper: clamp float value
static inline float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/*
 * @brief Initialize the motion control
 */
esp_err_t motion_control_init(const motion_control_config_t *config)
{
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }

    memset(&s_motion, 0, sizeof(s_motion));
    s_motion.cfg = *config;
    s_motion.spinlock = (portMUX_TYPE)portMUX_INITIALIZER_UNLOCKED;
    
    // Calculate steps per degree
    s_motion.steps_per_degree = (float)(config->steps_per_revolution * config->microstepping) 
                                * config->gear_ratio / 360.0f;
    
    // Convert motion parameters from degrees to steps
    s_motion.max_velocity_sps = config->max_velocity_dps * s_motion.steps_per_degree;
    s_motion.min_velocity_sps = config->min_velocity_dps * s_motion.steps_per_degree;
    s_motion.max_accel_sps2 = config->max_accel_dps2 * s_motion.steps_per_degree;
    
    // Configure GPIOs
    uint64_t pin_mask = (1ULL << config->step_gpio);
    if (config->dir_gpio != GPIO_NUM_NC) {
        pin_mask |= (1ULL << config->dir_gpio);
    }
    if (config->enable_gpio != GPIO_NUM_NC) {
        pin_mask |= (1ULL << config->enable_gpio);
    }
    
    gpio_config_t io_conf = {
        .pin_bit_mask = pin_mask,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&io_conf), TAG, "Failed to configure GPIOs");
    
    // Set initial GPIO states
    gpio_set_level(config->step_gpio, 0);
    if (config->dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(config->dir_gpio, 0);
    }
    if (config->enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(config->enable_gpio, 0); // Enable driver (active low on DRV8825)
    }
    
    // Configure LEDC timer for step generation
    ledc_timer_config_t timer_cfg = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_13_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = (uint32_t)s_motion.min_velocity_sps, // Start with minimum frequency
        .clk_cfg = LEDC_USE_APB_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_cfg), TAG, "Failed to configure LEDC timer");
    
    // Configure LEDC channel for step pin
    s_motion.ledc_timer = LEDC_TIMER_0;
    s_motion.ledc_channel = LEDC_CHANNEL_0;
    s_motion.duty_50_percent = (1 << (13 - 1)); // 50% duty for 13-bit resolution
    
    ledc_channel_config_t channel_cfg = {
        .gpio_num = config->step_gpio,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = s_motion.ledc_channel,
        .timer_sel = s_motion.ledc_timer,
        .duty = 0, // Start idle (no pulses)
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_cfg), TAG, "Failed to configure LEDC channel");
    
    ESP_LOGI(TAG, "Motion control initialized: %.3f steps/deg, max_vel=%.1f sps, min_vel=%.1f sps",
             s_motion.steps_per_degree, s_motion.max_velocity_sps, s_motion.min_velocity_sps);
    
    return ESP_OK;
}

/*
 * @brief Clamp angle to valid range
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
 * @brief Apply motion command
 */
esp_err_t motion_control_apply_command(const arm_motion_command_t *command)
{
    if (!command || !command->has_command) {
        return ESP_ERR_INVALID_ARG;
    }
    
    portENTER_CRITICAL(&s_motion.spinlock);
    if (s_motion.estop_active) {
        portEXIT_CRITICAL(&s_motion.spinlock);
        ESP_LOGW(TAG, "Ignoring motion command while ESTOP active");
        return ESP_ERR_INVALID_STATE;
    }
    portEXIT_CRITICAL(&s_motion.spinlock);
    
    float target_angle = clamp_angle(command->target_angle_deg);
    
    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.target_angle_deg = target_angle;
    
    // Update max velocity/accel if provided in command
    if (command->max_velocity_dps > 0) {
        s_motion.max_velocity_sps = command->max_velocity_dps * s_motion.steps_per_degree;
    }
    if (command->max_accel_dps2 > 0) {
        s_motion.max_accel_sps2 = command->max_accel_dps2 * s_motion.steps_per_degree;
    }
    portEXIT_CRITICAL(&s_motion.spinlock);
    
    ESP_LOGI(TAG, "Motion command: target %.2f deg", target_angle);
    return ESP_OK;
}

/*
 * @brief Handle ESTOP state
 */
void motion_control_handle_estop(estop_state_t state)
{
    bool active = (state == ESTOP_STATE_ACTIVE);
    
    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.estop_active = active;
    if (active) {
        // Stop motion immediately
        s_motion.is_moving = false;
        s_motion.current_velocity_dps = 0.0f;
        
        // Stop LEDC pulses
        ledc_set_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel, 0);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel);
        
        // Set step GPIO low
        gpio_set_level(s_motion.cfg.step_gpio, 0);
    }
    portEXIT_CRITICAL(&s_motion.spinlock);
    
    // Control enable pin (active low on DRV8825)
    if (s_motion.cfg.enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_motion.cfg.enable_gpio, active ? 1 : 0);
    }
}

/*
 * @brief Set position feedback (optional, for ADC or encoder feedback)
 */
void motion_control_set_position_feedback(float angle_deg)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.current_angle_deg = clamp_angle(angle_deg);
    s_motion.use_position_feedback = true;
    portEXIT_CRITICAL(&s_motion.spinlock);
}

/*
 * @brief Update motion control (call periodically from task)
 */
void motion_control_update(float dt_seconds)
{
    if (dt_seconds <= 0.0f) {
        return;
    }
    
    portENTER_CRITICAL(&s_motion.spinlock);
    bool estop = s_motion.estop_active;
    float target_angle = s_motion.target_angle_deg;
    float current_angle = s_motion.current_angle_deg;
    float current_vel = s_motion.current_velocity_dps;
    portEXIT_CRITICAL(&s_motion.spinlock);
    
    if (estop) {
        return;
    }
    
    // Calculate error
    float err_deg = target_angle - current_angle;
    float dist_deg = fabsf(err_deg);
    float dist_steps = dist_deg * s_motion.steps_per_degree;
    
    // Stop in deadband
    if (dist_deg <= s_motion.cfg.deadband_deg) {
        if (s_motion.is_moving) {
            ledc_set_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel, 0);
            ledc_update_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel);
            
            portENTER_CRITICAL(&s_motion.spinlock);
            s_motion.is_moving = false;
            s_motion.current_velocity_dps = 0.0f;
            portEXIT_CRITICAL(&s_motion.spinlock);
        }
        return;
    }
    
    // Set direction
    int dir = (err_deg > 0) ? 0 : 1; // Adjust to your mechanical convention
    if (s_motion.cfg.dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_motion.cfg.dir_gpio, dir);
    }
    
    // Ensure pulses are enabled when moving
    if (!s_motion.is_moving) {
        ledc_set_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel, s_motion.duty_50_percent);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel);
        
        portENTER_CRITICAL(&s_motion.spinlock);
        s_motion.is_moving = true;
        portEXIT_CRITICAL(&s_motion.spinlock);
    }
    
    // Calculate maximum velocity allowed by remaining distance (braking rule)
    float vmax_allowed_sps = sqrtf(2.0f * s_motion.max_accel_sps2 * dist_steps);
    
    // Target speed: min of max velocity and velocity allowed by distance
    float v_target_sps = fminf(s_motion.max_velocity_sps, vmax_allowed_sps);
    v_target_sps = fmaxf(v_target_sps, s_motion.min_velocity_sps);
    
    // Convert to degrees per second for state tracking
    float v_target_dps = v_target_sps / s_motion.steps_per_degree;
    
    // Slew velocity with acceleration limit
    float dv = s_motion.max_accel_sps2 * dt_seconds;
    float v_current_sps = current_vel * s_motion.steps_per_degree;
    
    if (v_current_sps < v_target_sps) {
        v_current_sps = fminf(v_current_sps + dv, v_target_sps);
    } else if (v_current_sps > v_target_sps) {
        v_current_sps = fmaxf(v_current_sps - dv, v_target_sps);
    }
    
    // Clamp to valid range
    v_current_sps = clampf(v_current_sps, s_motion.min_velocity_sps, s_motion.max_velocity_sps);
    int hz = (int)v_current_sps;
    
    // Update LEDC frequency
    ledc_set_freq(LEDC_LOW_SPEED_MODE, s_motion.ledc_timer, hz);
    
    // Update state
    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.current_velocity_dps = v_current_sps / s_motion.steps_per_degree;
    
    // Estimate position if not using feedback (integrate velocity)
    if (!s_motion.use_position_feedback) {
        float angle_change = s_motion.current_velocity_dps * dt_seconds;
        if (err_deg < 0) angle_change = -angle_change;
        s_motion.current_angle_deg = clamp_angle(s_motion.current_angle_deg + angle_change);
    }
    portEXIT_CRITICAL(&s_motion.spinlock);
}

/*
 * @brief Get current angle
 */
float motion_control_get_current_angle_deg(void)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    float angle = s_motion.current_angle_deg;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return angle;
}

/*
 * @brief Get target angle
 */
float motion_control_get_target_angle_deg(void)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    float angle = s_motion.target_angle_deg;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return angle;
}

/*
 * @brief Get error in degrees
 */
float motion_control_get_error_deg(void)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    float error = s_motion.target_angle_deg - s_motion.current_angle_deg;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return error;
}

/*
 * @brief Get status
 */
void motion_control_get_status(arm_status_t *status)
{
    if (!status) {
        return;
    }
    
    portENTER_CRITICAL(&s_motion.spinlock);
    status->angle_deg = s_motion.current_angle_deg;
    status->position_error_deg = s_motion.target_angle_deg - s_motion.current_angle_deg;
    status->estop_active = s_motion.estop_active;
    portEXIT_CRITICAL(&s_motion.spinlock);
}
