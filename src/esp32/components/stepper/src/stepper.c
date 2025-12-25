#include "stepper.h"

#include <math.h>

#include "esp_err.h"
#include "potentiometer.h"
#include "limb_utils.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "hal/ledc_types.h"
#include "soc/soc_caps.h"
#include "sys/param.h"

static const char *TAG = "stepper";

// Control constants
#define ALPHA 0.1f              // Low-pass filter coefficient (0.0-1.0)
#define DEADBAND_DEG 0.5f       // Deadband in degrees (stop if error < this)
#define MIN_FREQ_HZ 50          // Minimum LEDC frequency

// Control context
typedef struct {
    stepper_control_config_t cfg;
    portMUX_TYPE spinlock;

    // LEDC config
    ledc_timer_t ledc_timer;
    ledc_channel_t ledc_channel;
    uint32_t duty_50_percent;

    // Motion state
    bool estop_active;
    bool is_moving;
    AngularVelocity current_velocity;
    PotentiometerAngle target_angle;
    PotentiometerAngle current_angle;
    bool use_position_feedback;

    // Calculated parameters
    float steps_per_degree;
    float max_velocity_sps; // steps per second
    float min_velocity_sps;
    float max_accel_sps2;

    // ADC filter state
    float filt;

    bool is_initialized;
} motion_control_context_t;

// We only support at most as many steppers as there are LEDC channels, since
// they require exclusive access anyway.
static motion_control_context_t s_contexts[SOC_LEDC_CHANNEL_NUM] = {0};

static void stop_motor(stepper_control_handle_t handle) 
{
    motion_control_context_t *ctx = &s_contexts[handle];

    ledc_set_duty(LEDC_LOW_SPEED_MODE, ctx->ledc_channel, 0);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, ctx->ledc_channel);
    if (ctx->cfg.enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(ctx->cfg.enable_gpio, 1); // Disable (active low)
    }
    portENTER_CRITICAL(&ctx->spinlock);
    ctx->is_moving = false;
    ctx->current_velocity = (AngularVelocity){0.0F};
    portEXIT_CRITICAL(&ctx->spinlock);
}


static void apply_motor_velocity(stepper_control_handle_t handle, float velocity_sps) 
{
    motion_control_context_t *ctx = &s_contexts[handle];

    if (velocity_sps > 0.0f) {
        // Enable motor
        if (ctx->cfg.enable_gpio != GPIO_NUM_NC) {
            gpio_set_level(ctx->cfg.enable_gpio, 0); // Enable (active low)
        }
        
        // Clamp frequency
        uint32_t freq_hz = MAX((uint32_t)velocity_sps, MIN_FREQ_HZ);
        
        // Update frequency and duty
        ledc_set_freq(LEDC_LOW_SPEED_MODE, ctx->ledc_timer, freq_hz);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, ctx->ledc_channel, ctx->duty_50_percent);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, ctx->ledc_channel);
    } else {
        stop_motor(handle);
    }
}

// Initialization

esp_err_t stepper_init(const stepper_control_config_t *cfg, const uint16_t *latest_potentiometer_values, uint16_t latest_potentiometer_values_len, stepper_control_handle_t* out_handle)
{

    // Validate config
    if (!cfg) return ESP_ERR_INVALID_ARG;

    motion_control_context_t ctx = {0};

    stepper_control_handle_t handle = cfg->pwm_channel;

    // Reset and store config
    ctx.cfg = *cfg;
    ctx.spinlock = (portMUX_TYPE)portMUX_INITIALIZER_UNLOCKED;

    // Compute steps per degree and motion limits
    // We define the degrees per second and need to convert that into steps per second
    // To get the steps per degree, we use: steps per revolution * gear ratio / 360 degrees
    // Maybe we can directly define the steps per second? Or do we need degree per seconds?
    ctx.steps_per_degree = (float)cfg->steps_per_rev * cfg->gear_ratio / 360.0f;
    ctx.max_velocity_sps = cfg->max_velocity.dps * ctx.steps_per_degree;
    ctx.min_velocity_sps = cfg->min_velocity.dps * ctx.steps_per_degree;
    ctx.max_accel_sps2 = cfg->max_accel.dps2 * ctx.steps_per_degree;
    ctx.min_velocity_sps = MIN(ctx.min_velocity_sps, 1.0f);

    // Configure GPIOS for STEP, DIR and ENABLE
    uint64_t pin_mask = (1ULL << cfg->step_gpio);
    if (cfg->dir_gpio != GPIO_NUM_NC) {pin_mask |= (1ULL << cfg->dir_gpio);}
    if (cfg->enable_gpio != GPIO_NUM_NC) {pin_mask |= (1ULL << cfg->enable_gpio);}
    gpio_config_t io_conf = {
        .pin_bit_mask = pin_mask,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&io_conf), TAG, "Failed to configure GPIOs");

    // Set initial GPIO states
    gpio_set_level(cfg->step_gpio, 0);
    if (cfg->dir_gpio != GPIO_NUM_NC) {gpio_set_level(cfg->dir_gpio, 0);}
    if (cfg->enable_gpio != GPIO_NUM_NC) {gpio_set_level(cfg->enable_gpio, 0);} // active low on DRV8825

    // Configure LEDC timer
    uint32_t init_freq_hz = MAX((uint32_t)ctx.min_velocity_sps, 50);

    ledc_timer_config_t timer_cfg = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_13_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = init_freq_hz,
        .clk_cfg = LEDC_USE_APB_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_cfg), TAG, "Failed to configure LEDC timer");

    // Configure LEDC channel for STEP output
    ctx.ledc_timer = LEDC_TIMER_0;
    ctx.ledc_channel = cfg->pwm_channel;
    ctx.duty_50_percent = (1 << (13 - 1)); // 50% duty for 13 bit resolution
    
    ledc_channel_config_t channel_cfg = {
        .gpio_num = cfg->step_gpio,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = ctx.ledc_channel,
        .timer_sel = ctx.ledc_timer,
        .duty = 0, // IDLE (no pulses)
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_cfg), TAG, "Failed to configure LEDC channel");
    
    // ADC setup & filtered initial angle
    if (latest_potentiometer_values_len > 0) {
        // Use initial ADC value to set intial angle (averaged for stability).
        uint16_t raw_adc = limb_average16(latest_potentiometer_values, latest_potentiometer_values_len);

        PotentiometerAngle initial_angle = potentiometer_adc_to_angle(&ctx.cfg.potentiometer, raw_adc);
        ctx.current_angle = initial_angle;
        // Our target angle should be within the range of allowed angles, even if the current angle isn't.
        ctx.target_angle = clamp_potentiometer_angle(&ctx.cfg.potentiometer, ctx.current_angle);
        ctx.use_position_feedback = true;
        ctx.filt = initial_angle.degree; // Initialize filter state
        
        ESP_LOGI(TAG, "Potentiometer initialized: ADC channel=%d, raw=%u, angle=%.2f deg", 
                 cfg->pot_adc_channel, raw_adc, initial_angle);
    } else {
        ctx.use_position_feedback = false;
        ctx.current_angle.degree = 0.0f;
        ctx.target_angle.degree = 0.0f;
        ESP_LOGI(TAG, "Position feedback disabled (no ADC channel)");
    }
    
    // Initialize motion state
    ctx.estop_active = false;
    ctx.is_moving = false;
    ctx.current_velocity = (AngularVelocity){0.0f};

    ctx.is_initialized = true;

    s_contexts[handle] = ctx;

    *out_handle = handle;
    
    ESP_LOGI(TAG, "Stepper initialized: steps/deg=%.3f, max_vel=%.2f sps, max_accel=%.2f sps²", 
             ctx.steps_per_degree, ctx.max_velocity_sps, ctx.max_accel_sps2);
    
    return ESP_OK;
}

esp_err_t stepper_deinit(stepper_control_handle_t handle) 
{
    motion_control_context_t *ctx = &s_contexts[handle];

    // Stop the motor
    stop_motor(handle);

    // TODO(johan): Do we need to actually reset ledc/gpio as well?
    
    *ctx = (motion_control_context_t){0};

    return ESP_OK;
}

void stepper_update(stepper_control_handle_t handle, float dt_seconds, const uint16_t *latest_potentiometer_values, uint16_t latest_potentiometer_values_len) 
{
    motion_control_context_t *ctx = &s_contexts[handle];

    if (dt_seconds <= 0.0f || latest_potentiometer_values_len == 0) return;

    // Read & filter pot
    uint16_t raw = limb_average16(latest_potentiometer_values, latest_potentiometer_values_len);
    ctx->filt = ctx->filt + ALPHA * ((float)raw - ctx->filt);
    PotentiometerAngle angle_deg = potentiometer_adc_to_angle(&ctx->cfg.potentiometer, (uint16_t)(ctx->filt + 0.5f));

    // Take snapshot of shared state and update current angle from feedback
    portENTER_CRITICAL(&ctx->spinlock);
    bool estop = ctx->estop_active;
    PotentiometerAngle target_angle = ctx->target_angle;
    float current_velocity_sps = ctx->current_velocity.dps * ctx->steps_per_degree;
    ctx->current_angle = angle_deg;
    ctx->use_position_feedback = true;
    portEXIT_CRITICAL(&ctx->spinlock);

    if (estop) {
        stop_motor(handle);
        return;
    }
    
    // Compute the error and distance
    float error_deg = target_angle.degree - angle_deg.degree;
    float distance_deg = fabsf(error_deg);
    float distance_sps = distance_deg * ctx->steps_per_degree;

    // Stop in deadband
    if (distance_deg < DEADBAND_DEG) {
        stop_motor(handle);
        return;
    }

    // Direction control
    if (ctx->cfg.dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(ctx->cfg.dir_gpio, (error_deg > 0.0f) ? 1 : 0);
    }

    // Braking: max velocity from remaining distance (trapezoidal profile)
    // v_max^2 = 2 * a * d  =>  v_max = sqrt(2 * a * d)
    float vmax_from_distance = sqrtf(2.0f * ctx->max_accel_sps2 * distance_sps);
    float target_velocity_sps = fminf(ctx->max_velocity_sps, vmax_from_distance);

    // Velocity ramping (simplified with clamp)
    float accel_limit = ctx->max_accel_sps2 * dt_seconds;
    float velocity_delta = target_velocity_sps - current_velocity_sps;
    current_velocity_sps += LIMB_CLAMP(velocity_delta, -accel_limit, accel_limit);

    // Clamp to minimum velocity if moving
    bool is_moving = current_velocity_sps > 0.0f;
    if (is_moving) {
        current_velocity_sps = MAX(current_velocity_sps, ctx->min_velocity_sps);
    }

    // Apply motor velocity (handles enable/disable, frequency, duty)
    apply_motor_velocity(handle, current_velocity_sps);

    // Update shared state
    portENTER_CRITICAL(&ctx->spinlock);
    ctx->is_moving = is_moving;
    ctx->current_velocity = (AngularVelocity){current_velocity_sps / ctx->steps_per_degree};
    portEXIT_CRITICAL(&ctx->spinlock);

    // Logging (periodic, not every update)
    static uint32_t log_counter = 0;
    if (++log_counter >= 100) { // Log every 100 updates
        log_counter = 0;
        ESP_LOGD(TAG, "Update: target=%.2f°, current=%.2f°, error=%.2f°, vel=%.1f sps, moving=%d",
                 target_angle, angle_deg, error_deg, current_velocity_sps, ctx->is_moving);
    }

}

// ------ Setters ------

void stepper_set_target_angle(stepper_control_handle_t handle, PotentiometerAngle angle)
{
    motion_control_context_t *ctx = &s_contexts[handle];

    angle = clamp_potentiometer_angle(&ctx->cfg.potentiometer, angle);
    portENTER_CRITICAL(&ctx->spinlock);
    ctx->target_angle = angle;
    portEXIT_CRITICAL(&ctx->spinlock);
}

void stepper_set_estop(stepper_control_handle_t handle, bool active)
{
    motion_control_context_t *ctx = &s_contexts[handle];

    portENTER_CRITICAL(&ctx->spinlock);
    ctx->estop_active = active;
    portEXIT_CRITICAL(&ctx->spinlock);
    if (active) {
        stop_motor(handle);
    }
}

// ------ Getters ------

PotentiometerAngle stepper_get_current_angle(stepper_control_handle_t handle)
{
    const motion_control_context_t *ctx = &s_contexts[handle];

    portENTER_CRITICAL(&ctx->spinlock);
    PotentiometerAngle angle = ctx->current_angle;
    portEXIT_CRITICAL(&ctx->spinlock);
    return angle;
}

PotentiometerAngle stepper_get_target_angle(stepper_control_handle_t handle)
{
    const motion_control_context_t *ctx = &s_contexts[handle];

    portENTER_CRITICAL(&ctx->spinlock);
    PotentiometerAngle angle = ctx->target_angle;
    portEXIT_CRITICAL(&ctx->spinlock);
    return angle;
}

AngularVelocity stepper_get_current_velocity(stepper_control_handle_t handle)
{
    const motion_control_context_t *ctx = &s_contexts[handle];

    portENTER_CRITICAL(&ctx->spinlock);
    AngularVelocity velocity = ctx->current_velocity;
    portEXIT_CRITICAL(&ctx->spinlock);
    return velocity;
}

bool stepper_is_moving(stepper_control_handle_t handle)
{
    const motion_control_context_t *ctx = &s_contexts[handle];

    portENTER_CRITICAL(&ctx->spinlock);
    bool moving = ctx->is_moving;
    portEXIT_CRITICAL(&ctx->spinlock);
    return moving;
}

bool stepper_has_position_feedback(stepper_control_handle_t handle)
{
    const motion_control_context_t *ctx = &s_contexts[handle];

    portENTER_CRITICAL(&ctx->spinlock);
    bool has_feedback = ctx->use_position_feedback;
    portEXIT_CRITICAL(&ctx->spinlock);
    return has_feedback;
}

const stepper_control_config_t *stepper_get_cfg(stepper_control_handle_t handle)
{
    return &s_contexts[handle].cfg;
}
