#include "stepper.h"

#include <math.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "adc_manager.h"
#include "hal/adc_types.h"

static const char *TAG = "stepper";

// Control constants
#define ALPHA 0.1f              // Low-pass filter coefficient (0.0-1.0)
#define AVG_SAMPLES 5           // Number of ADC samples to average
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
    float current_veloctiy_dps;
    float target_angle_deg;
    float current_angle_deg;
    bool use_position_feedback;

    // Calculated parameters
    float steps_per_degree;
    float max_velocity_sps; // steps per second
    float min_velocity_sps;
    float max_accel_sps2;

    // ADC filter state
    float filt;
    
    // ADC manager handle
    adc_mgr_handle_t adc_handle;
} motion_control_context_t;


static motion_control_context_t s_context = {0};

// Helper functions

static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

static float clamp_angle(float angle_deg)
{
    return clampf(angle_deg, -MAX_JOINT_ANGLE_DEG, MAX_JOINT_ANGLE_DEG);
}

// Get potentiometer value (raw)
// Could keep track of success count and return 0 if no successful reads and divide by success count
static int read_adc_avg(int n) 
{
    int acc = 0;
    int raw = 0;
    for (int i = 0; i < n; i++) {
        if (adc_mgr_read(s_context.adc_handle, &raw) == ESP_OK) {
            acc += raw;
        }
    }
    return acc / n;
}

// Map pot raw -> degrees (calibrated)
static float map_pot_to_deg(int raw)
{
    raw = clampi(raw, RAW_MIN_CAL, RAW_MAX_CAL);

    const float span_raw = (float)(RAW_MAX_CAL - RAW_MIN_CAL);
    const float span_deg = (float)(DEG_MAX_CAL - DEG_MIN_CAL);
    return DEG_MIN_CAL + (span_deg * (raw - RAW_MIN_CAL) / span_raw);
}

static void stop_motor(void) 
{
    ledc_set_duty(LEDC_LOW_SPEED_MODE, s_context.ledc_channel, 0);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, s_context.ledc_channel);
    if (s_context.cfg.enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_context.cfg.enable_gpio, 1); // Disable (active low)
    }
    portENTER_CRITICAL(&s_context.spinlock);
    s_context.is_moving = false;
    s_context.current_veloctiy_dps = 0.0f;
    portEXIT_CRITICAL(&s_context.spinlock);
}


static void apply_motor_velocity(float velocity_sps) 
{
    if (velocity_sps > 0.0f) {
        // Enable motor
        if (s_context.cfg.enable_gpio != GPIO_NUM_NC) {
            gpio_set_level(s_context.cfg.enable_gpio, 0); // Enable (active low)
        }
        
        // Clamp frequency
        uint32_t freq_hz = (uint32_t)velocity_sps;
        if (freq_hz < MIN_FREQ_HZ) freq_hz = MIN_FREQ_HZ;
        
        // Update frequency and duty
        ledc_set_freq(LEDC_LOW_SPEED_MODE, s_context.ledc_timer, freq_hz);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, s_context.ledc_channel, s_context.duty_50_percent);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, s_context.ledc_channel);
    } else {
        stop_motor();
    }
}

// Initialization

esp_err_t stepper_init(const stepper_control_config_t *cfg)
{

    // Validate config
    if (!cfg) return ESP_ERR_INVALID_ARG;

    // Reset and store config
    memset(&s_context, 0, sizeof(s_context));
    s_context.cfg = *cfg;
    s_context.spinlock = (portMUX_TYPE)portMUX_INITIALIZER_UNLOCKED;
    s_context.adc_handle = -1; // Initialize ADC handle to invalid

    // Compute steps per degree and motion limits
    // We define the degrees per second and need to convert that into steps per second
    // To get the steps per degree, we use: steps per revolution * gear ratio / 360 degrees
    // Maybe we can directly define the steps per second? Or do we need degree per seconds?
    s_context.steps_per_degree = (float)cfg->steps_per_rev * cfg->gear_ratio / 360.0f;
    s_context.max_velocity_sps = cfg->max_velocity_dps * s_context.steps_per_degree;
    s_context.min_velocity_sps = cfg->min_velocity_dps * s_context.steps_per_degree;
    s_context.max_accel_sps2 = cfg->max_accel_dps2 * s_context.steps_per_degree;
    if (s_context.min_velocity_sps > 1.0f) {s_context.min_velocity_sps = 1.0f;}

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
    uint32_t init_freq_hz = (uint32_t)s_context.min_velocity_sps;
    if (init_freq_hz < 50) {init_freq_hz = 50;}

    ledc_timer_config_t timer_cfg = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_13_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = init_freq_hz,
        .clk_cfg = LEDC_USE_APB_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_cfg), TAG, "Failed to configure LEDC timer");

    // Configure LEDC channel for STEP output
    s_context.ledc_timer = LEDC_TIMER_0;
    s_context.ledc_channel = LEDC_CHANNEL_0;
    s_context.duty_50_percent = (1 << (13 - 1)); // 50% duty for 13 bit resolution
    
    ledc_channel_config_t channel_cfg = {
        .gpio_num = cfg->step_gpio,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = s_context.ledc_channel,
        .timer_sel = s_context.ledc_timer,
        .duty = 0, // IDLE (no pulses)
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_cfg), TAG, "Failed to configure LEDC channel");
    
    // ADC setup & filtered initial angle
    if (cfg->pot_adc_channel >= 0 && cfg->pot_adc_channel < SOC_ADC_MAX_CHANNEL_NUM) {
        // Register channel with ADC manager
        adc_oneshot_chan_cfg_t chan_config = {
            .bitwidth = ADC_BITWIDTH_DEFAULT,
            .atten = ADC_ATTEN_DB_12,
        };
        
        s_context.adc_handle = adc_mgr_register_channel(cfg->pot_adc_channel, &chan_config);
        if (s_context.adc_handle < 0) {
            ESP_LOGE(TAG, "Failed to register ADC channel with ADC manager");
            return ESP_FAIL;
        }
        
        // Read initial ADC value (averaged for stability)
        int raw_adc = read_adc_avg(10);
        float initial_angle = map_pot_to_deg(raw_adc);
        s_context.current_angle_deg = clamp_angle(initial_angle);
        s_context.target_angle_deg = s_context.current_angle_deg;
        s_context.use_position_feedback = true;
        s_context.filt = initial_angle; // Initialize filter state
        
        ESP_LOGI(TAG, "ADC initialized: channel=%d, raw=%d, angle=%.2f deg", 
                 cfg->pot_adc_channel, raw_adc, initial_angle);
    } else {
        s_context.use_position_feedback = false;
        s_context.current_angle_deg = 0.0f;
        s_context.target_angle_deg = 0.0f;
        ESP_LOGI(TAG, "Position feedback disabled (no ADC channel)");
    }
    
    // Initialize motion state
    s_context.estop_active = false;
    s_context.is_moving = false;
    s_context.current_veloctiy_dps = 0.0f;
    
    ESP_LOGI(TAG, "Stepper initialized: steps/deg=%.3f, max_vel=%.2f sps, max_accel=%.2f sps²", 
             s_context.steps_per_degree, s_context.max_velocity_sps, s_context.max_accel_sps2);
    
    return ESP_OK;
}

esp_err_t stepper_deinit(void) 
{
    // Stop the motor
    stop_motor();
    
    // Note: ADC manager handles channel cleanup automatically on deinit
    // We just mark our handle as invalid
    s_context.adc_handle = -1;
    
    return ESP_OK;
}

void stepper_update(float dt_seconds) 
{
    if (dt_seconds <= 0.0f) return;

    // Read & filter pot
    int raw = read_adc_avg(AVG_SAMPLES);
    s_context.filt = s_context.filt + ALPHA * ((float)raw - s_context.filt);
    float angle_deg = map_pot_to_deg((int)(s_context.filt + 0.5f));
    angle_deg = clamp_angle(angle_deg);

    // Take snapshot of shared state and update current angle from feedback
    portENTER_CRITICAL(&s_context.spinlock);
    bool estop = s_context.estop_active;
    float target_angle = s_context.target_angle_deg;
    float current_velocity_sps = s_context.current_veloctiy_dps * s_context.steps_per_degree;
    s_context.current_angle_deg = angle_deg;
    s_context.use_position_feedback = true;
    portEXIT_CRITICAL(&s_context.spinlock);

    if (estop) {
        stop_motor();
        return;
    }
    
    // Compute the error and distance
    float error_deg = target_angle - angle_deg;
    float distance_deg = fabsf(error_deg);
    float distance_sps = distance_deg * s_context.steps_per_degree;

    // Stop in deadband
    if (distance_deg < DEADBAND_DEG) {
        stop_motor();
        return;
    }

    // Direction control
    if (s_context.cfg.dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_context.cfg.dir_gpio, (error_deg > 0.0f) ? 1 : 0);
    }

    // Braking: max velocity from remaining distance (trapezoidal profile)
    // v_max^2 = 2 * a * d  =>  v_max = sqrt(2 * a * d)
    float vmax_from_distance = sqrtf(2.0f * s_context.max_accel_sps2 * distance_sps);
    float target_velocity_sps = fminf(s_context.max_velocity_sps, vmax_from_distance);

    // Velocity ramping (simplified with clamp)
    float accel_limit = s_context.max_accel_sps2 * dt_seconds;
    float velocity_delta = target_velocity_sps - current_velocity_sps;
    current_velocity_sps += clampf(velocity_delta, -accel_limit, accel_limit);

    // Clamp to minimum velocity if moving
    if (current_velocity_sps > 0.0f && current_velocity_sps < s_context.min_velocity_sps) {
        current_velocity_sps = s_context.min_velocity_sps;
    }

    // Apply motor velocity (handles enable/disable, frequency, duty)
    apply_motor_velocity(current_velocity_sps);

    // Update shared state
    portENTER_CRITICAL(&s_context.spinlock);
    s_context.is_moving = (current_velocity_sps > 0.0f);
    s_context.current_veloctiy_dps = current_velocity_sps / s_context.steps_per_degree;
    portEXIT_CRITICAL(&s_context.spinlock);

    // Logging (periodic, not every update)
    static uint32_t log_counter = 0;
    if (++log_counter >= 100) { // Log every 100 updates
        log_counter = 0;
        ESP_LOGD(TAG, "Update: target=%.2f°, current=%.2f°, error=%.2f°, vel=%.1f sps, moving=%d",
                 target_angle, angle_deg, error_deg, current_velocity_sps, s_context.is_moving);
    }

}

// ------ Setters ------

void stepper_set_target_angle_deg(float angle_deg)
{
    angle_deg = clamp_angle(angle_deg);
    portENTER_CRITICAL(&s_context.spinlock);
    s_context.target_angle_deg = angle_deg;
    portEXIT_CRITICAL(&s_context.spinlock);
}

void stepper_set_estop(bool active)
{
    portENTER_CRITICAL(&s_context.spinlock);
    s_context.estop_active = active;
    portEXIT_CRITICAL(&s_context.spinlock);
    if (active) {
        stop_motor();
    }
}

// ------ Getters ------

float stepper_get_current_angle_deg(void)
{
    portENTER_CRITICAL(&s_context.spinlock);
    float angle = s_context.current_angle_deg;
    portEXIT_CRITICAL(&s_context.spinlock);
    return angle;
}

float stepper_get_target_angle_deg(void)
{
    portENTER_CRITICAL(&s_context.spinlock);
    float angle = s_context.target_angle_deg;
    portEXIT_CRITICAL(&s_context.spinlock);
    return angle;
}

float stepper_get_current_velocity_dps(void)
{
    portENTER_CRITICAL(&s_context.spinlock);
    float velocity = s_context.current_veloctiy_dps;
    portEXIT_CRITICAL(&s_context.spinlock);
    return velocity;
}

bool stepper_is_moving(void)
{
    portENTER_CRITICAL(&s_context.spinlock);
    bool moving = s_context.is_moving;
    portEXIT_CRITICAL(&s_context.spinlock);
    return moving;
}

bool stepper_has_position_feedback(void)
{
    portENTER_CRITICAL(&s_context.spinlock);
    bool has_feedback = s_context.use_position_feedback;
    portEXIT_CRITICAL(&s_context.spinlock);
    return has_feedback;
}
