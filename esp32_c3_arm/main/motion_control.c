#include "motion_control.h"

#include <math.h>
#include <string.h>

#include "freertos/task.h"
#include "esp_log.h"
#include "esp_check.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/adc.h"

#include "app_config.h"

static const char *TAG = "motion";

// ========================
// Forward declarations
// ========================
static inline float clampf(float x, float lo, float hi);
static float clamp_angle(float angle_deg);
static int   read_adc_avg(int n);
static float map_pot_to_deg(int raw);

// Motion control context
typedef struct {
    motion_control_config_t cfg;
    portMUX_TYPE spinlock;

    // LEDC configuration
    ledc_timer_t   ledc_timer;
    ledc_channel_t ledc_channel;
    uint32_t       duty_50_percent;

    // Motion state
    volatile bool  estop_active;
    volatile bool  is_moving;
    volatile float current_velocity_dps;   // deg/s
    volatile float target_angle_deg;       // target angle (deg)
    volatile float current_angle_deg;      // current angle (deg)
    volatile bool  use_position_feedback;  // true when using external feedback (pot/encoder)

    // Calculated parameters
    float steps_per_degree;
    float max_velocity_sps;               // max velocity [steps/s]
    float min_velocity_sps;               // min velocity [steps/s]
    float max_accel_sps2;                 // max accel [steps/s^2]

    // ADC filter state
    float filt;
} motion_ctx_t;

static motion_ctx_t s_motion = {0};

// ========================
// Helpers
// ========================

static inline float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

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

// Get pot value (raw)
static int read_adc_avg(int n)
{
    int acc = 0;
    for (int i = 0; i < n; ++i) {
        acc += adc1_get_raw(POT_ADC_CHANNEL);
    }
    return acc / n;
}

// Map pot raw -> degrees (calibrated)
static float map_pot_to_deg(int raw)
{
    if (raw < RAW_MIN_CAL) raw = RAW_MIN_CAL;
    if (raw > RAW_MAX_CAL) raw = RAW_MAX_CAL;

    const float span_raw = (float)(RAW_MAX_CAL - RAW_MIN_CAL);
    const float span_deg = (float)(DEG_MAX_CAL - DEG_MIN_CAL);
    return DEG_MIN_CAL + (span_deg * (raw - RAW_MIN_CAL) / span_raw);
}

// ========================
// Initialization
// ========================

esp_err_t motion_control_init(const motion_control_config_t *config)
{
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }

    memset(&s_motion, 0, sizeof(s_motion));
    s_motion.cfg      = *config;
    s_motion.spinlock = (portMUX_TYPE)portMUX_INITIALIZER_UNLOCKED;

    // Calculate steps per degree
    s_motion.steps_per_degree =
        (float)(config->steps_per_revolution * config->microstepping) *
        config->gear_ratio / 360.0f;

    // Convert motion parameters from degrees to steps
    s_motion.max_velocity_sps = config->max_velocity_dps * s_motion.steps_per_degree;
    s_motion.min_velocity_sps = config->min_velocity_dps * s_motion.steps_per_degree;
    s_motion.max_accel_sps2   = config->max_accel_dps2   * s_motion.steps_per_degree;

    // Ensure a sane minimum internal value
    if (s_motion.min_velocity_sps < 1.0f) {
        s_motion.min_velocity_sps = 1.0f;
    }

    // Configure GPIOs (STEP, DIR, EN)
    uint64_t pin_mask = (1ULL << config->step_gpio);
    if (config->dir_gpio != GPIO_NUM_NC) {
        pin_mask |= (1ULL << config->dir_gpio);
    }
    if (config->enable_gpio != GPIO_NUM_NC) {
        pin_mask |= (1ULL << config->enable_gpio);
    }

    gpio_config_t io_conf = {
        .pin_bit_mask = pin_mask,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&io_conf), TAG, "Failed to configure GPIOs");

    // Initial GPIO states
    gpio_set_level(config->step_gpio, 0);
    if (config->dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(config->dir_gpio, 0);
    }
    if (config->enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(config->enable_gpio, 0); // enable driver (active low on DRV8825)
    }

    // ---------- LEDC timer config (FIXED) ----------
    // Use a reasonable, always-valid base frequency; real speed will be set later
    uint32_t init_freq_hz = (uint32_t)s_motion.min_velocity_sps;
    if (init_freq_hz < 50) {          // avoid super-low freqs that break with 13-bit resolution
        init_freq_hz = 50;            // 50 Hz is fine; duty=0 when idle anyway
    }

    ledc_timer_config_t timer_cfg = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_13_BIT,
        .timer_num       = LEDC_TIMER_0,
        .freq_hz         = init_freq_hz,
        .clk_cfg         = LEDC_USE_APB_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_cfg), TAG, "Failed to configure LEDC timer");
    // -----------------------------------------------

    // Configure LEDC channel for STEP pin
    s_motion.ledc_timer      = LEDC_TIMER_0;
    s_motion.ledc_channel    = LEDC_CHANNEL_0;
    s_motion.duty_50_percent = (1 << (13 - 1)); // 50% duty for 13-bit resolution

    ledc_channel_config_t channel_cfg = {
        .gpio_num   = config->step_gpio,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = s_motion.ledc_channel,
        .timer_sel  = s_motion.ledc_timer,
        .duty       = 0,   // idle (no pulses)
        .hpoint     = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_cfg), TAG, "Failed to configure LEDC channel");

    // ---- ADC init + seed filter ----
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(POT_ADC_CHANNEL, ADC_ATTEN_DB_11);

    s_motion.filt = (float)read_adc_avg(32);              // seed IIR filter
    float init_angle = map_pot_to_deg((int)(s_motion.filt + 0.5f));
    init_angle       = clamp_angle(init_angle);

    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.current_angle_deg     = init_angle;
    s_motion.target_angle_deg      = init_angle;          // start with no error
    s_motion.current_velocity_dps  = 0.0f;
    s_motion.use_position_feedback = true;                // we use pot for feedback
    portEXIT_CRITICAL(&s_motion.spinlock);

    ESP_LOGI(TAG,
             "Motion control initialized: %.3f steps/deg, max_vel=%.1f sps, min_vel=%.1f sps, init_angle=%.2f deg",
             s_motion.steps_per_degree,
             s_motion.max_velocity_sps,
             s_motion.min_velocity_sps,
             init_angle);

    return ESP_OK;
}


// ========================
// Command & ESTOP
// ========================

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

    // Update max velocity/accel if provided
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

void motion_control_handle_estop(estop_state_t state)
{
    bool active = (state == ESTOP_STATE_ACTIVE);

    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.estop_active         = active;
    s_motion.is_moving            = active ? false : s_motion.is_moving;
    s_motion.current_velocity_dps = active ? 0.0f : s_motion.current_velocity_dps;
    portEXIT_CRITICAL(&s_motion.spinlock);

    if (active) {
        // stop LEDC pulses
        ledc_set_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel, 0);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel);

        // step pin low
        gpio_set_level(s_motion.cfg.step_gpio, 0);
    }

    // Control enable pin (active low on DRV8825)
    if (s_motion.cfg.enable_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_motion.cfg.enable_gpio, active ? 1 : 0);
    }
}

void motion_control_set_position_feedback(float angle_deg)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.current_angle_deg     = clamp_angle(angle_deg);
    s_motion.use_position_feedback = true;
    portEXIT_CRITICAL(&s_motion.spinlock);
}

// ========================
// Periodic update
// ========================

void motion_control_update(float dt_seconds)
{
    if (dt_seconds <= 0.0f) {
        return;
    }

    // ---- Read & filter pot ----
    int raw = read_adc_avg(AVG_SAMPLES);
    s_motion.filt = s_motion.filt + ALPHA * ((float)raw - s_motion.filt);
    float angle_deg = map_pot_to_deg((int)(s_motion.filt + 0.5f));
    angle_deg       = clamp_angle(angle_deg);

    // Take a snapshot of shared state
    portENTER_CRITICAL(&s_motion.spinlock);
    bool  estop        = s_motion.estop_active;
    float target_angle = s_motion.target_angle_deg;
    float current_vel  = s_motion.current_velocity_dps;
    // update current angle from feedback
    s_motion.current_angle_deg     = angle_deg;
    s_motion.use_position_feedback = true;
    portEXIT_CRITICAL(&s_motion.spinlock);

    if (estop) {
        return;
    }

    // --- error / distance ---
    float err_deg    = target_angle - angle_deg;
    float dist_deg   = fabsf(err_deg);
    float dist_steps = dist_deg * s_motion.steps_per_degree;

    // --- stop in deadband ---
    if (dist_deg <= s_motion.cfg.deadband_deg) {
        bool was_moving = false;

        // FIX: handle is_moving under lock
        portENTER_CRITICAL(&s_motion.spinlock);
        was_moving = s_motion.is_moving;
        if (was_moving) {
            s_motion.is_moving            = false;
            s_motion.current_velocity_dps = 0.0f;
        }
        portEXIT_CRITICAL(&s_motion.spinlock);

        if (was_moving) {
            ledc_set_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel, 0);
            ledc_update_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel);
        }
        return;
    }

    // --- set direction ---
    int dir = (err_deg > 0) ? 0 : 1;  // adjust if your mechanics are reversed
    if (s_motion.cfg.dir_gpio != GPIO_NUM_NC) {
        gpio_set_level(s_motion.cfg.dir_gpio, dir);
    }

    // --- ensure pulses enabled when moving ---
    bool need_enable = false;
    portENTER_CRITICAL(&s_motion.spinlock);
    if (!s_motion.is_moving) {
        s_motion.is_moving = true;
        need_enable        = true;
    }
    portEXIT_CRITICAL(&s_motion.spinlock);

    if (need_enable) {
        ledc_set_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel, s_motion.duty_50_percent);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, s_motion.ledc_channel);
    }

    // --- braking rule: vmax allowed by remaining distance ---
    float vmax_allowed_sps = sqrtf(2.0f * s_motion.max_accel_sps2 * dist_steps);

    // --- target speed (steps/s) ---
    float v_target_sps = fminf(s_motion.max_velocity_sps, vmax_allowed_sps);
    // FIX: do NOT force v_target_sps up to min_velocity_sps here;
    //      we want it to go low when close to the target.

    // convert current vel from deg/s -> steps/s
    float v_current_sps = current_vel * s_motion.steps_per_degree;
    float dv            = s_motion.max_accel_sps2 * dt_seconds;

    // --- accel-limited ramp ---
    if (v_current_sps < v_target_sps) {
        v_current_sps = fminf(v_current_sps + dv, v_target_sps);
    } else if (v_current_sps > v_target_sps) {
        v_current_sps = fmaxf(v_current_sps - dv, v_target_sps);
    }

    // --- clamp & apply frequency ---
    // FIX: allow going down to 0; only clamp to [0, max]
    if (v_current_sps < 0.0f) {
        v_current_sps = 0.0f;
    }
    if (v_current_sps > s_motion.max_velocity_sps) {
        v_current_sps = s_motion.max_velocity_sps;
    }

    int hz = (int)(fabsf(v_current_sps) + 0.5f);
    if (hz < 1) hz = 1; // safety: valid LEDC frequency

    ledc_set_freq(LEDC_LOW_SPEED_MODE, s_motion.ledc_timer, hz);

    // --- update state ---
    portENTER_CRITICAL(&s_motion.spinlock);
    s_motion.current_velocity_dps = v_current_sps / s_motion.steps_per_degree;
    // no integration here because we use real feedback above
    portEXIT_CRITICAL(&s_motion.spinlock);

    // Optional: log for tuning
     ESP_LOGI(TAG, "ang=%.2f, tgt=%.2f, err=%.2f, v=%.0f sps, vt=%.0f sps",
              angle_deg, target_angle, err_deg, v_current_sps, v_target_sps);
}

// ========================
// Getters
// ========================

float motion_control_get_current_angle_deg(void)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    float angle = s_motion.current_angle_deg;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return angle;
}

float motion_control_get_target_angle_deg(void)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    float angle = s_motion.target_angle_deg;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return angle;
}

float motion_control_get_error_deg(void)
{
    portENTER_CRITICAL(&s_motion.spinlock);
    float error = s_motion.target_angle_deg - s_motion.current_angle_deg;
    portEXIT_CRITICAL(&s_motion.spinlock);
    return error;
}

void motion_control_get_status(arm_status_t *status)
{
    if (!status) return;

    portENTER_CRITICAL(&s_motion.spinlock);
    status->angle_deg          = s_motion.current_angle_deg;
    status->position_error_deg = s_motion.target_angle_deg - s_motion.current_angle_deg;
    status->estop_active       = s_motion.estop_active;
    portEXIT_CRITICAL(&s_motion.spinlock);
}
