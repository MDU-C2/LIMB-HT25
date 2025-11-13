#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "driver/gpio.h"
#include "driver/adc.h"
#include "esp_log.h"
#include <math.h>

#define TAG "STEP_MIN"

// ---------- PINS ----------
#define PIN_STEP  6
#define PIN_DIR   7
#define PIN_EN    8

// ---------- ADC (pot) ----------
#define POT_ADC_CHANNEL   ADC1_CHANNEL_2
#define ALPHA             0.20f
#define AVG_SAMPLES       8
// Measured raw limits for your pot mapping (edit if re-calibrated)
#define RAW_MIN_CAL       1044
#define RAW_MAX_CAL       2681
#define DEG_MIN_CAL       0.0f
#define DEG_MAX_CAL       90.0f

// ---------- LEDC ----------
#define DUTY_BITS         LEDC_TIMER_13_BIT
#define DUTY_50           (1 << (13 - 1))

// ---------- Motion profile ----------
#define STEP_ANGLE_DEG    1.8f        // motor full-step angle
#define MICROSTEP         16          // DRV8825 microstep setting (1,2,4,8,16,32)
#define DEG_PER_STEP      (STEP_ANGLE_DEG / (float)MICROSTEP)

#define CTRL_PERIOD_MS    10          // control tick
#define V_MAX_HZ          3000.0f     // max step rate (steps/s)
#define V_MIN_HZ          200.0f      // minimum reliable step rate while moving
#define ACCEL_SPS2        4000.0f     // accel/decel in steps/s^2
#define DEADBAND_DEG      2.0f        // stop window

// ------------- helpers -------------
static inline float clampf_(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static int read_adc_avg(int n) {
    int acc = 0;
    for (int i = 0; i < n; ++i) acc += adc1_get_raw(POT_ADC_CHANNEL);
    return acc / n;
}

static float map_pot_to_deg(int raw)
{
    if (raw < RAW_MIN_CAL) raw = RAW_MIN_CAL;
    if (raw > RAW_MAX_CAL) raw = RAW_MAX_CAL;

    const float span_raw = (float)(RAW_MAX_CAL - RAW_MIN_CAL);
    const float span_deg = (float)(DEG_MAX_CAL - DEG_MIN_CAL);
    return DEG_MIN_CAL + (span_deg * (raw - RAW_MIN_CAL) / span_raw);
}

// ------------- main -------------
void app_main(void)
{
    // GPIO: DIR/EN
    gpio_config_t io = {
        .pin_bit_mask = (1ULL<<PIN_DIR) | (1ULL<<PIN_EN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = 0, .pull_down_en = 0, .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io);
    gpio_set_level(PIN_EN, 0);   // enable driver (active low on DRV8825 breakout)
    gpio_set_level(PIN_DIR, 0);  // initial dir

    // LEDC timer + channel for STEP
    ledc_timer_config_t tcfg = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .duty_resolution = DUTY_BITS,
        .timer_num       = LEDC_TIMER_0,
        .freq_hz         = (int)V_MIN_HZ,    // seed with a valid non-zero freq
        .clk_cfg         = LEDC_USE_APB_CLK
    };
    ledc_timer_config(&tcfg);

    ledc_channel_config_t ch = {
        .gpio_num   = PIN_STEP,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_0,
        .timer_sel  = LEDC_TIMER_0,
        .duty       = 0,   // start idle
        .hpoint     = 0
    };
    ledc_channel_config(&ch);

    // ADC init
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(POT_ADC_CHANNEL, ADC_ATTEN_DB_11);

    // Motion state
    const float dt = CTRL_PERIOD_MS / 1000.0f;
    float v_cur = 0.0f;                       // steps/s (Hz)
    int desired_angle = 43;                   // target in degrees (replace with your input)
    float filt = (float)read_adc_avg(32);

    // Ensure STEP duty is non-zero only when moving
    bool is_moving = false;

    while (1)
    {
        // --- read & filter angle ---
        int raw = read_adc_avg(AVG_SAMPLES);
        filt = filt + ALPHA * ((float)raw - filt);
        float angle_deg = map_pot_to_deg((int)(filt + 0.5f));

        // --- error / distance ---
        float err_deg   = (float)desired_angle - angle_deg;
        float dist_deg  = fabsf(err_deg);
        float dist_steps = dist_deg / DEG_PER_STEP;

        // --- stop in deadband ---
        if (dist_deg <= DEADBAND_DEG) {
            if (is_moving) {
                ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, 0);
                ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
                is_moving = false;
            }
            v_cur = 0.0f;
            vTaskDelay(pdMS_TO_TICKS(CTRL_PERIOD_MS));
            continue;
        }

        // --- set direction ---
        if (err_deg > 0) gpio_set_level(PIN_DIR, 0); // adjust to your mechanical convention
        else             gpio_set_level(PIN_DIR, 1);

        // --- ensure pulses enabled when moving ---
        if (!is_moving) {
            ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, DUTY_50);
            ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
            is_moving = true;
        }

        // --- braking rule: vmax allowed by remaining distance ---
        float vmax_allowed = sqrtf(2.0f * ACCEL_SPS2 * dist_steps);

        // --- target speed & slew with accel ---
        float v_target = fminf(V_MAX_HZ, vmax_allowed);
        v_target = fmaxf(v_target, V_MIN_HZ);

        float dv = ACCEL_SPS2 * dt;
        if (v_cur < v_target)      v_cur = fminf(v_cur + dv, v_target);
        else if (v_cur > v_target) v_cur = fmaxf(v_cur - dv, v_target);

        int hz = (int)clampf_(v_cur, V_MIN_HZ, V_MAX_HZ);
        ledc_set_freq(LEDC_LOW_SPEED_MODE, LEDC_TIMER_0, hz);

        // (optional) log a little
        // ESP_LOGI(TAG, "ang=%.2f deg, err=%.2f, d=%.1f steps, v=%.0f Hz, vt=%.0f",
        //          angle_deg, err_deg, dist_steps, v_cur, v_target);

        vTaskDelay(pdMS_TO_TICKS(CTRL_PERIOD_MS));
    }
}
