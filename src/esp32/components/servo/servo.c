#include "servo.h"

#include "driver/ledc.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"

static const char *const TAG = "Servo";

typedef struct {
  ServoConfig cfg;
  float current_velocity_dps;
  PotentiometerAngle target_angle_deg;
  PotentiometerAngle current_angle_deg;
} ServoContext;

enum {
  SERVO_MAX_DUTY = ((1U << LEDC_TIMER_13_BIT) - 1),
  // We assume that the frequency used by the servo is 50 Hz.
  SERVO_FREQUENCY = 50,
  SERVO_PERIOD_US = 1000000UL / SERVO_FREQUENCY,
};

// We support a static amount of servo motors, so we statically allocate space
// for them.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static bool s_channels_assigned[LEDC_CHANNEL_MAX] = {false};
static ServoContext s_servo_contexts[LEDC_CHANNEL_MAX] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// Convert microseconds to duty cycle
static uint32_t us_to_duty(const ServoConfig *servo, uint16_t us) {
  us = LIMB_CLAMP(us, servo->min_pulse_us, servo->max_pulse_us);

  return (uint32_t)((uint64_t)SERVO_MAX_DUTY * us / SERVO_PERIOD_US);
}

esp_err_t servo_init(const ServoConfig *servo_config,
                     const uint16_t *latest_potentiometer_values,
                     uint16_t latest_potentiometer_values_len,
                     ServoHandle *out_handle) {
  // Configure LEDC timer (shared by all servos).
  ledc_timer_config_t ledc_timer = {
      .speed_mode = LEDC_LOW_SPEED_MODE,
      .duty_resolution = LEDC_TIMER_13_BIT,
      .timer_num = LEDC_TIMER_0,
      .freq_hz = SERVO_FREQUENCY,
      .clk_cfg = LEDC_AUTO_CLK,
  };

  ESP_RETURN_ON_ERROR(ledc_timer_config(&ledc_timer), TAG,
                      "Couldn't configure ledc_timer");
  ESP_LOGI(TAG, "Timer configured");

  ESP_LOGI(TAG, "Configuring %s on GPIO%d, Channel %d", servo_config->name,
           servo_config->gpio_pin, servo_config->ledc_channel);

  if (s_channels_assigned[servo_config->ledc_channel]) {
    ESP_LOGE(
        TAG,
        "Configuring multiple servos using the same channel in servos_init!");
    return ESP_ERR_INVALID_ARG;
  }
  s_channels_assigned[servo_config->ledc_channel] = true;

  ledc_channel_config_t channel_config = {
      .gpio_num = servo_config->gpio_pin,
      .speed_mode = LEDC_LOW_SPEED_MODE,
      .channel = servo_config->ledc_channel,
      .intr_type = LEDC_INTR_DISABLE,
      .timer_sel = LEDC_TIMER_0,
  };

  ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_config), TAG,
                      "Couldn't configure ledc_channel");

  uint16_t current_pot_adc_value = limb_average16(
      latest_potentiometer_values, latest_potentiometer_values_len);
  PotentiometerAngle current_angle = potentiometer_adc_to_angle(
      &servo_config->potentiometer, current_pot_adc_value);

  s_servo_contexts[servo_config->ledc_channel] = (ServoContext){
      .cfg = *servo_config,
      .current_angle_deg = current_angle,
      .target_angle_deg = servo_config->initial_angle,
  };

  *out_handle = servo_config->ledc_channel;

  ESP_LOGI(TAG, "Initializing servo angle to %.2f degrees.",
           current_angle.degree);

  // TODO(johan): Remove once we have an update function.
  servo_move_to_degree(*out_handle, servo_config->initial_angle);

  ESP_LOGI(TAG, "All channels configured");

  return ESP_OK;
}

static uint32_t angle_to_pulse_width(PotentiometerAngle deg,
                                     const ServoConfig *servo) {
  if (servo->direction == SERVO_DIR_REVERSE) {
    deg.degree =
        servo->max_angle.degree - (deg.degree - servo->min_angle.degree);
  }
  return LIMB_LERP_FROM_RANGE(deg.degree, servo->min_angle.degree,
                              servo->max_angle.degree, servo->min_pulse_us,
                              servo->max_pulse_us);
}

static ServoContext *servo_get_context(ServoHandle handle) {
  return &s_servo_contexts[handle];
}

static PotentiometerAngle clamp_servo_angle(const ServoConfig *cfg,
                                            PotentiometerAngle angle) {
  return (PotentiometerAngle){
      LIMB_CLAMP(angle.degree, cfg->min_angle.degree, cfg->max_angle.degree)};
}

void servo_move_to_pulse_width(ServoHandle handle, uint16_t pulse_width) {
  ServoContext *ctx = servo_get_context(handle);
  uint32_t duty = us_to_duty(&ctx->cfg, pulse_width);

  ledc_set_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.ledc_channel, duty);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.ledc_channel);
}

// Write angle to specific servo channel
void servo_move_to_degree(ServoHandle handle, PotentiometerAngle deg) {
  ServoContext *ctx = servo_get_context(handle);

  deg = clamp_servo_angle(&ctx->cfg, deg);

  uint32_t us = angle_to_pulse_width(deg, &ctx->cfg);

  // Set duty cycle
  uint32_t duty = us_to_duty(&ctx->cfg, us);

  // TODO(johan): Avoid jerkiness by adjusting movement speed based on distance
  // and direction?
  ledc_set_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.ledc_channel, duty);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.ledc_channel);

  // ESP_LOGI(TAG, "%s -> %.2f° (%lu us)", ctx->cfg.name, deg.degree, us);
}
