#include "servo.h"

#include "driver/ledc.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"

static const char *const TAG = "Servo";

enum {
  SERVO_MAX_DUTY = ((1U << LEDC_TIMER_13_BIT) - 1),
  // We assume that the frequency used by the servo is 50 Hz.
  SERVO_FREQUENCY = 50,
  SERVO_PERIOD_US = 1000000UL / SERVO_FREQUENCY,
};

// Convert microseconds to duty cycle
static uint32_t us_to_duty(const ServoConfig *servo, uint16_t us) {
  us = LIMB_CLAMP(us, servo->min_pulse_us, servo->max_pulse_us);

  return (uint32_t)((uint64_t)SERVO_MAX_DUTY * us / SERVO_PERIOD_US);
}

esp_err_t servos_init(const ServoConfig *servo_configs, uint8_t configs_len) {
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

  bool channels_assigned[LEDC_CHANNEL_MAX] = {false};

  // Configure each servo channel individually
  for (uint8_t i = 0; i < configs_len; i++) {
    const ServoConfig *servo_config = &servo_configs[i];
    ESP_LOGI(TAG, "Configuring %s on GPIO%d, Channel %d", servo_config->name,
             servo_config->gpio_pin, servo_config->ledc_channel);

    if (channels_assigned[servo_config->ledc_channel]) {
      ESP_LOGE(
          TAG,
          "Configuring multiple servos using the same channel in servos_init!");
      return ESP_ERR_INVALID_ARG;
    }
    channels_assigned[servo_config->ledc_channel] = true;

    ledc_channel_config_t channel_config = {
        .gpio_num = servo_config->gpio_pin,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = servo_config->ledc_channel,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = LEDC_TIMER_0,
    };

    ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_config), TAG,
                        "Couldn't configure ledc_channel");
    servo_move_to_degree(servo_config, servo_config->initial_angle);
  }

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

// Write angle to specific servo channel
void servo_move_to_degree(const ServoConfig *servo, PotentiometerAngle deg) {
  deg.degree =
      LIMB_CLAMP(deg.degree, servo->min_angle.degree, servo->max_angle.degree);

  uint32_t us = angle_to_pulse_width(deg, servo);

  // Set duty cycle
  uint32_t duty = us_to_duty(servo, us);

  // TODO(johan): Avoid jerkiness by adjusting movement speed based on distance
  // and direction?
  ledc_set_duty(LEDC_LOW_SPEED_MODE, servo->ledc_channel, duty);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, servo->ledc_channel);

  ESP_LOGI(TAG, "%s -> %.2f° (%lu us)", servo->name, deg.degree, us);
}
