#include "adc_manager.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "potentiometer.h"
#include "servo.h"

#define LIMB_ARR_LEN(arr) (sizeof(arr) / sizeof(*(arr)))

enum {
  HV2060_MIN_PULSEWIDTH_US = 850,
  HV2060_MAX_PULSEWIDTH_US = 2150,
  HV2060_MID_PULSEWIDTH_US =
      ((HV2060_MAX_PULSEWIDTH_US - HV2060_MIN_PULSEWIDTH_US) / 2) +
      HV2060_MIN_PULSEWIDTH_US,

  INTERNAL_SERVO_POTENTIOMETER_DEGREES_OF_MOTION = 220,

  // NOTE: These should be measured at the maximum and minimum extents of
  // the motor actuations you want to support. These current values aren't
  // accurate.
  HV2060_MIN_POTENTIOMETER_ANGLE = 56,
  HV2060_MAX_POTENTIOMETER_ANGLE = 212,
  HV2060_MID_POTENTIOMETER_ANGLE =
      ((HV2060_MAX_POTENTIOMETER_ANGLE - HV2060_MIN_POTENTIOMETER_ANGLE) / 2) +
      HV2060_MIN_POTENTIOMETER_ANGLE,
  HV2060_POTENTIOMETER_ANGLE_RANGE =
      HV2060_MAX_POTENTIOMETER_ANGLE - HV2060_MIN_POTENTIOMETER_ANGLE,

  SERVO_POT_ADC_CHANNEL = ADC_CHANNEL_1,
};

const ServoConfig servo_config = {
    .gpio_pin = GPIO_NUM_0,
    .direction = SERVO_DIR_NORMAL,
    .ledc_channel = LEDC_CHANNEL_0,

    .min_angle = {HV2060_MIN_POTENTIOMETER_ANGLE},
    .max_angle = {HV2060_MAX_POTENTIOMETER_ANGLE},
    .min_pulse_us = HV2060_MIN_PULSEWIDTH_US,
    .max_pulse_us = HV2060_MAX_PULSEWIDTH_US,
    .initial_angle = {HV2060_MID_POTENTIOMETER_ANGLE},

    .max_velocity_dps = 90.F,
    .max_accel_dps2 = 100.F,
    .pot_adc_channel = SERVO_POT_ADC_CHANNEL,
    .potentiometer =
        (Potentiometer){
            .degrees_of_motion = {285.F},
            // FIXME: These need to be calibrated.
            .min_adc_value = 32,
            .max_adc_value = 1500,
            .min_joint_angle_as_potentiometer_angle =
                {HV2060_MIN_POTENTIOMETER_ANGLE},
            .max_joint_angle_as_potentiometer_angle =
                {HV2060_MAX_POTENTIOMETER_ANGLE},
            .min_joint_angle = {0.F},
            .max_joint_angle = {HV2060_POTENTIOMETER_ANGLE_RANGE},
        },
    .name = "hv2060",
};

const AdcMgrChannelConfig adc_channel_config = {
    .channel = SERVO_POT_ADC_CHANNEL,
    .sample_rate = 1000,
};

const AdcMgrConfig adc_cfg = {
    .channel_configs = &adc_channel_config,
    .channel_configs_len = 1,
    .ms_worth_of_buffer_size = 100,
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
uint16_t s_servo_potentiometer_underlying_buf[1024] = {0};
AdcMgrReadResults s_adc_read_results = {
    .channel_buffers =
        {
            [SERVO_POT_ADC_CHANNEL] =
                {
                    .data = s_servo_potentiometer_underlying_buf,
                    .capacity =
                        LIMB_ARR_LEN(s_servo_potentiometer_underlying_buf),
                },
        },
};

AdcMgrChannelBuffer *s_servo_potentiometer_adc_channel_buffer =
    &s_adc_read_results.channel_buffers[SERVO_POT_ADC_CHANNEL];
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

void app_main(void) {
  ESP_ERROR_CHECK(adc_mgr_init(adc_cfg));
  ESP_ERROR_CHECK(adc_mgr_read(&s_adc_read_results, 10));

  ServoHandle servo = 0;
  ESP_ERROR_CHECK(
      servo_init(&servo_config, s_servo_potentiometer_adc_channel_buffer->data,
                 s_servo_potentiometer_adc_channel_buffer->length, &servo));
  s_servo_potentiometer_adc_channel_buffer->length = 0;

  PotentiometerAngle stops[] = {
      {0}, {10}, {30}, {60}, {120}, {90}, {60}, {30},
  };

  TickType_t current_tick = xTaskGetTickCount();
  ESP_LOGI("test", "tick: %d", current_tick);

  while (true) {
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(10));
    ESP_ERROR_CHECK(adc_mgr_read(&s_adc_read_results, 0));
    servo_update(servo, 0.01, s_servo_potentiometer_adc_channel_buffer->data,
                 s_servo_potentiometer_adc_channel_buffer->length);
    s_servo_potentiometer_adc_channel_buffer->length = 0;
  }
}
