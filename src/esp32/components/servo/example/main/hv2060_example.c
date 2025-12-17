#include "adc_manager.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "potentiometer.h"
#include "servo.h"

#define LIMB_ARR_LEN(arr) (sizeof(arr) / sizeof(*(arr)))

enum {
  // NOTE: All of these should be measured at the maximum and minimum extents of
  // the motor actuations you want to support. These values aren't accurate.
  HV2060_MIN_PULSEWIDTH_US = 850,
  HV2060_MAX_PULSEWIDTH_US = 2150,
  HV2060_MIN_ANGLE = 0,
  HV2060_MAX_ANGLE = 120,

  SERVO_POT_ADC_CHANNEL = ADC_CHANNEL_1,
};

const ServoConfig servo_config = {
    .gpio_pin = GPIO_NUM_0,
    .direction = SERVO_DIR_NORMAL,
    .initial_angle = {((float)(HV2060_MAX_ANGLE - HV2060_MIN_ANGLE) / 2.F) +
                      (float)HV2060_MIN_ANGLE},
    .ledc_channel = LEDC_CHANNEL_0,
    .min_angle = {HV2060_MIN_ANGLE},
    .max_angle = {HV2060_MAX_ANGLE},
    .min_pulse_us = HV2060_MIN_PULSEWIDTH_US,
    .max_pulse_us = HV2060_MAX_PULSEWIDTH_US,
    .pot_adc_channel = SERVO_POT_ADC_CHANNEL,
    .potentiometer =
        (Potentiometer){
            .degrees_of_motion = {120},
            .min_joint_angle_as_potentiometer_angle = {0},
            .max_joint_angle_as_potentiometer_angle = {120},
            .min_joint_angle = {0},
            .max_joint_angle = {120},
            .min_adc_value = 1035,
            .max_adc_value = 2957,
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

  while (true) {
    for (int i = 0; i < LIMB_ARR_LEN(stops); ++i) {
      servo_move_to_degree(&servo_config, stops[i]);
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }
}
