#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "hal/ledc_types.h"
#include "servo.h"

#define LIMB_ARR_LEN(arr) (sizeof(arr) / sizeof(*(arr)))

enum {
  // NOTE: All of these should be measured at the maximum and minimum extents of
  // the motor actuations you want to support. These values aren't accurate.
  HV2060_MIN_PULSEWIDTH_US = 850,
  HV2060_MAX_PULSEWIDTH_US = 2150,
  HV2060_MIN_ANGLE = 0,
  HV2060_MAX_ANGLE = 180,
};

const ServoConfig servo_configs[] = {
    (ServoConfig){
        .gpio_pin = GPIO_NUM_0,
        .direction = SERVO_DIR_NORMAL,
        .initial_angle = SERVO_DIR_NORMAL,
        .ledc_channel = LEDC_CHANNEL_0,
        .min_angle = HV2060_MIN_ANGLE,
        .max_angle = HV2060_MAX_ANGLE,
        .min_pulse_us = HV2060_MIN_PULSEWIDTH_US,
        .max_pulse_us = HV2060_MAX_PULSEWIDTH_US,
        .name = "hv2060",
    },
};

void app_main(void) {
  ESP_ERROR_CHECK(servos_init(servo_configs, LIMB_ARR_LEN(servo_configs)));
  int stops[] = {
      0, 10, 45, 90, 180, 135, 90, 45,
  };

  while (true) {
    for (int i = 0; i < LIMB_ARR_LEN(stops); ++i) {
      servo_move_to_degree(&servo_configs[0], stops[i]);
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }
}
