#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "servo.h"
#include "soc/gpio_num.h"

#define LIMB_ARR_LEN(x) (sizeof(x) / sizeof(*(x)))

static const char* const TAG = "Shoulder module";

enum {
  // NOTE: All of these should be measured at the maximum and minimum extents of
  // the motor actuations you want to support. These values aren't accurate.
  HV2060_MIN_PULSEWIDTH_US = 850,
  HV2060_MAX_PULSEWIDTH_US = 2150,
};

enum {
  SERVO_UP_DOWN_GPIO = GPIO_NUM_0,
  // The channel number corresponds to the GPIO number.
  POTENTIOMETER_UP_DOWN_CHANNEL = ADC_CHANNEL_1,
  SERVO_LEFT_RIGHT_GPIO = GPIO_NUM_2,
  // The channel number corresponds to the GPIO number.
  POTENTIOMETER_LEFT_RIGHT_CHANNEL = ADC_CHANNEL_3,
  CAN_TX_GPIO = GPIO_NUM_5,
  CAN_RX_GPIO = GPIO_NUM_4,
};

// Motors are hv2060
static const ServoConfig kServoConfigs[] = {
    (ServoConfig){
        .gpio_pin = SERVO_UP_DOWN_GPIO,
        .ledc_channel = LEDC_CHANNEL_0,
        .name = "Shoulder up/down servo",
        // TODO(johan): These need to be changed after testing on actual arm.
        .direction = SERVO_DIR_NORMAL,
        .min_angle = 285.F / 2.F - 90.F,
        .max_angle = 285.F / 2.F + 90.F,
        .min_pulse_us = HV2060_MIN_PULSEWIDTH_US,
        .max_pulse_us = HV2060_MAX_PULSEWIDTH_US,
        .initial_angle = 285.F / 2.F,
    },
    (ServoConfig){
        .gpio_pin = SERVO_LEFT_RIGHT_GPIO,
        .ledc_channel = LEDC_CHANNEL_1,
        .name = "Shoulder left/right servo",
        // TODO(johan): These need to be changed after testing on actual arm.
        .direction = SERVO_DIR_NORMAL,
        .min_angle = 285.F / 2.F - 90.F,
        .max_angle = 285.F / 2.F + 90.F,
        .min_pulse_us = HV2060_MIN_PULSEWIDTH_US,
        .max_pulse_us = HV2060_MAX_PULSEWIDTH_US,
        .initial_angle = 285.F / 2.F,
    },
};

static const ServoConfig* const kUpDownServo = &kServoConfigs[0];
static const ServoConfig* const kLeftRightServo = &kServoConfigs[1];

void app_main(void) {
  esp_err_t err = servos_init(kServoConfigs, LIMB_ARR_LEN(kServoConfigs));
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Error calling servos_init: %s", esp_err_to_name(err));
    return;
  }

  vTaskDelay(pdMS_TO_TICKS(5000));
  servo_move_to_degree(kUpDownServo, kUpDownServo->max_angle);
  vTaskDelay(pdMS_TO_TICKS(1000));
  servo_move_to_degree(kUpDownServo, kUpDownServo->min_angle);
}
