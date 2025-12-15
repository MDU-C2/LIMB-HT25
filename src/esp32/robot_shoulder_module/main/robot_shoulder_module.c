#include "adc_manager.h"
#include "can_driver.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "imu.h"
#include "potentiometer.h"
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
  IMU_SDA_GPIO = GPIO_NUM_6,
  IMU_SCL_GPIO = GPIO_NUM_7,
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

static const AdcMgrChannelConfig kAdcMgrChannelConfigs[] = {
    (AdcMgrChannelConfig){
        .channel = POTENTIOMETER_UP_DOWN_CHANNEL,
        .sample_rate = 1000,
    },
    (AdcMgrChannelConfig){
        .channel = POTENTIOMETER_LEFT_RIGHT_CHANNEL,
        .sample_rate = 1000,
    },
};

static const AdcMgrConfig kAdcMgrConfig = {
    .channel_configs = kAdcMgrChannelConfigs,
    .channel_configs_len = LIMB_ARR_LEN(kAdcMgrChannelConfigs),
    .ms_worth_of_buffer_size = 100,
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static uint16_t s_potentiometer_up_down_underlying_buffer[1024] = {0};
static uint16_t s_potentiometer_left_right_underlying_buffer[1024] = {0};

static AdcMgrReadResults s_adc_read_results = {
    .channel_buffers =
        {
            [POTENTIOMETER_UP_DOWN_CHANNEL] =
                {
                    .data = s_potentiometer_up_down_underlying_buffer,
                    .capacity =
                        LIMB_ARR_LEN(s_potentiometer_up_down_underlying_buffer),
                },
            [POTENTIOMETER_LEFT_RIGHT_CHANNEL] =
                {
                    .data = s_potentiometer_left_right_underlying_buffer,
                    .capacity = LIMB_ARR_LEN(
                        s_potentiometer_left_right_underlying_buffer),
                },
        },
};

static AdcMgrChannelBuffer* s_potentiometer_up_down_buffer =
    &s_adc_read_results.channel_buffers[POTENTIOMETER_UP_DOWN_CHANNEL];
static AdcMgrChannelBuffer* s_potentiometer_left_right_buffer =
    &s_adc_read_results.channel_buffers[POTENTIOMETER_LEFT_RIGHT_CHANNEL];

static uint16_t s_latest_potentiometer_up_down_value = 0;
static uint16_t s_latest_potentiometer_left_right_value = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static const Potentiometer kUpDownPotentiometer = {
    .degrees_of_motion = {285},
    // TODO(Johan): These are temporary test values. The proper values need to
    // be measured on the actual robot.
    .min_adc_value = 0,
    .max_adc_value = 4095,
    .min_joint_angle_as_potentiometer_angle = {(285.F / 2.F) - 90.F},
    .max_joint_angle_as_potentiometer_angle = {(285.F / 2.F) + 90.F},
    .min_joint_angle = {0},
    .max_joint_angle = {180},
};

static const Potentiometer kLeftRightPotentiometer = {
    .degrees_of_motion = {285},
    // TODO(Johan): These are temporary test values. The proper values need to
    // be measured on the actual robot.
    .min_adc_value = 0,
    .max_adc_value = 4095,
    .min_joint_angle_as_potentiometer_angle = {(285.F / 2.F) - 90.F},
    .max_joint_angle_as_potentiometer_angle = {(285.F / 2.F) + 90.F},
    .min_joint_angle = {0},
    .max_joint_angle = {180},
};

static uint16_t average(const uint16_t* values, const uint16_t value_len) {
  uint32_t sum = 0;

  for (uint16_t i = 0; i < value_len; ++i) {
    sum += values[i];
  }

  return sum / value_len;
}

static void adc_read_task([[maybe_unused]] void* arg) {
  // We're only reading potentiometer ADC values. To make the values more
  // stable, we want to average the last couple of values. To do that, we have
  // a higher sample rate than the frequency we actually want to send the data
  // at.
  enum {
    CAN_POTENTIOMETER_MSG_FREQUENCY_MS = 10,
  };

  TickType_t current_tick = xTaskGetTickCount();
  ESP_LOGI(TAG, "adc_read_task started!");

  while (true) {
    xTaskDelayUntil(&current_tick,
                    pdMS_TO_TICKS(CAN_POTENTIOMETER_MSG_FREQUENCY_MS));

    {
      esp_err_t err = adc_mgr_read(&s_adc_read_results, 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error calling adc_mgr_read: %s", esp_err_to_name(err));
        continue;
      }
    }

    // Get the average of the potentiometer values.
    if (s_potentiometer_up_down_buffer->length > 0) {
      s_latest_potentiometer_up_down_value =
          average(s_potentiometer_up_down_buffer->data,
                  s_potentiometer_up_down_buffer->length);
      s_potentiometer_up_down_buffer->length = 0;
    }

    if (s_potentiometer_left_right_buffer->length > 0) {
      s_latest_potentiometer_left_right_value =
          average(s_potentiometer_left_right_buffer->data,
                  s_potentiometer_left_right_buffer->length);
      s_potentiometer_left_right_buffer->length = 0;
    }

    // Send up/down potentiometer status update.
    {
      PotentiometerAngle potentiometer_angle = potentiometer_adc_to_angle(
          &kUpDownPotentiometer, s_latest_potentiometer_up_down_value);
      JointAngle joint_angle =
          to_joint_angle(&kUpDownPotentiometer, potentiometer_angle);
      esp_err_t err =
          can_send(CAN_ID_ROBOT_SHOULDER_UP_DOWN_POTENTIOMETER,
                   (uint8_t*)&joint_angle.degree, sizeof(joint_angle.degree));
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error sending shoulder up/down angle over CAN: %s",
                 esp_err_to_name(err));
      }
    }

    // Send left/right potentiometer status update.
    {
      PotentiometerAngle potentiometer_angle = potentiometer_adc_to_angle(
          &kLeftRightPotentiometer, s_latest_potentiometer_left_right_value);
      JointAngle joint_angle =
          to_joint_angle(&kLeftRightPotentiometer, potentiometer_angle);
      esp_err_t err =
          can_send(CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_POTENTIOMETER,
                   (uint8_t*)&joint_angle.degree, sizeof(joint_angle.degree));
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error sending shoulder left/right angle over CAN: %s",
                 esp_err_to_name(err));
      }
    }
  }

  vTaskDelete(NULL);
}

void app_main(void) {
  {
    esp_err_t err = servos_init(kServoConfigs, LIMB_ARR_LEN(kServoConfigs));
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling servos_init: %s", esp_err_to_name(err));
      return;
    }
  }

  vTaskDelay(pdMS_TO_TICKS(5000));
  servo_move_to_degree(kUpDownServo, kUpDownServo->max_angle);
  vTaskDelay(pdMS_TO_TICKS(1000));
  servo_move_to_degree(kUpDownServo, kUpDownServo->min_angle);

  {
    imu_config_t imu_config = IMU_CONFIG_DEFAULT();
    imu_config.sda_pin = IMU_SDA_GPIO;
    imu_config.scl_pin = IMU_SCL_GPIO;

    esp_err_t err = imu_init(&imu_config);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling imu_init: %s", esp_err_to_name(err));
      return;
    }

    if (!imu_is_present()) {
      ESP_LOGE(TAG, "Error: IMU sensor isn't present");
      return;
    }

    ESP_LOGI(TAG, "IMU initialized!");
  }

  {
    CanMsgFilter can_filter = {
        .id = CAN_RECIPIENT_ROBOT_SHOULDER,
        .ignore_mask = create_filter_mask(CAN_MESSAGE_TYPE_FILTER_ANY,
                                          CAN_RECIPIENT_FILTER_EXACT,
                                          CAN_GENERIC_FILTER_ANY),
    };

    esp_err_t err = can_init(CAN_TX_GPIO, CAN_RX_GPIO, 1000000, &can_filter);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling can_init: %s", esp_err_to_name(err));
      return;
    }
  }

  {
    esp_err_t err = adc_mgr_init(kAdcMgrConfig);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling adc_mgr_init: %s", esp_err_to_name(err));
      return;
    }

    err = adc_mgr_read(&s_adc_read_results, 10);
    ESP_LOGI(TAG, "Latest up/down value: %d",
             s_potentiometer_up_down_buffer
                 ->data[s_potentiometer_up_down_buffer->length - 1]);
    s_potentiometer_up_down_buffer->length = 0;
    ESP_LOGI(TAG, "Latest left/right value: %d",
             s_potentiometer_left_right_buffer
                 ->data[s_potentiometer_left_right_buffer->length - 1]);
    s_potentiometer_left_right_buffer->length = 0;
  }

  xTaskCreate(adc_read_task, "ADC read task", 1024 * 2 * 2, NULL, 5, NULL);
}
