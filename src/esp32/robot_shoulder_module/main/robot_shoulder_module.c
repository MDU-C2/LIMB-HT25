#include "adc_manager.h"
#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "imu.h"
#include "limb_utils.h"
#include "portmacro.h"
#include "potentiometer.h"
#include "servo.h"
#include "soc/gpio_num.h"

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
  CAN_TX_GPIO = GPIO_NUM_6,
  CAN_RX_GPIO = GPIO_NUM_7,
  IMU_SDA_GPIO = GPIO_NUM_5,
  IMU_SCL_GPIO = GPIO_NUM_4,
};

// Motors are hv2060

static const ServoConfig kUpDownServoConfig = {
    .gpio_pin = SERVO_UP_DOWN_GPIO,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .name = "Shoulder up/down servo",
    // TODO(johan): These need to be changed after testing on actual arm.
    .direction = SERVO_DIR_NORMAL,
    .motionless_pw = 1500,
    .max_capable_angular_velocity = {400},
    // FIXME: Figure out what to set this to. With no load the servo maxes out
    // at an offset of about 140-150, but it might be different with load.
    .max_capable_angular_velocity_pw_offset = 800,
    // This velocity was roughly eyeballed when the servo was moving at its
    // fastest pace with no load.
    .min_capable_angular_velocity = {34.F},
    // This is the lowest pulse width offset that results in the servo actually
    // moving.
    .min_capable_angular_velocity_pw_offset = 17,
    // FIXME: Figure out good values for these.
    .max_angular_velocity = {60.F},
    .max_angular_acceleration = {200.F},
    .pot_adc_channel = POTENTIOMETER_UP_DOWN_CHANNEL,
    .potentiometer =
        (Potentiometer){
            .degrees_of_motion = {285.F},
            .min_adc_value = 6,
            .max_adc_value = 3087,
            // FIXME: These need to be calibrated.
            .min_potentiometer_angle = {20},
            .max_potentiometer_angle = {70},
            .min_potentiometer_angle_as_joint_angle = {0.F},
            .joint_angle_to_potentiometer_angle_ratio = 1.F,
        },
};

static const ServoConfig kLeftRightServoConfig = {
    .gpio_pin = SERVO_LEFT_RIGHT_GPIO,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_1,
    .name = "Shoulder left/right servo",
    // TODO(johan): These need to be changed after testing on actual arm.
    .direction = SERVO_DIR_NORMAL,
    .motionless_pw = 1500,
    .max_capable_angular_velocity = {400},
    // FIXME: Figure out what to set this to. With no load the servo maxes out
    // at an offset of about 140-150, but it might be different with load.
    .max_capable_angular_velocity_pw_offset = 800,
    // FIXME: All of the below values need to be calibrated. Different servo (of
    // the same model) that might have slightly different characteristics, and
    // also different movement.
    // This velocity was roughly eyeballed when the
    // servo was moving at its fastest pace with no load.
    .min_capable_angular_velocity = {34.F},
    // This is the lowest pulse width offset that results in the servo actually
    // moving.
    .min_capable_angular_velocity_pw_offset = 17,
    // FIXME: Figure out good values for these.
    .max_angular_velocity = {60.F},
    .max_angular_acceleration = {200.F},
    .pot_adc_channel = POTENTIOMETER_UP_DOWN_CHANNEL,
    .potentiometer =
        (Potentiometer){
            .degrees_of_motion = {285.F},
            .min_adc_value = 6,
            .max_adc_value = 3087,
            // FIXME: These need to be calibrated.
            .min_potentiometer_angle = {20},
            .max_potentiometer_angle = {70},
            .min_potentiometer_angle_as_joint_angle = {0.F},
            .joint_angle_to_potentiometer_angle_ratio = 1.F,
        },
};

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

static ServoHandle s_left_right_servo_handle;
static ServoHandle s_up_down_servo_handle;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static void can_rx_task([[maybe_unused]] void* arg) {
  uint32_t can_id = 0;
  uint8_t can_buf[CAN_MAX_MESSAGE_SIZE] = {0};
  uint8_t can_buf_len = 0;

  while (true) {
    esp_err_t err = can_receive(&can_id, can_buf, &can_buf_len, portMAX_DELAY);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "Error calling can_receive: %s", esp_err_to_name(err));
      continue;
    }

    switch (can_id) {
      case CAN_ID_ROBOT_SHOULDER_UP_DOWN_STOP: {
        servo_set_target_angle(s_up_down_servo_handle, (JointAngle){0});
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_STOP: {
        servo_set_target_angle(s_left_right_servo_handle, (JointAngle){0});
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION: {
        assert(can_buf_len == 4);
        JointAngle joint_angle = {*(float*)can_buf};
        ESP_LOGI(TAG, "Received up/down joint angle %f", joint_angle.degree);
        servo_set_target_angle(s_up_down_servo_handle, joint_angle);
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION: {
        assert(can_buf_len == 4);
        JointAngle joint_angle = {*(float*)can_buf};
        ESP_LOGI(TAG, "Received left/right joint angle %f", joint_angle.degree);
        servo_set_target_angle(s_left_right_servo_handle, joint_angle);
        break;
      }
      default: {
        ESP_LOGW(TAG, "Unknown CAN Message received: 0x%x", can_id);
        break;
      }
    }
  }

  vTaskDelete(NULL);
}

static void imu_read_task([[maybe_unused]] void* arg) {
  enum {
    IMU_FREQUENCY_MS = 10,
    CAN_LEN = 3,
  };

  TickType_t current_tick = xTaskGetTickCount();
  ESP_LOGI(TAG, "imu_read_task started!");

  uint16_t imu_xyz_buf[3] = {0};

  while (true) {
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(IMU_FREQUENCY_MS));
    imu_data_t data = {0};
    {
      esp_err_t err = imu_read_data(&data);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error calling imu_read_data: %s", esp_err_to_name(err));
        continue;
      }
    }

    {
      imu_xyz_buf[0] = data.gyro.x;
      imu_xyz_buf[1] = data.gyro.y;
      imu_xyz_buf[2] = data.gyro.z;
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_IMU_GYRO,
                               (uint8_t*)imu_xyz_buf, sizeof(imu_xyz_buf), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error calling can_send with IMU gyro: %s",
                 esp_err_to_name(err));
      }
    }

    {
      imu_xyz_buf[0] = data.accel.x;
      imu_xyz_buf[1] = data.accel.y;
      imu_xyz_buf[2] = data.accel.z;
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_IMU_ACCEL,
                               (uint8_t*)imu_xyz_buf, sizeof(imu_xyz_buf), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error calling can_send with IMU accel: %s",
                 esp_err_to_name(err));
      }
    }
  }

  vTaskDelete(NULL);
}

static void servos_update_task([[maybe_unused]] void* args) {
  enum { PERIOD_MS = 10 };

  TickType_t current_tick = xTaskGetTickCount();

  while (true) {
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(PERIOD_MS));

    // Read ADC.
    esp_err_t err = adc_mgr_read(&s_adc_read_results, 0);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "Error calling adc_mgr_read: %s", esp_err_to_name(err));
    }

    s_latest_potentiometer_up_down_value =
        moving_average16(s_latest_potentiometer_up_down_value,
                         s_potentiometer_up_down_buffer->data,
                         s_potentiometer_up_down_buffer->length);
    s_potentiometer_up_down_buffer->length = 0;

    s_latest_potentiometer_left_right_value =
        moving_average16(s_latest_potentiometer_left_right_value,
                         s_potentiometer_left_right_buffer->data,
                         s_potentiometer_left_right_buffer->length);
    s_potentiometer_left_right_buffer->length = 0;

    // Update servos.
    servo_update(s_up_down_servo_handle, PERIOD_MS,
                 s_latest_potentiometer_up_down_value);
    servo_update(s_left_right_servo_handle, PERIOD_MS,
                 s_latest_potentiometer_left_right_value);

    // FIXME: Figure out how often we want to send the status updates.
    // Send up/down potentiometer status update.
    {
      static int i = 0;
      PotentiometerAngle potentiometer_angle =
          potentiometer_adc_to_angle(&kUpDownServoConfig.potentiometer,
                                     s_latest_potentiometer_up_down_value);
      JointAngle joint_angle = to_joint_angle(&kUpDownServoConfig.potentiometer,
                                              potentiometer_angle);
      if (++i == 100) {
        ESP_LOGI(TAG, "Current up/down pot angle: %f",
                 potentiometer_angle.degree);
        ESP_LOGI(TAG, "Current up/down joint angle: %f", joint_angle.degree);
        i = 0;
      }
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_UP_DOWN_POTENTIOMETER,
                               (uint8_t*)&joint_angle.degree,
                               sizeof(joint_angle.degree), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error sending shoulder up/down angle over CAN: %s",
                 esp_err_to_name(err));
      }
    }

    // Send left/right potentiometer status update.
    {
      PotentiometerAngle potentiometer_angle =
          potentiometer_adc_to_angle(&kLeftRightServoConfig.potentiometer,
                                     s_latest_potentiometer_left_right_value);
      JointAngle joint_angle = to_joint_angle(
          &kLeftRightServoConfig.potentiometer, potentiometer_angle);
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_POTENTIOMETER,
                               (uint8_t*)&joint_angle.degree,
                               sizeof(joint_angle.degree), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error sending shoulder left/right angle over CAN: %s",
                 esp_err_to_name(err));
      }
    }
  }

  vTaskDelete(NULL);
}

void app_main(void) {
  // IMU initialization.
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

  // CAN initialization.
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

  // ADC initialization.
  {
    esp_err_t err = adc_mgr_init(kAdcMgrConfig);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling adc_mgr_init: %s", esp_err_to_name(err));
      return;
    }
  }

  // Servo initialization.
  {
    vTaskDelay(pdMS_TO_TICKS(10));
    esp_err_t err = adc_mgr_read(&s_adc_read_results, 0);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling adc_mgr_read: %s", esp_err_to_name(err));
      return;
    }

    s_latest_potentiometer_up_down_value =
        limb_average16(s_potentiometer_up_down_buffer->data,
                       s_potentiometer_up_down_buffer->length);
    s_potentiometer_up_down_buffer->length = 0;
    s_latest_potentiometer_left_right_value =
        limb_average16(s_potentiometer_left_right_buffer->data,
                       s_potentiometer_left_right_buffer->length);
    s_potentiometer_left_right_buffer->length = 0;

    err = servo_init(&kUpDownServoConfig, s_latest_potentiometer_up_down_value,
                     &s_up_down_servo_handle);
    s_potentiometer_up_down_buffer->length = 0;
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling servos_init for up/down servo: %s",
               esp_err_to_name(err));
      return;
    }

    err = servo_init(&kLeftRightServoConfig,
                     s_latest_potentiometer_left_right_value,
                     &s_left_right_servo_handle);
    s_potentiometer_left_right_buffer->length = 0;
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling servos_init for left/right servo: %s",
               esp_err_to_name(err));
      return;
    }
  }

  xTaskCreate(can_rx_task, "CAN rx task", 1024 * 2 * 2, NULL, 5, NULL);
  xTaskCreate(imu_read_task, "IMU read task", 1024 * 2 * 2, NULL, 5, NULL);
  xTaskCreate(servos_update_task, "Servos update task", 1024 * 2 * 2, NULL, 6,
              NULL);
}
