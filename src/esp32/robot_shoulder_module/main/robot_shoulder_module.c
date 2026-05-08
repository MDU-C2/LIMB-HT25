#include "adc_manager.h"
#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "portmacro.h"
#include "potentiometer.h"
#include "servo.h"
#include "soc/gpio_num.h"
#include "stepper.h"

static const char* const TAG = "Shoulder module";

enum {
  // NOTE: All of these should be measured at the maximum and minimum extents of
  // the motor actuations you want to support. These values aren't accurate.
  HV2060_MIN_PULSEWIDTH_US = 850,
  HV2060_MAX_PULSEWIDTH_US = 2150,
};

enum {
  SERVO_UP_DOWN_GPIO = GPIO_NUM_3,
  // The channel number corresponds to the GPIO number.
  POTENTIOMETER_UP_DOWN_CHANNEL = ADC_CHANNEL_2,

  SERVO_LEFT_RIGHT_GPIO = GPIO_NUM_4,
  // The channel number corresponds to the GPIO number.
  POTENTIOMETER_LEFT_RIGHT_CHANNEL = ADC_CHANNEL_1,

  // The channel number corresponds to the GPIO number.
  POTENTIOMETER_ROTATION_CHANNEL = ADC_CHANNEL_0,
  STEPPER_UPPER_ARM_ROTATION_ENABLE_GPIO = GPIO_NUM_NC,
  STEPPER_UPPER_ARM_ROTATION_DIR_GPIO = GPIO_NUM_10,
  STEPPER_UPPER_ARM_ROTATION_STEP_GPIO = GPIO_NUM_7,
  // Microstepping pins.
  // NOTE: Since we're setting the stepper driver's pins in the hardware, we
  // don't need to use these pins in software.
  STEPPER_UPPER_ARM_ROTATION_M0_GPIO = GPIO_NUM_NC,
  STEPPER_UPPER_ARM_ROTATION_M1_GPIO = GPIO_NUM_NC,
  STEPPER_UPPER_ARM_ROTATION_M2_GPIO = GPIO_NUM_NC,
  CAN_TX_GPIO = GPIO_NUM_6,
  CAN_RX_GPIO = GPIO_NUM_5,
};

// Motors are hv2060

static const ServoConfig kUpDownServoConfig = {
    .gpio_pin = SERVO_UP_DOWN_GPIO,
    .pwm_timer = LEDC_TIMER_0,
    .pwm_channel = LEDC_CHANNEL_0,
    .name = "Shoulder up/down servo",
    // TODO(johan): These need to be changed after testing on actual arm.
    .direction = SERVO_DIR_REVERSE,
    .motionless_pw = 1500,
    .max_capable_angular_velocity = {400},
    .max_capable_angular_velocity_pw_offset = 150,
    .gear_ratio = 15.F,
    .max_velocity_positive = {10.F},
    .max_velocity_negative = {20.F},
    .max_accel = {15.F},
    .pot_adc_channel = POTENTIOMETER_UP_DOWN_CHANNEL,
    .potentiometer =
        (Potentiometer){
            .range_of_motion = {285.F},
            // We step down voltage we feed to the potentiometer from 3300 mV to
            // 2200 mV.
            .min_adc_value = 0,
            .max_adc_value = 2200,
            // should be [0,90], but is actually like [0,75]
            .min_potentiometer_angle = {145},
            .max_potentiometer_angle = {235},
            .min_potentiometer_angle_as_joint_angle = {0.F},
            .joint_angle_to_potentiometer_angle_ratio = 1.F,
            .is_reversed = true,
        },
};

static const ServoConfig kLeftRightServoConfig = {
    .gpio_pin = SERVO_LEFT_RIGHT_GPIO,
    .pwm_timer = LEDC_TIMER_0,
    .pwm_channel = LEDC_CHANNEL_1,
    .name = "Shoulder left/right servo",
    // TODO(johan): These need to be changed after testing on actual arm.
    .direction = SERVO_DIR_NORMAL,
    .motionless_pw = 1500,
    .max_capable_angular_velocity = {400},
    .max_capable_angular_velocity_pw_offset = 150,
    .gear_ratio = 15.F,
    .max_velocity_positive = {20.F},
    .max_velocity_negative = {10.F},
    .max_accel = {15.F},
    .pot_adc_channel = POTENTIOMETER_LEFT_RIGHT_CHANNEL,
    .potentiometer =
        (Potentiometer){
            .range_of_motion = {285.F},
            // We step down voltage we feed to the potentiometer from 3300 mV to
            // 2200 mV.
            .min_adc_value = 0,
            .max_adc_value = 2200,
            // Corresponds to between 5 and 40 joint degrees.
            .min_potentiometer_angle = {86},
            .max_potentiometer_angle = {128},
            .min_potentiometer_angle_as_joint_angle = {5.F},
            .joint_angle_to_potentiometer_angle_ratio = 18.F / 15.F,
        },
};

// FIXME: All these need to be configured properly.
static const stepper_control_config_t kUpperArmRotationStepperConfig = {
    .enable_gpio = STEPPER_UPPER_ARM_ROTATION_ENABLE_GPIO,
    .dir_gpio = STEPPER_UPPER_ARM_ROTATION_DIR_GPIO,
    .step_gpio = STEPPER_UPPER_ARM_ROTATION_STEP_GPIO,
    .microstepping_mode = MICROSTEP_1_32,
    .microstepping_type = MICROSTEP_HARDWARE,
    .microstep_m0_gpio = STEPPER_UPPER_ARM_ROTATION_M0_GPIO,
    .microstep_m1_gpio = STEPPER_UPPER_ARM_ROTATION_M1_GPIO,
    .microstep_m2_gpio = STEPPER_UPPER_ARM_ROTATION_M2_GPIO,
    .direction = STEPPER_DIR_NORMAL,
    .gear_ratio = 15.F,
    .max_velocity_negative = {40.F},
    .max_velocity_positive = {40.F},
    .max_accel = {20.F},
    .pot_adc_channel = POTENTIOMETER_ROTATION_CHANNEL,
    .pwm_channel = LEDC_CHANNEL_2,
    .pwm_timer = LEDC_TIMER_1,
    .steps_per_rev = 200,
    .potentiometer =
        {
            .range_of_motion = {285.F},
            // We step down voltage we feed to the potentiometer from 3300 mV to
            // 2200 mV.
            .min_adc_value = 0,
            .max_adc_value = 2200,
            // Red is Vin and black is ground.
            .min_potentiometer_angle = {40},
            .max_potentiometer_angle = {160},
            .min_potentiometer_angle_as_joint_angle = {-60.F},
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
    (AdcMgrChannelConfig){
        .channel = POTENTIOMETER_ROTATION_CHANNEL,
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
static uint16_t s_potentiometer_rotation_underlying_buffer[1024] = {0};

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
            [POTENTIOMETER_ROTATION_CHANNEL] =
                {
                    .data = s_potentiometer_rotation_underlying_buffer,
                    .capacity = LIMB_ARR_LEN(
                        s_potentiometer_rotation_underlying_buffer),
                },
        },
};

static AdcMgrChannelBuffer* s_potentiometer_up_down_buffer =
    &s_adc_read_results.channel_buffers[POTENTIOMETER_UP_DOWN_CHANNEL];
static AdcMgrChannelBuffer* s_potentiometer_left_right_buffer =
    &s_adc_read_results.channel_buffers[POTENTIOMETER_LEFT_RIGHT_CHANNEL];
static AdcMgrChannelBuffer* s_potentiometer_rotation_buffer =
    &s_adc_read_results.channel_buffers[POTENTIOMETER_ROTATION_CHANNEL];

static uint16_t s_latest_potentiometer_up_down_value = 0;
static uint16_t s_latest_potentiometer_left_right_value = 0;
static uint16_t s_latest_potentiometer_rotation_value = 0;

static ServoHandle s_left_right_servo_handle;
static ServoHandle s_up_down_servo_handle;
static stepper_control_handle_t s_upper_arm_rotation_stepper_handle;
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
        servo_set_estop(s_up_down_servo_handle, true);
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_STOP: {
        servo_set_estop(s_left_right_servo_handle, true);
        break;
      }
      case CAN_ID_ROBOT_UPPER_ARM_ROTATION_STOP: {
        stepper_set_estop(s_upper_arm_rotation_stepper_handle, true);
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION: {
        enum {
          kExpectedPayloadSize = 2 * sizeof(float),
        };
        if (can_buf_len != kExpectedPayloadSize) {
          ESP_LOGW(TAG,
                   "Invalid up/down actuation payload length: %u (expected %u)",
                   can_buf_len, kExpectedPayloadSize);
          break;
        }
        JointAngle joint_angle = {
            deserialize_float(can_buf, kFromLittleEndian)};
        AngularVelocity target_velocity = {
            deserialize_float(can_buf + sizeof(float), kFromLittleEndian)};
        ESP_LOGI(TAG, "Received up/down actuation: angle=%f, velocity=%f dps",
                 joint_angle.degree, target_velocity.dps);
        servo_set_target_angle(s_up_down_servo_handle, joint_angle);
        servo_set_target_velocity(s_up_down_servo_handle, target_velocity);
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION: {
        enum {
          kExpectedPayloadSize = 2 * sizeof(float),
        };
        if (can_buf_len != kExpectedPayloadSize) {
          ESP_LOGW(
              TAG,
              "Invalid left/right actuation payload length: %u (expected %u)",
              can_buf_len, kExpectedPayloadSize);
          break;
        }
        JointAngle joint_angle = {
            deserialize_float(can_buf, kFromLittleEndian)};
        AngularVelocity target_velocity = {
            deserialize_float(can_buf + sizeof(float), kFromLittleEndian)};
        ESP_LOGI(TAG,
                 "Received left/right actuation: angle=%f, velocity=%f dps",
                 joint_angle.degree, target_velocity.dps);
        servo_set_target_angle(s_left_right_servo_handle, joint_angle);
        servo_set_target_velocity(s_left_right_servo_handle, target_velocity);
        break;
      }
      case CAN_ID_ROBOT_UPPER_ARM_ROTATION_ACTUATION: {
        enum {
          kExpectedPayloadSize = 2 * sizeof(float),
        };
        if (can_buf_len != kExpectedPayloadSize) {
          ESP_LOGW(TAG,
                   "Invalid upper-arm rotation actuation payload length: %u "
                   "(expected %u)",
                   can_buf_len, kExpectedPayloadSize);
          break;
        }
        JointAngle joint_angle = {
            deserialize_float(can_buf, kFromLittleEndian)};
        AngularVelocity target_velocity = {
            deserialize_float(can_buf + sizeof(float), kFromLittleEndian)};
        ESP_LOGI(
            TAG,
            "Received upper-arm rotation actuation: angle=%f, velocity=%f dps",
            joint_angle.degree, target_velocity.dps);
        stepper_set_target_angle(s_upper_arm_rotation_stepper_handle,
                                 joint_angle);
        stepper_set_target_velocity(s_upper_arm_rotation_stepper_handle,
                                    target_velocity);
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

static void motors_update_task([[maybe_unused]] void* args) {
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

    s_latest_potentiometer_rotation_value =
        moving_average16(s_latest_potentiometer_rotation_value,
                         s_potentiometer_rotation_buffer->data,
                         s_potentiometer_rotation_buffer->length);
    s_potentiometer_rotation_buffer->length = 0;

    // Update servos.
    servo_update(s_up_down_servo_handle, PERIOD_MS,
                 s_latest_potentiometer_up_down_value);
    servo_update(s_left_right_servo_handle, PERIOD_MS,
                 s_latest_potentiometer_left_right_value);
    stepper_update(s_upper_arm_rotation_stepper_handle, PERIOD_MS,
                   s_latest_potentiometer_rotation_value);

    // Send up/down potentiometer status update.
    {
      PotentiometerAngle potentiometer_angle =
          potentiometer_adc_to_angle(&kUpDownServoConfig.potentiometer,
                                     s_latest_potentiometer_up_down_value);
      JointAngle joint_angle = to_joint_angle(&kUpDownServoConfig.potentiometer,
                                              potentiometer_angle);
      static int i = 0;
      if (++i == 100) {
        i = 0;
        PotentiometerAngle target_pot_angle =
            servo_get_target_angle(s_up_down_servo_handle);
        JointAngle target_joint_angle =
            to_joint_angle(&kUpDownServoConfig.potentiometer, target_pot_angle);
        AngularVelocity velocity =
            servo_get_current_velocity(s_up_down_servo_handle);
        ESP_LOGI(TAG,
                 "up/down: adc=%u, curr pot=%.2f, target pot=%.2f, curr "
                 "joint=%.2f, target joint=%.2f, velocity: %.2f",
                 s_latest_potentiometer_up_down_value,
                 potentiometer_angle.degree, target_pot_angle.degree,
                 joint_angle.degree, target_joint_angle.degree, velocity.dps);
      }

      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_UP_DOWN_POTENTIOMETER,
                               (uint8_t*)&joint_angle.degree,
                               sizeof(joint_angle.degree), 0);
      static uint32_t can_errors_count = 0;
      if (err != ESP_OK) {
        if (can_errors_count++ % 100 == 0) {
          ESP_LOGW(TAG,
                   "Error sending shoulder up/down angle over CAN: %s, total "
                   "error count: %u",
                   esp_err_to_name(err), can_errors_count);
        }
      }
    }

    // Send left/right potentiometer status update.
    {
      PotentiometerAngle potentiometer_angle =
          potentiometer_adc_to_angle(&kLeftRightServoConfig.potentiometer,
                                     s_latest_potentiometer_left_right_value);
      JointAngle joint_angle = to_joint_angle(
          &kLeftRightServoConfig.potentiometer, potentiometer_angle);

      static int i = 33;
      if (++i == 100) {
        i = 0;
        PotentiometerAngle target_pot_angle =
            servo_get_target_angle(s_left_right_servo_handle);
        JointAngle target_joint_angle = to_joint_angle(
            &kLeftRightServoConfig.potentiometer, target_pot_angle);
        AngularVelocity velocity =
            servo_get_current_velocity(s_left_right_servo_handle);
        ESP_LOGI(TAG,
                 "left/right: adc=%u, curr pot=%.2f, target pot=%.2f, curr "
                 "joint=%.2f, target joint=%.2f, velocity: %.2f",
                 s_latest_potentiometer_left_right_value,
                 potentiometer_angle.degree, target_pot_angle.degree,
                 joint_angle.degree, target_joint_angle.degree, velocity.dps);
      }

      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_POTENTIOMETER,
                               (uint8_t*)&joint_angle.degree,
                               sizeof(joint_angle.degree), 0);
      static uint32_t can_errors_count = 0;
      if (err != ESP_OK) {
        if (can_errors_count++ % 100 == 0) {
          ESP_LOGW(TAG,
                   "Error sending shoulder left/right angle over CAN: %s, "
                   "total error count: %u",
                   esp_err_to_name(err), can_errors_count);
        }
      }
    }

    // Send upper arm rotation potentiometer status update.
    {
      PotentiometerAngle potentiometer_angle = potentiometer_adc_to_angle(
          &kUpperArmRotationStepperConfig.potentiometer,
          s_latest_potentiometer_rotation_value);
      JointAngle joint_angle = to_joint_angle(
          &kUpperArmRotationStepperConfig.potentiometer, potentiometer_angle);

      static uint32_t i = 66;
      if (++i >= 100) {  // Every 100ms
        i = 0;
        PotentiometerAngle target_pot_angle =
            stepper_get_target_angle(s_upper_arm_rotation_stepper_handle);
        JointAngle target_angle = to_joint_angle(
            &kUpperArmRotationStepperConfig.potentiometer, target_pot_angle);
        AngularVelocity velocity =
            stepper_get_current_velocity(s_upper_arm_rotation_stepper_handle);
        bool moving = stepper_is_moving(s_upper_arm_rotation_stepper_handle);

        ESP_LOGI(TAG,
                 "adc: %u, Stepper upper arm rotation - Current(pot): %.2f°, "
                 "Target(pot): "
                 "%.2f°, Current(Joint): %.2f, Target(Joint): %.2f, Velocity: "
                 "%.2f°/s, Moving: %s",
                 s_latest_potentiometer_rotation_value,
                 potentiometer_angle.degree, target_pot_angle.degree,
                 joint_angle.degree, target_angle.degree, velocity.dps,
                 moving ? "Yes" : "No");
      }

      esp_err_t err = can_send(CAN_ID_ROBOT_UPPER_ARM_ROTATION_POTENTIOMETER,
                               (uint8_t*)&joint_angle.degree,
                               sizeof(joint_angle.degree), 0);
      static uint32_t can_errors_count = 0;
      if (err != ESP_OK) {
        if (can_errors_count++ % 100 == 0) {
          ESP_LOGW(TAG,
                   "Error sending upper arm rotation angle over CAN: %s, total "
                   " error count : %u",
                   esp_err_to_name(err), can_errors_count);
        }
      }
    }
  }

  vTaskDelete(NULL);
}

static void reenable_can_task([[maybe_unused]] void* pvParameter) {
  while (true) {
    can_automatically_reenable_on_bus_off();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

void app_main(void) {
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
      abort();
    }
  }

  // ADC initialization.
  {
    esp_err_t err = adc_mgr_init(kAdcMgrConfig);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling adc_mgr_init: %s", esp_err_to_name(err));
      abort();
    }
  }

  // Servo and stepper initialization.
  {
    vTaskDelay(pdMS_TO_TICKS(10));
    esp_err_t err = adc_mgr_read(&s_adc_read_results, 0);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling adc_mgr_read: %s", esp_err_to_name(err));
      abort();
    }

    s_latest_potentiometer_up_down_value =
        limb_average16(s_potentiometer_up_down_buffer->data,
                       s_potentiometer_up_down_buffer->length);
    s_potentiometer_up_down_buffer->length = 0;
    s_latest_potentiometer_left_right_value =
        limb_average16(s_potentiometer_left_right_buffer->data,
                       s_potentiometer_left_right_buffer->length);
    s_potentiometer_left_right_buffer->length = 0;
    s_latest_potentiometer_rotation_value =
        limb_average16(s_potentiometer_rotation_buffer->data,
                       s_potentiometer_rotation_buffer->length);
    s_potentiometer_rotation_buffer->length = 0;

    err = servo_init(&kUpDownServoConfig, s_latest_potentiometer_up_down_value,
                     &s_up_down_servo_handle);
    s_potentiometer_up_down_buffer->length = 0;
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling servos_init for up/down servo: %s",
               esp_err_to_name(err));
      abort();
    }

    err = servo_init(&kLeftRightServoConfig,
                     s_latest_potentiometer_left_right_value,
                     &s_left_right_servo_handle);
    s_potentiometer_left_right_buffer->length = 0;
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error calling servos_init for left/right servo: %s",
               esp_err_to_name(err));
      abort();
    }

    err = stepper_init(&kUpperArmRotationStepperConfig,
                       s_latest_potentiometer_rotation_value,
                       &s_upper_arm_rotation_stepper_handle);
    s_potentiometer_rotation_buffer->length = 0;
    if (err != ESP_OK) {
      ESP_LOGE(TAG,
               "Error calling stepper_init for upper arm rotation stepper: %s",
               esp_err_to_name(err));
      abort();
    }
  }

  {
    BaseType_t err =
        xTaskCreate(can_rx_task, "CAN rx task", 1024 * 2 * 2, NULL, 5, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create reenable_can_task, err code: %d");
      abort();
    }
  }

  {
    BaseType_t err = xTaskCreate(motors_update_task, "Motors update task",
                                 1024 * 2 * 2, NULL, 6, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create reenable_can_task, err code: %d");
      abort();
    }
  }

#if CONFIG_FORCE_REENABLE_CAN_ON_BUS_OFF
  {
    BaseType_t err = xTaskCreate(reenable_can_task, "reenable_can_task",
                                 1024 * 2 * 2, NULL, 6, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create reenable_can_task, err code: %d");
      abort();
    }
  }
#endif
}
