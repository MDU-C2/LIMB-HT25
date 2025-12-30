#include "adc_manager.h"
#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "imu.h"
#include "limb_utils.h"
#include "portmacro.h"
#include "potentiometer.h"
#include "soc/gpio_num.h"
#include "stepper.h"

static const char *const TAG = "robot_elbow_module";

enum {
  // CAN configuration
  CAN_TX_PIN = 5,
  CAN_RX_PIN = 4,
  CAN_BAUDRATE = 1000000,

  // IMU pins.
  IMU_SDA_PIN = GPIO_NUM_2,
  IMU_SCL_PIN = GPIO_NUM_3,

  // GPIO pin definitions
  STEPPER_ELBOW_STEP_PIN = GPIO_NUM_6,
  STEPPER_ELBOW_DIR_PIN = GPIO_NUM_7,
  STEPPER_ELBOW_ENABLE_PIN = GPIO_NUM_8,

  // ADC channels
  ADC_ELBOW_CHANNEL = ADC_CHANNEL_0,  // GPIO 0

  // PWM timers and channels
  PWM_ELBOW_CHANNEL = LEDC_CHANNEL_0,
  PWM_ELBOW_TIMER = LEDC_TIMER_0,
};

// Stepper configurations
const stepper_control_config_t s_elbow_stepper_cfg = {
    .step_gpio = STEPPER_ELBOW_STEP_PIN,
    .dir_gpio = STEPPER_ELBOW_DIR_PIN,
    .enable_gpio = STEPPER_ELBOW_ENABLE_PIN,

    // Set these if you want microstepping.
    .microstepping_mode = MICROSTEP_NONE,
    .microstep_m0_gpio = GPIO_NUM_NC,
    .microstep_m1_gpio = GPIO_NUM_NC,
    .microstep_m2_gpio = GPIO_NUM_NC,

    .direction = STEPPER_DIR_REVERSE,
    .steps_per_rev = 200,
    .gear_ratio = 15.0F,
    .max_velocity = {20.0F},
    .max_accel = {20.0F},
    .pot_adc_channel = ADC_ELBOW_CHANNEL,
    .pwm_channel = PWM_ELBOW_CHANNEL,
    .pwm_timer = PWM_ELBOW_TIMER,
    .potentiometer =
        (Potentiometer){
            .degrees_of_motion = {285.F},
            .min_adc_value = 282,
            .max_adc_value = 3130,
            .min_potentiometer_angle_as_joint_angle = {0.F},
            .min_potentiometer_angle = {80.F},
            // Using the ratio between the joint angles and potentiomter angles,
            // 60 degrees of joint angle becomes 72 degrees of potentiometer
            // angle.
            .max_potentiometer_angle = {152.F},
            .joint_angle_to_potentiometer_angle_ratio = 18.F / 15.F,
        },
};

// ADC manager configuration.
const AdcMgrChannelConfig s_adc_mgr_channel_configs[] = {
    {
        .channel = ADC_ELBOW_CHANNEL,
        .sample_rate = 1000,
    },
};

const AdcMgrConfig s_adc_mgr_config = {
    .channel_configs = s_adc_mgr_channel_configs,
    .channel_configs_len = LIMB_ARR_LEN(s_adc_mgr_channel_configs),
    .ms_worth_of_buffer_size = 100,
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
// Assuming a sample rate of 1 kHz, we can comfortably keep a buffer of 1
// seconds worth of data.
enum {
  ADC_STEPPERS_UNDERLYING_BUF_SIZE = 1000,
};

uint16_t
    s_adc_elbow_channel_underlying_buffer[ADC_STEPPERS_UNDERLYING_BUF_SIZE] = {
        0};

AdcMgrReadResults s_adc_mgr_read_results = {
    .channel_buffers = {
        [ADC_ELBOW_CHANNEL] = {.data = s_adc_elbow_channel_underlying_buffer,
                               .capacity = LIMB_ARR_LEN(
                                   s_adc_elbow_channel_underlying_buffer)},
    }};

AdcMgrChannelBuffer *s_adc_mgr_elbow_buffer =
    &s_adc_mgr_read_results.channel_buffers[ADC_ELBOW_CHANNEL];

uint16_t s_latest_potentiometer_adc_value;

stepper_control_handle_t s_elbow_stepper_handle = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

void can_rx_task([[maybe_unused]] void *pvParameter) {
  uint8_t msg_rx[CAN_MAX_MESSAGE_SIZE];
  uint8_t rx_len = CAN_MAX_MESSAGE_SIZE;
  uint32_t rx_id = 0;

  while (1) {
    esp_err_t err = can_receive(&rx_id, msg_rx, &rx_len, portMAX_DELAY);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "Error calling can_receive: %s", esp_err_to_name(err));
      continue;
    }

    switch (rx_id) {
      case CAN_ID_ROBOT_ELBOW_UP_DOWN_ACTUATION: {
        JointAngle target_angle = {*(float *)msg_rx};
        stepper_set_target_angle(s_elbow_stepper_handle, target_angle);
        ESP_LOGI(TAG, "Received command: elbow target angle = %f degrees",
                 target_angle.degree);
        break;
      }
      case CAN_ID_ROBOT_ELBOW_UP_DOWN_STOP: {
        stepper_set_estop(s_elbow_stepper_handle, true);
        ESP_LOGI(TAG, "Received stop command: 0x%x", rx_id);
        break;
      }
      default: {
        ESP_LOGW(TAG, "Received unknown CAN message: 0x%x", rx_id);
      }
    }
  }
}

void imu_task([[maybe_unused]] void *pvParameter) {
  imu_data_t imu_data;  // (imu_vector_t) accel and (imu_vector_t) gyro

  // The buffer is used for the xyz values of both the gyro and the accel
  // messages.
  uint16_t imu_can_msg_buf[3] = {0};

  TickType_t current_tick = xTaskGetTickCount();
  while (1) {
    // Period of 100 Hz / 10 ms.
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(10));

    // Read IMU data.
    {
      esp_err_t err = imu_read_data(&imu_data);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error reading IMU: %s", esp_err_to_name(err));
        continue;
      }
    }

    {
      imu_can_msg_buf[0] = imu_data.gyro.x;
      imu_can_msg_buf[1] = imu_data.gyro.y;
      imu_can_msg_buf[2] = imu_data.gyro.z;
      esp_err_t err =
          can_send(CAN_ID_ROBOT_ELBOW_IMU_GYRO, (uint8_t *)imu_can_msg_buf,
                   sizeof(imu_can_msg_buf), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error sending IMU gyro over CAN: %s",
                 esp_err_to_name(err));
      }
    }
    {
      imu_can_msg_buf[0] = imu_data.accel.x;
      imu_can_msg_buf[1] = imu_data.accel.y;
      imu_can_msg_buf[2] = imu_data.accel.z;
      esp_err_t err =
          can_send(CAN_ID_ROBOT_ELBOW_IMU_ACCEL, (uint8_t *)imu_can_msg_buf,
                   sizeof(imu_can_msg_buf), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error sending IMU accel over CAN: %s",
                 esp_err_to_name(err));
      }
    }
  }
}

void stepper_task([[maybe_unused]] void *pvParameter) {
  TickType_t last_wake_time = xTaskGetTickCount();
  const TickType_t period_ms = pdMS_TO_TICKS(10);  // 10ms = 100Hz update rate

  while (1) {
    const uint16_t dt_ms = 10;  // 10ms in seconds

    adc_mgr_read(&s_adc_mgr_read_results, 0);
    s_latest_potentiometer_adc_value = moving_average16(
        s_latest_potentiometer_adc_value, s_adc_mgr_elbow_buffer->data,
        s_adc_mgr_elbow_buffer->length);
    s_adc_mgr_elbow_buffer->length = 0;
    stepper_update(s_elbow_stepper_handle, dt_ms,
                   s_latest_potentiometer_adc_value);

    // Send status over CAN and log
    static uint32_t status_counter = 0;
    enum {
      ITERATIONS_PER_LOGGING = 10,
    };
    if (++status_counter >= ITERATIONS_PER_LOGGING) {  // Every 100ms
      status_counter = 0;
      PotentiometerAngle current_pot_angle =
          stepper_get_current_angle(s_elbow_stepper_handle);
      PotentiometerAngle target_pot_angle =
          stepper_get_target_angle(s_elbow_stepper_handle);
      JointAngle current_angle =
          to_joint_angle(&s_elbow_stepper_cfg.potentiometer, current_pot_angle);
      JointAngle target_angle =
          to_joint_angle(&s_elbow_stepper_cfg.potentiometer, target_pot_angle);
      AngularVelocity velocity =
          stepper_get_current_velocity(s_elbow_stepper_handle);
      bool moving = stepper_is_moving(s_elbow_stepper_handle);

      // Send status over CAN
      uint8_t can_data[CAN_MAX_MESSAGE_SIZE] = {0};
      *(float *)can_data = current_angle.degree;
      esp_err_t err = can_send(CAN_ID_ROBOT_ELBOW_UP_DOWN_POTENTIOMETER,
                               can_data, sizeof(current_angle.degree), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "Error sending elbow status over CAN: %s",
                 esp_err_to_name(err));
      }

      // Also log locally
      ESP_LOGI(TAG,
               "Stepper elbow - Current(pot): %.2f°, Target(pot): %.2f°, "
               "Current(Joint): %.2f, Target(Joint): %.2f, Velocity: %.2f°/s, "
               "Moving: %s",
               current_pot_angle.degree, target_pot_angle.degree,
               current_angle.degree, target_angle.degree, velocity.dps,
               moving ? "Yes" : "No");
    }

    xTaskDelayUntil(&last_wake_time, period_ms);
  }
}

// Test task to cycle through different target angles
void stepper_test_task([[maybe_unused]] void *pvParameter) {
  // Wait a bit for system to initialize
  vTaskDelay(pdMS_TO_TICKS(2000));

  // Test angles to cycle through (in degrees)
  const float test_angles[] = {0.0F, 60.F};
  int num_angles = sizeof(test_angles) / sizeof(test_angles[0]);
  int angle_index = 0;

  ESP_LOGI(TAG, "Stepper test task started - will cycle through test angles");

  while (1) {
    // Set new target angle
    float target = test_angles[angle_index];
    stepper_set_target_angle(s_elbow_stepper_handle, (JointAngle){target});
    ESP_LOGI(TAG, ">>> Setting target angle to %.1f°", target);

    TickType_t start_time = xTaskGetTickCount();

    // Allow motor to start moving.
    vTaskDelay(pdMS_TO_TICKS(500));

    TickType_t ticks_since_start_time = xTaskGetTickCount() - start_time;

    // Wait for stepper to reach target (or timeout after 5 seconds)
    while (stepper_is_moving(s_elbow_stepper_handle) &&
           (ticks_since_start_time < pdMS_TO_TICKS(15000))) {
      vTaskDelay(pdMS_TO_TICKS(100));
      ticks_since_start_time = xTaskGetTickCount() - start_time;
    }

    // Hold at this position for 2 seconds
    vTaskDelay(pdMS_TO_TICKS(2000));

    // Move to next angle
    angle_index = (angle_index + 1) % num_angles;
  }
}

void app_main(void) {
  ESP_LOGI(TAG, "Robot elbow module starting...");

  // Initialize CAN.
  {
    CanMsgFilter can_filter = {
        // We want to accept all messages that are sent with the elbow node as
        // the recipient.
        .id = CAN_RECIPIENT_ROBOT_ELBOW,
        .ignore_mask = create_filter_mask(CAN_MESSAGE_TYPE_FILTER_ANY,
                                          CAN_RECIPIENT_FILTER_EXACT,
                                          CAN_GENERIC_FILTER_ANY),
    };
    esp_err_t err = can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE, &can_filter);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Failed to initialize CAN driver: %s",
               esp_err_to_name(err));
      return;
    }
    ESP_LOGI(TAG, "CAN initialized (TX=%d, RX=%d, %d baud)", CAN_TX_PIN,
             CAN_RX_PIN, CAN_BAUDRATE);
  }

  // Initialize IMU.
  {
    imu_config_t imu_cfg = IMU_CONFIG_DEFAULT();
    imu_cfg.sda_pin = IMU_SDA_PIN;
    imu_cfg.scl_pin = IMU_SCL_PIN;
    esp_err_t err = imu_init(&imu_cfg);
    if (err) {
      ESP_LOGE(TAG, "Failed to initialize IMU driver: %s",
               esp_err_to_name(err));
      return;
    }
    ESP_LOGI(TAG, "IMU initialized (SDA=%d, SCL=%d)", imu_cfg.sda_pin,
             imu_cfg.scl_pin);
  }

  // Initialize ADC.
  {
    esp_err_t err = adc_mgr_init(s_adc_mgr_config);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Failed to initialize ADC manager: %s",
               esp_err_to_name(err));
      return;
    }
    ESP_LOGI(TAG, "ADC manager initialized");
  }

  // Wait for a bit to get initial results.
  vTaskDelay(pdMS_TO_TICKS(10));
  adc_mgr_read(&s_adc_mgr_read_results, 0);

  s_latest_potentiometer_adc_value = limb_average16(
      s_adc_mgr_elbow_buffer->data, s_adc_mgr_elbow_buffer->length);

  // Initialize stepper motor.
  {
    esp_err_t err =
        stepper_init(&s_elbow_stepper_cfg, s_latest_potentiometer_adc_value,
                     &s_elbow_stepper_handle);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Failed to initialize elbow stepper");
      return;
    }
  }
  s_adc_mgr_elbow_buffer->length = 0;

  // Create FreeRTOS tasks
  enum {
    TASK_STACK_DEPTH = 4096,
    TASK_HIGH_PRIORITY = 5,
    TASK_LOW_PRIORITY = 4,
  };

  {
    BaseType_t err = xTaskCreate(can_rx_task, "can_rx", TASK_STACK_DEPTH, NULL,
                                 TASK_HIGH_PRIORITY, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create can_rx task, err code: %d");
      return;
    }

    err = xTaskCreate(imu_task, "imu_task", TASK_STACK_DEPTH, NULL,
                      TASK_HIGH_PRIORITY, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create imu task, err code: %d");
      return;
    }

    err = xTaskCreate(stepper_task, "stepper_task", TASK_STACK_DEPTH, NULL,
                      TASK_HIGH_PRIORITY, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create stepper task, err code: %d");
      return;
    }

    // Lower priority than stepper_task.
    // err = xTaskCreate(stepper_test_task, "stepper_test", TASK_STACK_DEPTH,
    // NULL,
    //                   TASK_LOW_PRIORITY, NULL);
    // if (err != pdPASS) {
    //   ESP_LOGE(TAG, "Failed to create stepper_test task, err code: %d");
    //   return;
    // }
  }

  ESP_LOGI(TAG, "Tasks created, system running");
}
