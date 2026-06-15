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

static const char* const TAG = "robot_elbow_module";

enum {
  // CAN configuration
  CAN_TX_PIN = GPIO_NUM_0,
  CAN_RX_PIN = GPIO_NUM_1,
  CAN_BAUDRATE = 1000000,

  // IMU pins.
  IMU_SDA_PIN = GPIO_NUM_3,
  IMU_SCL_PIN = GPIO_NUM_4,

  // GPIO pin definitions
  // The enable pin is enabled by default.
  STEPPER_ELBOW_ENABLE_PIN = GPIO_NUM_NC,
  STEPPER_ELBOW_DIR_PIN = GPIO_NUM_6,
  STEPPER_ELBOW_STEP_PIN = GPIO_NUM_7,

  // ADC channels
  ADC_ELBOW_CHANNEL = ADC_CHANNEL_2,  // GPIO 2

  // PWM timers and channels
  PWM_ELBOW_CHANNEL = LEDC_CHANNEL_0,
  PWM_ELBOW_TIMER = LEDC_TIMER_0,
};

// Stepper configurations
static const stepper_control_config_t s_elbow_stepper_cfg = {
    .step_gpio = STEPPER_ELBOW_STEP_PIN,
    .dir_gpio = STEPPER_ELBOW_DIR_PIN,
    .enable_gpio = STEPPER_ELBOW_ENABLE_PIN,

    // Set these if you want microstepping.
    .microstepping_type = MICROSTEP_HARDWARE,
    .microstepping_mode = MICROSTEP_1_32,
    // We're setting the stepper driver's pins in hardware, so we don't need to
    // use them in the software.
    .microstep_m0_gpio = GPIO_NUM_NC,
    .microstep_m1_gpio = GPIO_NUM_NC,
    .microstep_m2_gpio = GPIO_NUM_NC,

    .direction = STEPPER_DIR_REVERSE,
    .steps_per_rev = 200,
    .gear_ratio = 15.0F,
    .max_speed_decreasing_angle = {40.0F},
    .max_speed_increasing_angle = {40.0F},
    .max_accel = {20.0F},
    .pwm_channel = PWM_ELBOW_CHANNEL,
    .pwm_timer = PWM_ELBOW_TIMER,
    .potentiometer =
        (Potentiometer){
            .range_of_motion = {285.F},
            // We step down voltage we feed to the potentiometer from 3300 mV to
            // 2200 mV.
            .min_adc_value = 0,
            .max_adc_value = 2200,
            .min_potentiometer_angle_as_joint_angle = {0.F},
            // Angle increases as the elbow is pulled in.
            .min_potentiometer_angle = {123.F},
            // Using the ratio between the joint angles and potentiomter angles,
            // 72 degrees of potentiometer angle becomes 60 degrees
            // of joint angle.
            .max_potentiometer_angle = {195.F},
            .joint_angle_to_potentiometer_angle_ratio = 18.F / 15.F,
            .is_reversed = true,
        },
};

// ADC manager configuration.
static const AdcMgrChannelConfig s_adc_mgr_channel_configs[] = {
    {
        .channel = ADC_ELBOW_CHANNEL,
        .sample_rate = 1000,
    },
};

static const AdcMgrConfig s_adc_mgr_config = {
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

static uint16_t
    s_adc_elbow_channel_underlying_buffer[ADC_STEPPERS_UNDERLYING_BUF_SIZE] = {
        0};

static AdcMgrReadResults s_adc_mgr_read_results = {
    .channel_buffers = {
        [ADC_ELBOW_CHANNEL] = {.data = s_adc_elbow_channel_underlying_buffer,
                               .capacity = LIMB_ARR_LEN(
                                   s_adc_elbow_channel_underlying_buffer)},
    }};

static AdcMgrChannelBuffer* s_adc_mgr_elbow_buffer =
    &s_adc_mgr_read_results.channel_buffers[ADC_ELBOW_CHANNEL];

static uint16_t s_latest_potentiometer_adc_value;

static stepper_control_handle_t s_elbow_stepper_handle = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static void can_rx_task([[maybe_unused]] void* pvParameter) {
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
        enum {
          kExpectedPayloadSize = 2 * sizeof(float),
        };
        if (rx_len != kExpectedPayloadSize) {
          ESP_LOGW(
              TAG,
              "Invalid actuation payload length for elbow: %u (expected %u)",
              rx_len, kExpectedPayloadSize);
          break;
        }

        JointAngle target_angle = {
            deserialize_float(msg_rx, kFromLittleEndian)};
        AngularVelocity target_velocity = {
            deserialize_float(msg_rx + sizeof(float), kFromLittleEndian)};
        stepper_set_target_angle(s_elbow_stepper_handle, target_angle);
        stepper_set_target_velocity(s_elbow_stepper_handle, target_velocity);
        ESP_LOGI(TAG,
                 "Received elbow actuation: angle=%f degrees, velocity=%f dps",
                 target_angle.degree, target_velocity.dps);
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

[[maybe_unused]]
static void imu_task([[maybe_unused]] void* pvParameter) {
  ImuRawData raw_data;

  uint32_t can_error_count = 0;
  uint32_t can_error_count_since_last_log = 0;
  esp_err_t err = ESP_OK;

  TickType_t current_tick = xTaskGetTickCount();
  while (1) {
    // Period of 100 Hz / 10 ms.
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(10));

    // Read IMU data.
    err = imu_read_data(&raw_data);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "Error reading IMU: %s", esp_err_to_name(err));
      continue;
    }

    const ImuData data = imu_to_mg_and_mdps(raw_data);

    // We first copy the floats we want to send to a buffer so we can reverse
    // the bytes if necessary to guarantee that we send them in little-endian
    // byte order.
    float can_buf[1] = {0};

    can_buf[0] = htolef(data.gyro.pitch);
    err = can_send(CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_PITCH, (uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.gyro.roll);
    err = can_send(CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_ROLL, (uint8_t*)&can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.gyro.yaw);
    err = can_send(CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_YAW, (uint8_t*)&can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.x);
    err = can_send(CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_X, (uint8_t*)&can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.y);
    err = can_send(CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Y, (uint8_t*)&can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.z);
    err = can_send(CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Z, (uint8_t*)&can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    enum {
      kMinCanErrorCountPerLogging = 100,
    };
    if (can_error_count_since_last_log > kMinCanErrorCountPerLogging) {
      can_error_count += can_error_count_since_last_log;
      can_error_count_since_last_log = 0;
      ESP_LOGW(TAG, "CAN errors: %d, last_error: %s", can_error_count,
               esp_err_to_name(err));
    }
  }
}

[[maybe_unused]]
static void reenable_can_task([[maybe_unused]] void* pvParameter) {
  while (true) {
    can_automatically_reenable_on_bus_off();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

static void stepper_task([[maybe_unused]] void* pvParameter) {
  TickType_t last_wake_time = xTaskGetTickCount();

  while (1) {
    const uint16_t dt_ms = 10;
    xTaskDelayUntil(&last_wake_time, pdMS_TO_TICKS(dt_ms));

    adc_mgr_read(&s_adc_mgr_read_results, 0);
    s_latest_potentiometer_adc_value = moving_average16(
        s_latest_potentiometer_adc_value, s_adc_mgr_elbow_buffer->data,
        s_adc_mgr_elbow_buffer->length);
    s_adc_mgr_elbow_buffer->length = 0;
    stepper_update(s_elbow_stepper_handle, dt_ms,
                   s_latest_potentiometer_adc_value);

    // Send current angle over CAN.
    PotentiometerAngle current_pot_angle =
        stepper_get_current_angle(s_elbow_stepper_handle);
    JointAngle current_angle =
        to_joint_angle(&s_elbow_stepper_cfg.potentiometer, current_pot_angle);
    esp_err_t err = can_send(CAN_ID_ROBOT_ELBOW_UP_DOWN_POTENTIOMETER,
                             (uint8_t*)&current_angle.degree,
                             sizeof(current_angle.degree), 0);
    static int can_errors_count = 0;
    if (err != ESP_OK) {
      if (can_errors_count++ % 100 == 0) {
        ESP_LOGW(
            TAG,
            "Error sending elbow status over CAN: %s, total error count: %u",
            esp_err_to_name(err), can_errors_count);
      }
    }

    static uint32_t status_counter = 0;
    enum {
      ITERATIONS_PER_LOGGING = 10,
    };
    if (++status_counter >= ITERATIONS_PER_LOGGING) {  // Every 100ms
      status_counter = 0;
      PotentiometerAngle target_pot_angle =
          stepper_get_target_angle(s_elbow_stepper_handle);
      JointAngle target_angle =
          to_joint_angle(&s_elbow_stepper_cfg.potentiometer, target_pot_angle);
      AngularVelocity velocity =
          stepper_get_current_velocity(s_elbow_stepper_handle);
      bool moving = stepper_is_moving(s_elbow_stepper_handle);

      // Also log locally
      ESP_LOGI(TAG,
               "adc: %u, Stepper elbow - Current(pot): %.2f°, Target(pot): "
               "%.2f°, Current(Joint): %.2f, Target(Joint): %.2f, Velocity: "
               "%.2f°/s, Moving: %s",
               s_latest_potentiometer_adc_value, current_pot_angle.degree,
               target_pot_angle.degree, current_angle.degree,
               target_angle.degree, velocity.dps, moving ? "Yes" : "No");
    }
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
      abort();
    }
    ESP_LOGI(TAG, "CAN initialized (TX=%d, RX=%d, %d baud)", CAN_TX_PIN,
             CAN_RX_PIN, CAN_BAUDRATE);
  }

#if CONFIG_IMU_ENABLED
  // Initialize IMU.
  {
    ImuConfig imu_cfg = IMU_CONFIG_DEFAULT();
    imu_cfg.sda_pin = IMU_SDA_PIN;
    imu_cfg.scl_pin = IMU_SCL_PIN;
    esp_err_t err = imu_init(&imu_cfg);
    if (err) {
      ESP_LOGE(TAG, "Failed to initialize IMU driver: %s",
               esp_err_to_name(err));
      abort();
    }
    ESP_LOGI(TAG, "IMU initialized (SDA=%d, SCL=%d)", imu_cfg.sda_pin,
             imu_cfg.scl_pin);
  }
#endif

  // Initialize ADC.
  {
    esp_err_t err = adc_mgr_init(s_adc_mgr_config);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Failed to initialize ADC manager: %s",
               esp_err_to_name(err));
      abort();
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
      abort();
    }
  }
  s_adc_mgr_elbow_buffer->length = 0;

  // Create FreeRTOS tasks
  enum {
    TASK_STACK_DEPTH = 4096,
    TASK_STEPPER_UPDATE_PRIORITY = 6,
    TASK_CAN_RX_PRIORITY = 5,
    TASK_IMU_PRIORITY = 4,
    TASK_STEPPER_TEST_PRIORITY = 3,
  };

  {
    BaseType_t err = xTaskCreate(can_rx_task, "can_rx", TASK_STACK_DEPTH, NULL,
                                 TASK_CAN_RX_PRIORITY, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create can_rx task, err code: %d");
      abort();
    }
  }

#if CONFIG_FORCE_REENABLE_CAN_ON_BUS_OFF
  {
    BaseType_t err =
        xTaskCreate(reenable_can_task, "reenable_can_task", TASK_STACK_DEPTH,
                    NULL, TASK_CAN_RX_PRIORITY + 1, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create reenable_can_task, err code: %d");
      abort();
    }
  }
#endif

#if CONFIG_IMU_ENABLED
  {
    BaseType_t err = xTaskCreate(imu_task, "imu_task", TASK_STACK_DEPTH, NULL,
                                 TASK_IMU_PRIORITY, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create imu task, err code: %d");
      abort();
    }
  }
#endif

  {
    BaseType_t err = xTaskCreate(stepper_task, "stepper_task", TASK_STACK_DEPTH,
                                 NULL, TASK_STEPPER_UPDATE_PRIORITY, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create stepper task, err code: %d");
      abort();
    }
  }

  ESP_LOGI(TAG, "Tasks created, system running");
}
