#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "imu.h"
#include "limb_utils.h"
#include "portmacro.h"
#include "servo.h"

static const char* TAG = "Hand motor module";

enum {
  CAN_TX_PIN = GPIO_NUM_6,
  CAN_RX_PIN = GPIO_NUM_10,
  CAN_BAUDRATE = 1000000,

  IMU_SDA_GPIO = GPIO_NUM_9,
  IMU_SCL_GPIO = GPIO_NUM_7,

  THUMB_CONFIG_INDEX = 0,
  INDEX_CONFIG_INDEX = 1,
  MIDDLE_CONFIG_INDEX = 2,
  RING_CONFIG_INDEX = 3,
  PINKY_CONFIG_INDEX = 4,
  WRIST_CONFIG_INDEX = 5,

  THUMB_SERVO_GPIO = GPIO_NUM_0,
  INDEX_SERVO_GPIO = GPIO_NUM_1,
  MID_SERVO_GPIO = GPIO_NUM_2,
  RING_SERVO_GPIO = GPIO_NUM_3,
  PINKY_SERVO_GPIO = GPIO_NUM_4,
  TWIST_SERVO_GPIO = GPIO_NUM_5,
};

// Servo configurations - customize each servo individually
static const servo_config_t s_servo_configs[] = {
    // Thumb servo
    [THUMB_CONFIG_INDEX] = {.gpio_pin = THUMB_SERVO_GPIO,
                            .ledc_channel = LEDC_CHANNEL_0,
                            .max_angle = 30,
                            .min_angle = 0,
                            .min_pulse_us = 1400,
                            .max_pulse_us = 1900,
                            .max_speed = {40},
                            .direction = SERVO_DIR_REVERSE,
                            .name = "Thumb"},
    // Index finger
    [INDEX_CONFIG_INDEX] = {.gpio_pin = INDEX_SERVO_GPIO,
                            .ledc_channel = LEDC_CHANNEL_1,
                            .max_angle = 85,
                            .min_angle = 0,
                            .min_pulse_us = 1100,
                            .max_pulse_us = 1900,
                            .max_speed = {40},
                            .direction = SERVO_DIR_REVERSE,
                            .name = "Index"},
    // Middle finger
    [MIDDLE_CONFIG_INDEX] = {.gpio_pin = MID_SERVO_GPIO,
                             .ledc_channel = LEDC_CHANNEL_2,
                             .max_angle = 90,
                             .min_angle = 0,
                             .min_pulse_us = 800,
                             .max_pulse_us = 1700,
                             .max_speed = {40},
                             .direction = SERVO_DIR_REVERSE,
                             .name = "Middle"},
    // Ring finger
    [RING_CONFIG_INDEX] = {.gpio_pin = RING_SERVO_GPIO,
                           .ledc_channel = LEDC_CHANNEL_3,
                           .max_angle = 50,
                           .min_angle = 0,
                           .min_pulse_us = 1400,
                           .max_pulse_us = 2200,
                           .max_speed = {40},
                           .direction = SERVO_DIR_REVERSE,
                           .name = "Ring"},
    // Pinky finger
    [PINKY_CONFIG_INDEX] = {.gpio_pin = PINKY_SERVO_GPIO,
                            .ledc_channel = LEDC_CHANNEL_4,
                            .max_angle = 90,
                            .min_angle = 0,
                            .min_pulse_us = 700,
                            .max_pulse_us = 1600,
                            .max_speed = {120},
                            .direction = SERVO_DIR_REVERSE,
                            .name = "Pinky"},
    [WRIST_CONFIG_INDEX] = {.gpio_pin = TWIST_SERVO_GPIO,
                            .ledc_channel = LEDC_CHANNEL_5,
                            .min_angle = 0,
                            .max_angle = 140,
                            .min_pulse_us = 500,
                            .max_pulse_us = 2500,
                            .max_speed = {100},
                            .direction = SERVO_DIR_REVERSE,
                            .name = "Wrist"},
};

static void reenable_can_task([[maybe_unused]] void* pvParameter) {
  while (true) {
    can_automatically_reenable_on_bus_off();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

static void imu_task([[maybe_unused]] void* pvParameter) {
  uint32_t can_error_count = 0;
  esp_err_t err = ESP_OK;
  uint32_t can_error_count_since_last_log = 0;

  const uint16_t period_ms = 10;
  TickType_t current_tick = xTaskGetTickCount();
  while (true) {
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(period_ms));
    ImuRawData raw_data = {0};
    err = imu_read_data(&raw_data);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "Error reading IMU data: %s", esp_err_to_name(err));
      continue;
    }

    ImuData data = imu_to_mg_and_mdps(raw_data);

    ESP_LOGI(
        TAG,
        "Read IMU accel [%.2f, %.2f, %.2f] mG, gyro [%.2f, %.2f, %.2f] mdps",
        data.accel.x, data.accel.y, data.accel.z, data.gyro.pitch,
        data.gyro.roll, data.gyro.yaw);

    // We first copy the floats we want to send to a buffer so we can reverse
    // the bytes if necessary to guarantee that we send them in little-endian
    // byte order.
    float can_buf[1] = {0};

    can_buf[0] = htolef(data.gyro.pitch);
    err = can_send(CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_PITCH,
                   (const uint8_t*)can_buf, sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.gyro.roll);
    err = can_send(CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_ROLL,
                   (const uint8_t*)can_buf, sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.gyro.yaw);
    err = can_send(CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_YAW, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.x);
    err = can_send(CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_X, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.y);
    err = can_send(CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_Y, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.z);
    err = can_send(CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_Z, (const uint8_t*)can_buf,
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

static void can_rx_task([[maybe_unused]] void* pvParameter) {
  ESP_LOGI(TAG, "Starting CAN rx task");

  while (true) {
    uint32_t rx_id = 0;
    uint8_t rx_data[CAN_MAX_MESSAGE_SIZE] = {0};
    uint8_t rx_len = 0;

    esp_err_t err = can_receive(&rx_id, rx_data, &rx_len, portMAX_DELAY);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "Error receiving CAN message: %s", esp_err_to_name(err));
      continue;
    }

    if (rx_id == CAN_ID_ROBOT_THUMB_ACTUATION) {
      if (rx_len != 2 * sizeof(float)) {
        ESP_LOGW(TAG, "Received thumb activation with invalid len: %u", rx_len);
        continue;
      }

      float angle = deserialize_float(rx_data, kFromLittleEndian);
      AngularVelocity velocity = {
          deserialize_float(rx_data + sizeof(float), kFromLittleEndian)};
      servo_move_to_angle_with_speed(&s_servo_configs[THUMB_CONFIG_INDEX],
                                     angle, velocity);
      ESP_LOGI(TAG, "Actuation thumb to %.2f degrees at %.2f dps", angle,
               velocity.dps);
    } else if (rx_id == CAN_ID_ROBOT_INDEX_ACTUATION) {
      if (rx_len != 2 * sizeof(float)) {
        ESP_LOGW(TAG, "Received index activation with invalid len: %u", rx_len);
        continue;
      }

      float angle = deserialize_float(rx_data, kFromLittleEndian);
      AngularVelocity velocity = {
          deserialize_float(rx_data + sizeof(float), kFromLittleEndian)};
      servo_move_to_angle_with_speed(&s_servo_configs[INDEX_CONFIG_INDEX],
                                     angle, velocity);
      ESP_LOGI(TAG, "Actuation index to %.2f degrees at %.2f dps", angle,
               velocity.dps);
    } else if (rx_id == CAN_ID_ROBOT_MIDDLE_ACTUATION) {
      if (rx_len != 2 * sizeof(float)) {
        ESP_LOGW(TAG, "Received middle activation with invalid len: %u",
                 rx_len);
        continue;
      }

      float angle = deserialize_float(rx_data, kFromLittleEndian);
      AngularVelocity velocity = {
          deserialize_float(rx_data + sizeof(float), kFromLittleEndian)};
      servo_move_to_angle_with_speed(&s_servo_configs[MIDDLE_CONFIG_INDEX],
                                     angle, velocity);
      ESP_LOGI(TAG, "Actuation middle to %.2f degrees at %.2f dps", angle,
               velocity.dps);
    } else if (rx_id == CAN_ID_ROBOT_RING_ACTUATION) {
      if (rx_len != 2 * sizeof(float)) {
        ESP_LOGW(TAG, "Received ring activation with invalid len: %u", rx_len);
        continue;
      }

      float angle = deserialize_float(rx_data, kFromLittleEndian);
      AngularVelocity velocity = {
          deserialize_float(rx_data + sizeof(float), kFromLittleEndian)};
      servo_move_to_angle_with_speed(&s_servo_configs[RING_CONFIG_INDEX], angle,
                                     velocity);
      ESP_LOGI(TAG, "Actuation ring to %.2f degrees at %.2f dps", angle,
               velocity.dps);
    } else if (rx_id == CAN_ID_ROBOT_PINKY_ACTUATION) {
      if (rx_len != 2 * sizeof(float)) {
        ESP_LOGW(TAG, "Received pinky activation with invalid len: %u", rx_len);
        continue;
      }

      float angle = deserialize_float(rx_data, kFromLittleEndian);
      AngularVelocity velocity = {
          deserialize_float(rx_data + sizeof(float), kFromLittleEndian)};
      servo_move_to_angle_with_speed(&s_servo_configs[PINKY_CONFIG_INDEX],
                                     angle, velocity);
      ESP_LOGI(TAG, "Actuation pinky to %.2f degrees at %.2f dps", angle,
               velocity.dps);
    } else if (rx_id == CAN_ID_ROBOT_LOWER_ARM_ROTATION_ACTUATION) {
      if (rx_len != 2 * sizeof(float)) {
        ESP_LOGW(TAG, "Received rotation activation with invalid len: %u",
                 rx_len);
        continue;
      }

      float angle = deserialize_float(rx_data, kFromLittleEndian);
      AngularVelocity velocity = {
          deserialize_float(rx_data + sizeof(float), kFromLittleEndian)};
      servo_move_to_angle_with_speed(&s_servo_configs[WRIST_CONFIG_INDEX],
                                     angle, velocity);
      ESP_LOGI(TAG, "Actuation wrist to %.2f degrees at %.2f dps", angle,
               velocity.dps);
    }
  }
}

void app_main() {
  ESP_LOGI(TAG, "Starting servo control application");
  // vTaskDelay(pdMS_TO_TICKS(2000));

  // Initialize all servos
  ESP_LOGI(TAG, "Initializing servos...");
  servo_led_init(s_servo_configs, LIMB_ARR_LEN(s_servo_configs));
  // vTaskDelay(pdMS_TO_TICKS(1000));

#if CONFIG_IMU_ENABLED
  {
    ESP_LOGI(TAG, "Initializing IMUs...");
    ImuConfig imu_config = IMU_CONFIG_DEFAULT();
    imu_config.sda_pin = IMU_SDA_GPIO;
    imu_config.scl_pin = IMU_SCL_GPIO;
    ESP_ERROR_CHECK_WITHOUT_ABORT(imu_init(&imu_config));
    if (!imu_is_present()) {
      ESP_LOGW(TAG, "IMU isn't present");
      abort();
    }
  }
#endif

  // Initialize rotary encoder
  // ESP_LOGI(TAG, "Initializing rotary encoder...");
  // rotary_encoder_init();
  // vTaskDelay(pdMS_TO_TICKS(1000));

  // Start calibration mode
  // Uncomment the line below to enter calibration mode
  // start_calibration_mode();

  // init CAN CX---------------
  {
    esp_err_t err = can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE, NULL);
    if (err) {
      ESP_LOGE(TAG, "Couldn't start can driver: %s", esp_err_to_name(err));
      abort();
    }
  }

#if CONFIG_IMU_ENABLED
  {
    BaseType_t err =
        xTaskCreate(imu_task, "imu_task", 1024 * 2 * 2, NULL, 5, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create imu task, err code: %d", err);
      can_deinit();
      imu_deinit();
      abort();
    }
  }
#endif

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

  {
    BaseType_t err =
        xTaskCreate(can_rx_task, "can_rx_task", 1024 * 2 * 2, NULL, 6, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create imu task, err code: %d", err);
      abort();
    }
  }
}
