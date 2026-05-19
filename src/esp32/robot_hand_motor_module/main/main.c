#include "HS422_led.h"
#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "imu.h"
#include "limb_utils.h"
#include "portmacro.h"

static const char* TAG = "SERVOS";

enum {
  CAN_TX_PIN = 3,
  CAN_RX_PIN = 4,
  CAN_BAUDRATE = 1000000,
};

static void imu_task([[maybe_unused]] void* pvParameter) {
  uint32_t can_error_count = 0;
  esp_err_t err = ESP_OK;
  uint32_t can_error_count_since_last_log = 0;

  const uint16_t period_ms = 100;
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
    err = can_send(CAN_ID_ROBOT_HAND_IMU_GYRO_PITCH, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.gyro.roll);
    err = can_send(CAN_ID_ROBOT_HAND_IMU_GYRO_ROLL, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.gyro.yaw);
    err = can_send(CAN_ID_ROBOT_HAND_IMU_GYRO_YAW, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.x);
    err = can_send(CAN_ID_ROBOT_HAND_IMU_ACCEL_X, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.y);
    err = can_send(CAN_ID_ROBOT_HAND_IMU_ACCEL_Y, (const uint8_t*)can_buf,
                   sizeof(can_buf), 0);
    if (err != ESP_OK) {
      ++can_error_count_since_last_log;
    }

    can_buf[0] = htolef(data.accel.z);
    err = can_send(CAN_ID_ROBOT_HAND_IMU_ACCEL_Z, (const uint8_t*)can_buf,
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
      if (rx_len != 1) {
        ESP_LOGW(TAG, "Received grip activation with invalid len: %u", rx_len);
        continue;
      }
      int angle = (int)rx_data[0];

      for (int i = 0; i < NUM_SERVOS; i++) {
        servo_write_deg_channel(i, angle);
        vTaskDelay(pdMS_TO_TICKS(50));
      }

      ESP_LOGI(TAG, "RX-angles %d", angle);
    } else if (rx_id == CAN_ID_ROBOT_LOWER_ARM_ROTATION_ACTUATION) {
      if (rx_len != sizeof(float)) {
        ESP_LOGW(TAG, "Received rotation activation with invalid len: %u",
                 rx_len);
        continue;
      }
      float angle = deserialize_float(rx_data, kFromLittleEndian);
      float velocity =
          deserialize_float(rx_data + sizeof(float), kFromLittleEndian);
      servo_write_deg_channel(WRIST_SERVO_CONFIG_INDEX, angle);
      ESP_LOGI(TAG, "Actuation wrist to %.2f degrees at %.2f dps", angle,
               velocity);
    }
  }
}

void app_main() {
  ESP_LOGI(TAG, "Starting servo control application");
  // vTaskDelay(pdMS_TO_TICKS(2000));

  // Initialize all servos
  ESP_LOGI(TAG, "Initializing servos...");
  servo_led_init();
  // vTaskDelay(pdMS_TO_TICKS(1000));

  {
    ESP_LOGI(TAG, "Initializing IMUs...");
    ImuConfig imu_config = IMU_CONFIG_DEFAULT();
    imu_config.sda_pin = GPIO_NUM_0;
    imu_config.scl_pin = GPIO_NUM_1;
    ESP_ERROR_CHECK_WITHOUT_ABORT(imu_init(&imu_config));
    if (!imu_is_present()) {
      ESP_LOGW(TAG, "IMU isn't present");
      abort();
    }
  }

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
  {
    BaseType_t err =
        xTaskCreate(can_rx_task, "can_rx_task", 1024 * 2 * 2, NULL, 6, NULL);
    if (err != pdPASS) {
      ESP_LOGE(TAG, "Failed to create imu task, err code: %d", err);
      abort();
    }
  }
}
