
#include <stdint.h>
#include <stdio.h>

#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "can_rx";

// CAN configuration
#define CAN_TX_PIN 5
#define CAN_RX_PIN 4
#define CAN_BAUDRATE 500000

// CAN Frame IDs (must match TX)
#define CAN_ID_IMU_DATA 0x100
#define CAN_ID_IMU_CONTROL 0x200

// IMU CAN payload structure (matches TX)
// Note: CAN frames are limited to 8 bytes, so we use int16 values
// Scale factors: accel = ±4g (8192 counts/g), gyro = ±250 dps (16384
// counts/dps)
typedef struct {
  int16_t accel_x;  // Raw accel X
  int16_t accel_y;  // Raw accel Y
  int16_t gyro_x;   // Raw gyro X
  int16_t gyro_y;   // Raw gyro Y
} imu_can_payload_t;

// Receive timeout (ms)
#define CAN_RX_TIMEOUT_MS 100

/**
 * @brief Parse received CAN data into IMU payload structure
 *
 * CAN frame contains 8 bytes: 4 int16 values (little-endian)
 * Bytes 0-1: accel_x (int16)
 * Bytes 2-3: accel_y (int16)
 * Bytes 4-5: gyro_x (int16)
 * Bytes 6-7: gyro_y (int16)
 */
static void parse_imu_frame(const uint8_t* data, uint8_t len,
                            imu_can_payload_t* payload) {
  if (len >= 8) {
    // Parse 4 int16 values (little-endian)
    payload->accel_x = (int16_t)((data[1] << 8) | data[0]);
    payload->accel_y = (int16_t)((data[3] << 8) | data[2]);
    payload->gyro_x = (int16_t)((data[5] << 8) | data[4]);
    payload->gyro_y = (int16_t)((data[7] << 8) | data[6]);
  } else {
    // Not enough data
    payload->accel_x = 0;
    payload->accel_y = 0;
    payload->gyro_x = 0;
    payload->gyro_y = 0;
  }
}

/**
 * @brief Task to receive CAN frames and parse into IMU payload
 *
 * Continuously receives CAN frames and logs IMU data.
 */
static void can_rx_task(void* arg) {
  uint32_t can_id;
  uint8_t rx_buffer[8];
  uint8_t dlc;  // Data Length Code
  imu_can_payload_t payload = {0};

  ESP_LOGI(TAG, "CAN RX task started");

  while (1) {
    // Tryto receive a CAN frame
    esp_err_t ret = can_receive(&can_id, rx_buffer, &dlc, CAN_RX_TIMEOUT_MS);
    if (ret == ESP_OK) {
      // Frame received successfully
      ESP_LOGI(TAG,
               "RX: ID=0x%03X, DLC=%d, data=[%02X %02X %02X %02X %02X %02X "
               "%02X %02X]",
               can_id, dlc, rx_buffer[0], rx_buffer[1], rx_buffer[2],
               rx_buffer[3], rx_buffer[4], rx_buffer[5], rx_buffer[6],
               rx_buffer[7]);

      if (can_id == CAN_ID_IMU_DATA) {
        parse_imu_frame(rx_buffer, dlc, &payload);

        // Convert to float for display (matching TX scaling)
        float accel_x_f = (float)payload.accel_x / 8192.0f;  // ±4g
        float accel_y_f = (float)payload.accel_y / 8192.0f;
        float gyro_x_f = (float)payload.gyro_x / 16384.0f;  // ±250 dps
        float gyro_y_f = (float)payload.gyro_y / 16384.0f;

        ESP_LOGI(
            TAG,
            "IMU Data: accel_x=%.3f, accel_y=%.3f, gyro_x=%.3f, gyro_y=%.3f",
            accel_x_f, accel_y_f, gyro_x_f, gyro_y_f);
      }

      // Handle control commands
      if (can_id == CAN_ID_IMU_CONTROL && dlc > 0) {
        if (rx_buffer[0] == 1) {
          ESP_LOGI(TAG, "IMU control: Recalibrate/zero command received");
        }
      }
    } else if (ret == ESP_ERR_TIMEOUT) {
      ESP_LOGD(TAG, "RX timeout, no CAN frame received");
    } else {
      ESP_LOGE(TAG, "RX error: %s (0x%x)", esp_err_to_name(ret), ret);
    }
  }
}

void app_main(void) {
  ESP_LOGI(TAG, "Starting CAN RX task");

  // Init CAN
  esp_err_t ret = can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize CAN: %s", esp_err_to_name(ret));
    return;
  }

  ESP_LOGI(TAG, "CAN initialized: TX=GPIO%d, RX=GPIO%d, %d bps", CAN_TX_PIN,
           CAN_RX_PIN, CAN_BAUDRATE);

  // Create CAN receive task
  xTaskCreate(can_rx_task, "can_rx_task", 4096, NULL, 5, NULL);

  ESP_LOGI(TAG, "Application started. Waiting for CAN frames...");
  ESP_LOGI(TAG, "Listening for IMU data (ID: 0x%03X)", CAN_ID_IMU_DATA);
}