#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"
#include "portmacro.h"

const char *const TAG = "Example shoulder controller";

#define LIMB_ARR_LEN(x) (sizeof(x) / sizeof(*(x)))

static void can_rx([[maybe_unused]] void *arg) {
  uint32_t can_id = 0;
  uint8_t can_data[CAN_MAX_MESSAGE_SIZE] = {0};
  uint8_t can_data_len = 0;
  while (true) {
    esp_err_t err =
        can_receive(&can_id, can_data, &can_data_len, portMAX_DELAY);
    if (err) {
      ESP_LOGW(TAG, "Error calling can_receive.", esp_err_to_name(err));
      continue;
    }
    switch (can_id) {
      case CAN_ID_ROBOT_SHOULDER_IMU_ACCEL: {
        if (can_data_len != 6) {
          ESP_LOGW(TAG, "IMU data received is %dB, not 6.", can_data_len);
        }
        static int i = 0;
        if (++i == 100) {
          uint16_t *imu_xyz = (uint16_t *)can_data;
          ESP_LOGI(TAG, "IMU accel received: %d, %d, %d", imu_xyz[0],
                   imu_xyz[1], imu_xyz[2]);
          i = 0;
        }
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_IMU_GYRO: {
        if (can_data_len != 6) {
          ESP_LOGW(TAG, "IMU data received is %dB, not 6.", can_data_len);
        }
        static int i = 0;
        if (++i == 100) {
          uint16_t *imu_xyz = (uint16_t *)can_data;
          ESP_LOGI(TAG, "IMU gyro received: %d, %d, %d", imu_xyz[0], imu_xyz[1],
                   imu_xyz[2]);
          i = 0;
        }
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_UP_DOWN_POTENTIOMETER: {
        static int i = 0;
        if (++i == 100) {
          float angle = *(float *)can_data;
          ESP_LOGI(TAG, "Shoulder up/down angle received: %f", angle);
          i = 0;
        }
        break;
      }
      case CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_POTENTIOMETER: {
        static int i = 0;
        if (++i == 100) {
          float angle = *(float *)can_data;
          ESP_LOGI(TAG, "Shoulder left/right angle received: %f", angle);
          i = 0;
        }
        break;
      }
      default: {
        ESP_LOGW(TAG, "Unknown CAN ID received: %x", can_id);
        break;
      }
    }
  }

  vTaskDelete(NULL);
}

static void can_tx([[maybe_unused]] void *arg) {
  const float angles[] = {
      0, 15, 30, 45, 90, 135, 150, 165, 180,
  };
  while (true) {
    ESP_LOGI(TAG, "Testing sending actuations");
    for (int i = 0; i < LIMB_ARR_LEN(angles); ++i) {
      {
        float angle = angles[i];
        esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION,
                                 (uint8_t *)&angle, sizeof(angle), 0);
        if (err != ESP_OK) {
          ESP_LOGW(TAG,
                   "Error calling can_send for shoulder up down actuation: %s",
                   esp_err_to_name(err));
        }
      }
      {
        float angle = angles[LIMB_ARR_LEN(angles) - 1 - i];
        esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION,
                                 (uint8_t *)&angle, sizeof(angle), 0);
        if (err != ESP_OK) {
          ESP_LOGW(
              TAG,
              "Error calling can_send for shoulder left_right actuation: %s",
              esp_err_to_name(err));
        }
      }
      vTaskDelay(pdMS_TO_TICKS(1000));
    }

    ESP_LOGI(TAG, "Testing sending stop messages");
    {
      float angle = 0;
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION,
                               (uint8_t *)&angle, sizeof(angle), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG,
                 "Error calling can_send for shoulder up/down actuation: %s",
                 esp_err_to_name(err));
      }
    }
    {
      float angle = 0;
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION,
                               (uint8_t *)&angle, sizeof(angle), 0);
      if (err != ESP_OK) {
        ESP_LOGW(
            TAG,
            "Error calling can_send for shoulder left/right  actuation: %s",
            esp_err_to_name(err));
      }
    }
    // Wait for servos to fully actuate.
    vTaskDelay(pdMS_TO_TICKS(1000));

    // Send actuation message to other extreme and send a stop message before
    // the motor completes its movement.
    // NOTE: There's a bug in the version of ESP-IDF that we use where empty
    // messages sent over CAN are sent with a data length of 8. So to the
    // receiver, it will look like we're sending data even though we aren't.
    // Issue: https://github.com/espressif/esp-idf/issues/17467
    {
      float angle = 180;
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION,
                               (uint8_t *)&angle, sizeof(angle), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG,
                 "Error calling can_send for shoulder up/down actuation: %s",
                 esp_err_to_name(err));
      }
    }
    vTaskDelay(pdMS_TO_TICKS(200));
    {
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_UP_DOWN_STOP, NULL, 0, 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG,
                 "Error calling can_send for shoulder up/down stop message: %s",
                 esp_err_to_name(err));
      }
    }
    {
      float angle = 180;
      esp_err_t err = can_send(CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION,
                               (uint8_t *)&angle, sizeof(angle), 0);
      if (err != ESP_OK) {
        ESP_LOGW(TAG,
                 "Error calling can_send for shoulder left/right actuation: %s",
                 esp_err_to_name(err));
      }
    }
    vTaskDelay(pdMS_TO_TICKS(200));
    {
      esp_err_t err =
          can_send(CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_STOP, NULL, 0, 0);
      if (err != ESP_OK) {
        ESP_LOGW(
            TAG,
            "Error calling can_send for shoulder left/right stop message: %s",
            esp_err_to_name(err));
      }
    }
    vTaskDelay(pdMS_TO_TICKS(2000));
  }

  vTaskDelete(NULL);
}

void app_main(void) {
  ESP_ERROR_CHECK(can_init(5, 4, 1000000, NULL));
  xTaskCreate(can_tx, "can tx task", 1024 * 2 * 2, NULL, 5, NULL);
  xTaskCreate(can_rx, "can rx task", 1024 * 2 * 2, NULL, 5, NULL);
}
