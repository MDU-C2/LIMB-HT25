#include "limb_ble_periph.h"

#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "sensors_service.h"

void SendEmgDataTask([[maybe_unused]] void* arg) {
  uint8_t* emg_buf = get_emg_buf();

  while (true) {
    bool sent = TryNotifyEmgSubscribers();
    if (sent) {
      ++emg_buf[0];
    } else {
      if (emg_buf[0]) {
        printf("EMG times notified: %d\n", emg_buf[0]);
      }
      emg_buf[0] = 0;
    }
    enum {
      kDelayTime = pdMS_TO_TICKS(kEmgBufInMs),
    };
    vTaskDelay(kDelayTime);
  }
  vTaskDelete(NULL);
}

void SendImuDataTask([[maybe_unused]] void* arg) {
  uint8_t* imu_buf = get_imu_buf();

  while (true) {
    bool sent = TryNotifyImuSubscribers();
    if (sent) {
      ++imu_buf[0];
    } else {
      if (imu_buf[0]) {
        printf("IMU times notified: %d\n", imu_buf[0]);
      }
      imu_buf[0] = 0;
    }
    vTaskDelay(pdMS_TO_TICKS(kImuBufInMs));
  }
  vTaskDelete(NULL);
}

void SendPiezoDataTask([[maybe_unused]] void* arg) {
  uint8_t* piezo_buf = get_piezo_buf();

  while (true) {
    bool sent = TryNotifyPiezoSubscribers();
    if (sent) {
      ++piezo_buf[0];
    } else {
      if (piezo_buf[0]) {
        printf("Piezo times notified: %d\n", piezo_buf[0]);
      }
      piezo_buf[0] = 0;
    }
    vTaskDelay(pdMS_TO_TICKS(kPiezoBufInMs));
  }
  vTaskDelete(NULL);
}

void app_main(void) {
  enum {
    kStackDepth = 4 * 1024,
    kTaskPriority = 5,
  };

  xTaskCreate(BleTask, "BleTask", kStackDepth, NULL, kTaskPriority, NULL);
  xTaskCreate(SendEmgDataTask, "SendEmgDataTask", kStackDepth, NULL,
              kTaskPriority, NULL);
  xTaskCreate(SendPiezoDataTask, "SendPiezoDataTask", kStackDepth, NULL,
              kTaskPriority, NULL);
  xTaskCreate(SendImuDataTask, "SendImuDataTask", kStackDepth, NULL,
              kTaskPriority, NULL);
  printf("Done\n");
}
