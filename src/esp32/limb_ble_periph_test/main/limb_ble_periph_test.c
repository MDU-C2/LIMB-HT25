#include "limb_ble_periph.h"

#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "portmacro.h"
#include "sensors_service.h"

enum { kMsInS = 1000 };

void SendEmgDataTask([[maybe_unused]] void* arg) {
  CharacteristicBuffer emg_buf = get_emg_buf();
  // The amount of milliseconds worth of data the EMG buffer can hold.
  const uint16_t emg_buf_size_in_ms =
      emg_buf.size / kEmgBytesPerSample * kMsInS / kEmgFrequency;
  const TickType_t delay_time = pdMS_TO_TICKS(emg_buf_size_in_ms);

  while (true) {
    bool sent = TryNotifyEmgSubscribers();
    if (sent) {
      ++emg_buf.data[0];
    } else {
      if (emg_buf.data[0]) {
        printf("EMG times notified: %d\n", emg_buf.data[0]);
      }
      emg_buf.data[0] = 0;
    }

    vTaskDelay(delay_time);
  }

  vTaskDelete(NULL);
}

void SendImuDataTask([[maybe_unused]] void* arg) {
  CharacteristicBuffer imu_buf = get_imu_buf();
  // The amount of milliseconds worth of data the IMU buffer can hold.
  const uint16_t imu_buf_size_in_ms =
      imu_buf.size / kImuBytesPerSample * kMsInS / kImuFrequency;
  const TickType_t delay_time = pdMS_TO_TICKS(imu_buf_size_in_ms);

  while (true) {
    bool sent = TryNotifyImuSubscribers();
    if (sent) {
      ++imu_buf.data[0];
    } else {
      if (imu_buf.data[0]) {
        printf("IMU times notified: %d\n", imu_buf.data[0]);
      }
      imu_buf.data[0] = 0;
    }

    vTaskDelay(delay_time);
  }

  vTaskDelete(NULL);
}

void SendPiezoDataTask([[maybe_unused]] void* arg) {
  CharacteristicBuffer piezo_buf = get_piezo_buf();
  // The amount of milliseconds worth of data the piezo buffer can hold.
  const uint16_t piezo_buf_size_in_ms =
      piezo_buf.size / kPiezoBytesPerSample * kMsInS / kPiezoFrequency;
  const TickType_t delay_time = pdMS_TO_TICKS(piezo_buf_size_in_ms);

  while (true) {
    bool sent = TryNotifyPiezoSubscribers();
    if (sent) {
      ++piezo_buf.data[0];
    } else {
      if (piezo_buf.data[0]) {
        printf("Piezo times notified: %d\n", piezo_buf.data[0]);
      }
      piezo_buf.data[0] = 0;
    }

    vTaskDelay(delay_time);
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
