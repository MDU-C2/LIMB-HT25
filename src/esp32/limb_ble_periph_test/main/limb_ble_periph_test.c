#include "limb_ble_periph.h"

#include <stdio.h>

#include "endian.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "portmacro.h"
#include "sensors_service.h"

enum { kMsInS = 1000 };

// The amount of milliseconds worth of data a sensor buffer can hold.
static uint16_t BufSizeInMs(uint16_t buf_size, uint8_t bytes_per_sample,
                            uint8_t sensor_count, uint16_t frequency) {
  return (buf_size - kSequenceNumberSize) / (bytes_per_sample * sensor_count) *
         kMsInS / frequency;
}

void SendEmgDataTask([[maybe_unused]] void* arg) {
  CharacteristicBuffer emg_buf = get_emg_buf();
  const uint16_t emg_buf_size_in_ms = BufSizeInMs(
      emg_buf.size, kEmgBytesPerSample, kEmgSensorCount, kEmgFrequency);
  const TickType_t delay_time = pdMS_TO_TICKS(emg_buf_size_in_ms);

  bool is_sending = false;
  uint32_t starting_value = 0;

  static_assert(kEmgBytesPerValue == 2,
                "Since we're using uint16_t, the bytes per sample must be 2.");
  // Skip the sequence number and treat the buffer as 16-bit values.
  uint16_t* emg_buf_data = (uint16_t*)(emg_buf.data + 4);
  uint16_t emg_buf_data_size = (emg_buf.size - 4) / 2;

  TickType_t wakeup_time = xTaskGetTickCount();
  while (true) {
    uint32_t sequence_number = le32toh(*(uint32_t*)emg_buf.data);

    // Fill buffer.
    *(uint32_t*)emg_buf.data = htole32(sequence_number + 1);
    for (uint16_t i = 0; i < emg_buf_data_size; ++i) {
      emg_buf_data[i] = htole16(i + 1);
    }

    bool sent = TryNotifyEmgSubscribers();

    if (!is_sending && sent) {
      is_sending = true;
      starting_value = sequence_number;
    } else if (is_sending && !sent) {
      is_sending = false;
      const uint32_t notifications_sent_count =
          sequence_number - starting_value;
      ESP_LOGW("emgsender", "Sent %d emg notifications in a row.",
               notifications_sent_count);
    }

    xTaskDelayUntil(&wakeup_time, delay_time);
  }

  vTaskDelete(NULL);
}

void SendImuDataTask([[maybe_unused]] void* arg) {
  CharacteristicBuffer imu_buf = get_imu_buf();
  const uint16_t imu_buf_size_in_ms = BufSizeInMs(
      imu_buf.size, kImuBytesPerSample, kImuSensorCount, kImuFrequency);
  const TickType_t delay_time = pdMS_TO_TICKS(imu_buf_size_in_ms);

  bool is_sending = false;
  uint32_t starting_value = 0;

  static_assert(kImuBytesPerValue == 2,
                "Since we're using uint16_t, the bytes per sample must be 2.");
  // Skip the sequence number and treat the buffer as 16-bit values.
  uint16_t* imu_buf_data = (uint16_t*)(imu_buf.data + 4);
  uint16_t imu_buf_data_size = (imu_buf.size - 4) / kImuBytesPerValue;

  TickType_t wakeup_time = xTaskGetTickCount();
  while (true) {
    uint32_t sequence_number = le32toh(*(uint32_t*)imu_buf.data);

    // Fill buffer.
    *(uint32_t*)imu_buf.data = htole32(sequence_number + 1);
    for (uint16_t i = 0; i < imu_buf_data_size; ++i) {
      imu_buf_data[i] = htole16(i + 1);
    }

    bool sent = TryNotifyImuSubscribers();
    if (!is_sending && sent) {
      is_sending = true;
      starting_value = sequence_number;
    } else if (is_sending && !sent) {
      is_sending = false;
      const uint32_t notifications_sent_count =
          sequence_number - starting_value;
      ESP_LOGW("imusender", "Sent %d imu notifications in a row.",
               notifications_sent_count);
    }

    xTaskDelayUntil(&wakeup_time, delay_time);
  }

  vTaskDelete(NULL);
}

void SendPiezoDataTask([[maybe_unused]] void* arg) {
  CharacteristicBuffer piezo_buf = get_piezo_buf();
  const uint16_t piezo_buf_size_in_ms = BufSizeInMs(
      piezo_buf.size, kPiezoBytesPerSample, kPiezoSensorCount, kPiezoFrequency);
  const TickType_t delay_time = pdMS_TO_TICKS(piezo_buf_size_in_ms);

  bool is_sending = false;
  uint32_t starting_value = 0;

  static_assert(kPiezoBytesPerValue == 2,
                "Since we're using uint16_t, the bytes per sample must be 2.");
  // Skip the sequence number and treat the buffer as 16-bit values.
  uint16_t* piezo_buf_data = (uint16_t*)(piezo_buf.data + 4);
  uint16_t piezo_buf_data_size = (piezo_buf.size - 4) / kPiezoBytesPerSample;

  TickType_t wakeup_time = xTaskGetTickCount();
  while (true) {
    uint32_t sequence_number = le32toh(*(uint32_t*)piezo_buf.data);

    // Fill buffer.
    *(uint32_t*)piezo_buf.data = htole32(sequence_number + 1);
    for (uint16_t i = 0; i < piezo_buf_data_size; ++i) {
      piezo_buf_data[i] = htole16(i + 1);
    }

    bool sent = TryNotifyPiezoSubscribers();

    if (!is_sending && sent) {
      is_sending = true;
      starting_value = sequence_number;
    } else if (is_sending && !sent) {
      is_sending = false;
      const uint32_t notifications_sent_count =
          sequence_number - starting_value;
      ESP_LOGW("piezosender", "Sent %d piezo notifications in a row.",
               notifications_sent_count);
    }

    xTaskDelayUntil(&wakeup_time, delay_time);
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
