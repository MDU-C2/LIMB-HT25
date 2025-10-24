#pragma once
#include "host/ble_gap.h"
#include "host/ble_gatt.h"

// The settings for the sensors used. These are used to calculate the buffer
// sizes for the different characteristics.
enum {
  kEmgFrequency = 4000,
  kEmgMsPerWindow = 200,
  kEmgMsPerOverlap = 50,
  kEmgBytesPerSample = 2,
  kEmgSensorCount = 2,

  kImuFrequency = 100,
  kImuMsPerWindow = 200,
  kImuMsPerOverlap = 50,
  kImuBytesPerSample = 12,
  kImuSensorCount = 2,

  kPiezoFrequency = 100,
  kPiezoMsPerWindow = 200,
  kPiezoMsPerOverlap = 50,
  kPiezoBytesPerSample = 2,
  kPiezoSensorCount = 1,
};

// Constants to determine the characteristic buffer sizes.
enum {
  // The amount of the new samples in a window that should be buffered before
  // being sent. E.g. 30 means one 30th of the new samples are buffered before
  // being sent.
  kPartOfWindowPerSend = 15,

  kEmgSamplesPerWindow = kEmgMsPerWindow * kEmgFrequency / 1000,
  kEmgSamplesPerOverlap = kEmgMsPerOverlap * kEmgFrequency / 1000,
  kEmgNewSamplesPerWindow = kEmgSamplesPerWindow - kEmgSamplesPerOverlap,
  kEmgSamplesToSend = kEmgNewSamplesPerWindow / kPartOfWindowPerSend,
  kEmgBufSize = kEmgSamplesToSend * kEmgBytesPerSample * kEmgSensorCount,

  kImuSamplesPerWindow = kImuMsPerWindow * kImuFrequency / 1000,
  kImuSamplesPerOverlap = kImuMsPerOverlap * kImuFrequency / 1000,
  kImuNewSamplesPerWindow = kImuSamplesPerWindow - kImuSamplesPerOverlap,
  kImuSamplesToSend = kImuNewSamplesPerWindow / kPartOfWindowPerSend,
  kImuBufSize = kImuSamplesToSend * kImuBytesPerSample * kImuSensorCount,

  kPiezoSamplesPerWindow = kPiezoMsPerWindow * kPiezoFrequency / 1000,
  kPiezoSamplesPerOverlap = kPiezoMsPerOverlap * kPiezoFrequency / 1000,
  kPiezoNewSamplesPerWindow = kPiezoSamplesPerWindow - kPiezoSamplesPerOverlap,
  kPiezoSamplesToSend = kPiezoNewSamplesPerWindow / kPartOfWindowPerSend,
  kPiezoBufSize =
      kPiezoSamplesToSend * kPiezoBytesPerSample * kPiezoSensorCount,
};

typedef struct {
  uint8_t* data;
  uint16_t size;
} CharacteristicBuffer;

// Gets an array containing the sensors service.
const struct ble_gatt_svc_def* get_sensor_services(void);

// Establish a subscription for the characteristic specified by the GAP event.
void SensorSubscribe(struct ble_gap_event* event);

// Get the buffer used when sending EMG data.
CharacteristicBuffer get_emg_buf(void);
// Notify subscribers of current EMG data.
bool TryNotifyEmgSubscribers(void);

// Get the buffer used when sending IMU data.
CharacteristicBuffer get_imu_buf(void);
// Notify subscribers of current IMU data.
bool TryNotifyImuSubscribers(void);

// Get the buffer used when sending piezo data.
CharacteristicBuffer get_piezo_buf(void);
// Notify subscribers of current piezo data.
bool TryNotifyPiezoSubscribers(void);
