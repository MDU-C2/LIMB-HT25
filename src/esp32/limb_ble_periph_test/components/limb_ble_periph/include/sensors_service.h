#pragma once
#include "host/ble_gap.h"
#include "host/ble_gatt.h"

// The sizes of the characteristic buffers returned by get_*_buf.
enum {
  kPartOfWindowPerSend = 30,

  kEmgFrequency = 5000,
  kEmgSamplesPerWindow = kEmgFrequency / 5,
  kEmgWindowOverlapDivisor = 4,
  kEmgNewSamplesPerWindow =
      kEmgSamplesPerWindow - (kEmgSamplesPerWindow / kEmgWindowOverlapDivisor),
  kEmgSamplesToSend = kEmgNewSamplesPerWindow / kPartOfWindowPerSend,
  kEmgBytesPerSample = 4,
  kEmgBufSize = kEmgSamplesToSend * kEmgBytesPerSample,
  kEmgBufInMs = 1000 * kEmgSamplesToSend / kEmgFrequency,

  kImuFrequency = 1000,
  kImuSamplesPerWindow = kImuFrequency / 5,
  kImuWindowOverlapDivisor = 4,
  kImuNewSamplesPerWindow =
      kImuSamplesPerWindow - (kImuSamplesPerWindow / kImuWindowOverlapDivisor),
  kImuSamplesToSend = kImuNewSamplesPerWindow / kPartOfWindowPerSend,
  kImuBytesPerSample = 24,
  kImuBufSize = kImuSamplesToSend * kImuBytesPerSample,
  kImuBufInMs = 1000 * kImuSamplesToSend / kImuFrequency,

  kPiezoFrequency = 5000,
  kPiezoSamplesPerWindow = kPiezoFrequency / 5,
  kPiezoWindowOverlapDivisor = 4,
  kPiezoNewSamplesPerWindow =
      kPiezoSamplesPerWindow -
      (kPiezoSamplesPerWindow / kPiezoWindowOverlapDivisor),
  kPiezoSamplesToSend = kPiezoNewSamplesPerWindow / kPartOfWindowPerSend,
  kPiezoBytesPerSample = 4,
  kPiezoBufSize = kPiezoSamplesToSend * kPiezoBytesPerSample,
  kPiezoBufInMs = 1000 * kPiezoSamplesToSend / kPiezoFrequency,
};

// Gets an array containing the sensors service.
const struct ble_gatt_svc_def* get_sensor_services(void);

// Establish a subscription for the characteristic specified by the GAP event.
void SensorSubscribe(struct ble_gap_event* event);

// Get the buffer used when sending EMG data.
uint8_t* get_emg_buf(void);
// Notify subscribers of current EMG data.
bool TryNotifyEmgSubscribers(void);

// Get the buffer used when sending IMU data.
uint8_t* get_imu_buf(void);
// Notify subscribers of current IMU data.
bool TryNotifyImuSubscribers(void);

// Get the buffer used when sending piezo data.
uint8_t* get_piezo_buf(void);
// Notify subscribers of current piezo data.
bool TryNotifyPiezoSubscribers(void);
