#pragma once
#include "host/ble_gap.h"
#include "host/ble_gatt.h"

// The settings for the sensors used. These are used to calculate the buffer
// sizes for the different characteristics.
enum {
  kEmgFrequency = 5000,
  kEmgMsPerWindow = 200,
  kEmgMsPerOverlap = 50,
  kEmgBytesPerSample = 2,
  kEmgSensorCount = 2,

  kImuFrequency = 1000,
  kImuMsPerWindow = 200,
  kImuMsPerOverlap = 50,
  kImuBytesPerSample = 12,
  kImuSensorCount = 2,

  kPiezoFrequency = 5000,
  kPiezoMsPerWindow = 200,
  kPiezoMsPerOverlap = 50,
  kPiezoBytesPerSample = 2,
  kPiezoSensorCount = 1,
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


