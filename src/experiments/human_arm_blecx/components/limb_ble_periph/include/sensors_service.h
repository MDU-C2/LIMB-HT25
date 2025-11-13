#pragma once

#include "gap.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"

// The settings for the sensors used. These are used to calculate the buffer
// sizes for the different characteristics.
enum {
  kEmgFrequency = 4000,
  kEmgMsPerWindow = 100,
  kEmgMsPerOverlap = 0,
  kEmgBytesPerValue = 2,
  kEmgValuesPerSample = 1,
  kEmgBytesPerSample = kEmgBytesPerValue * kEmgValuesPerSample,
  kEmgSensorCount = 1,

  kImuFrequency = 100,
  kImuMsPerWindow = 100,
  kImuMsPerOverlap = 0,
  kImuBytesPerValue = 4,
  kImuValuesPerSample = 9,
  kImuBytesPerSample = kImuBytesPerValue * kImuValuesPerSample,
  kImuSensorCount = 1,

  kPiezoFrequency = 100,
  kPiezoMsPerWindow = 100,
  kPiezoMsPerOverlap = 0,
  kPiezoBytesPerValue = 2,
  kPiezoValuesPerSample = 1,
  kPiezoBytesPerSample = kPiezoBytesPerValue * kPiezoValuesPerSample,
  kPiezoSensorCount = 1,
};

// Constants to determine the characteristic buffer sizes.
enum {
  // The amount of the new samples in a window that should be buffered before
  // being sent. E.g. 30 means one 30th of the new samples are buffered before
  // being sent.
  kPartOfWindowPerSend = 10,
  kSequenceNumberSize = 4,

  kEmgSamplesPerWindow = kEmgMsPerWindow * kEmgFrequency / 1000,
  kEmgSamplesPerOverlap = kEmgMsPerOverlap * kEmgFrequency / 1000,
  kEmgNewSamplesPerWindow = kEmgSamplesPerWindow - kEmgSamplesPerOverlap,
  kEmgSamplesToSend = kEmgNewSamplesPerWindow / kPartOfWindowPerSend,
  kEmgBufSize = (kEmgSamplesToSend * kEmgBytesPerSample * kEmgSensorCount) +
                kSequenceNumberSize,
  kEmgPacketSendRateHz = kEmgFrequency / kEmgSamplesToSend,

  kImuSamplesPerWindow = kImuMsPerWindow * kImuFrequency / 1000,
  kImuSamplesPerOverlap = kImuMsPerOverlap * kImuFrequency / 1000,
  kImuNewSamplesPerWindow = kImuSamplesPerWindow - kImuSamplesPerOverlap,
  kImuSamplesToSend = kImuNewSamplesPerWindow / kPartOfWindowPerSend,
  kImuBufSize = (kImuSamplesToSend * kImuBytesPerSample * kImuSensorCount) +
                kSequenceNumberSize,
  kImuPacketSendRateHz = kImuFrequency / kImuSamplesToSend,

  kPiezoSamplesPerWindow = kPiezoMsPerWindow * kPiezoFrequency / 1000,
  kPiezoSamplesPerOverlap = kPiezoMsPerOverlap * kPiezoFrequency / 1000,
  kPiezoNewSamplesPerWindow = kPiezoSamplesPerWindow - kPiezoSamplesPerOverlap,
  kPiezoSamplesToSend = kPiezoNewSamplesPerWindow / kPartOfWindowPerSend,
  kPiezoBufSize =
      (kPiezoSamplesToSend * kPiezoBytesPerSample * kPiezoSensorCount) +
      kSequenceNumberSize,
  kPiezoPacketSendRateHz = kPiezoFrequency / kPiezoSamplesToSend,
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

// Sanity checks using static asserts.

// We're comparing between enums defined in different places, but they represent
// general buffer size constants, so it's fine.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wenum-compare"
static_assert(kEmgBufSize <= kMaxAttDataSize,
              "The sensor buffer sizes shouldn't exceed the max ATT data size "
              "to avoid splitting the data into multiple packets.");
static_assert(kImuBufSize <= kMaxAttDataSize,
              "The sensor buffer sizes shouldn't exceed the max ATT data size "
              "to avoid splitting the data into multiple packets.");
static_assert(kPiezoBufSize <= kMaxAttDataSize,
              "The sensor buffer sizes shouldn't exceed the max ATT data size "
              "to avoid splitting the data into multiple packets.");
#pragma GCC diagnostic pop

// Since we make decisions based on time windows of sensor readings, we want to
// send the same rate of packets for the different sensors.
static_assert(kEmgPacketSendRateHz == kImuPacketSendRateHz &&
                  kImuPacketSendRateHz == kPiezoPacketSendRateHz,
              "The send rates for the sensor readings should be the same.");

// Helper to include the variable name in the static_assert message.
#define LIMB_STRINGIFY(x) #x

// The part of the window to send must be a common factor between the different
// sensor samples sent per packet (i.e. there shouldn't be any truncation when
// dividing by it).
static_assert(
    (kEmgNewSamplesPerWindow / kPartOfWindowPerSend * kPartOfWindowPerSend) ==
        kEmgNewSamplesPerWindow,
    LIMB_STRINGIFY(kPartOfWindowPerSend) " must be a factor of " LIMB_STRINGIFY(
        kEmgNewSamplesPerWindow) ".");
static_assert(
    (kImuNewSamplesPerWindow / kPartOfWindowPerSend * kPartOfWindowPerSend) ==
        kImuNewSamplesPerWindow,
    LIMB_STRINGIFY(kPartOfWindowPerSend) " must be a factor of " LIMB_STRINGIFY(
        kImuNewSamplesPerWindow) ".");
static_assert(
    (kPiezoNewSamplesPerWindow / kPartOfWindowPerSend * kPartOfWindowPerSend) ==
        kPiezoNewSamplesPerWindow,
    LIMB_STRINGIFY(kPartOfWindowPerSend) " must be a factor of " LIMB_STRINGIFY(
        kPiezoNewSamplesPerWindow) ".");
