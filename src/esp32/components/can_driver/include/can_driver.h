#pragma once

#include "esp_err.h"
#include <stdint.h>

// Values with CAN ID bits set that represent a certain message type.
typedef enum {
  CAN_MESSAGE_TYPE_STOP = 0x100,
  CAN_MESSAGE_TYPE_ACTUATION = 0x200,
  CAN_MESSAGE_TYPE_EMG = 0x300,
  CAN_MESSAGE_TYPE_POTENTIOMETER = 0x400,
  CAN_MESSAGE_TYPE_IMU = 0x500,
  CAN_MESSAGE_TYPE_PIEZO = 0x600,
  CAN_MESSAGE_TYPE_PRESSURE = 0x700,
} CanMessageType;

// Mask with CAN ID bits set to filter certain message types.
typedef enum {
  CAN_MESSAGE_TYPE_FILTER_ANY = 0x700,
  CAN_MESSAGE_TYPE_FILTER_EXACT = 0x000,
} CanMessageTypeFilterMask;

// Values with CAN ID bits set that represent a certain CAN recipient.
typedef enum {
  CAN_RECIPIENT_ROBOT_SHOULDER = 0x020,
  CAN_RECIPIENT_ROBOT_ELBOW = 0x040,
  CAN_RECIPIENT_ROBOT_WRIST = 0x060,
  CAN_RECIPIENT_ROBOT_HAND = 0x080,
  CAN_RECIPIENT_ROBOT_MAIN_COMPUTER = 0x0A0,
  CAN_RECIPIENT_HUMAN_UPPER_ARM = 0x0C0,
} CanRecipientNode;

// Mask with CAN ID bits set to filter certain CAN recipients.
typedef enum {
  CAN_RECIPIENT_FILTER_ANY = 0x0E0,
  CAN_RECIPIENT_FILTER_EXACT = 0x000,
} CanRecipientNodeFilterMask;

// Mask with CAN ID bits set to filter the generic ID part of the CAN ID.
typedef enum {
  CAN_GENERIC_FILTER_ANY = 0x01F,
  CAN_GENERIC_FILTER_EXACT = 0x000,
} CanGenericFilterMask;

// The CAN IDs that correspond to the different messages we use for the LIMB project.
typedef enum {
  // Stop messages.
  CAN_ID_ROBOT_SHOULDER_FRONT_BACK_STOP = 0x120,
  CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_STOP = 0x121,
  CAN_ID_ROBOT_ELBOW_UP_DOWN_STOP = 0x140,
  CAN_ID_ROBOT_UPPER_ARM_ROTATION_STOP = 0x141,
  CAN_ID_ROBOT_LOWER_ARM_ROTATION_STOP = 0x160,
  CAN_ID_ROBOT_FINGERS_STOP = 0x180,
  CAN_ID_ROBOT_THUMB_STOP = 0x181,
  CAN_ID_ROBOT_INDEX_STOP = 0x182,
  CAN_ID_ROBOT_MIDDLE_STOP = 0x183,
  CAN_ID_ROBOT_RING_STOP = 0x184,
  CAN_ID_ROBOT_PINKY_STOP = 0x185,

  // Actuation messages.
  CAN_ID_ROBOT_SHOULDER_FRONT_BACK_ACTUATION = 0x220,
  CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION = 0x221,
  CAN_ID_ROBOT_ELBOW_UP_DOWN_ACTUATION = 0x240,
  CAN_ID_ROBOT_UPPER_ARM_ROTATION_ACTUATION = 0x241,
  CAN_ID_ROBOT_LOWER_ARM_ROTATION_ACTUATION = 0x260,
  CAN_ID_ROBOT_THUMB_ACTUATION = 0x281,
  CAN_ID_ROBOT_INDEX_ACTUATION = 0x282,
  CAN_ID_ROBOT_MIDDLE_ACTUATION = 0x283,
  CAN_ID_ROBOT_RING_ACTUATION = 0x284,
  CAN_ID_ROBOT_PINKY_ACTUATION = 0x285,
  CAN_ID_ROBOT_HAND_SET_GRIP_STATE = 0x286,

  // Potentiometer messages.
  CAN_ID_ROBOT_ELBOW_UP_DOWN_POTENTIOMETER = 0x440,
  CAN_ID_ROBOT_UPPER_ARM_ROTATION_POTENTIOMETER = 0x441,

  // IMU messages.
  CAN_ID_ROBOT_SHOULDER_IMU_GYRO = 0x520,
  CAN_ID_ROBOT_SHOULDER_IMU_ACCEL = 0x521,
  CAN_ID_ROBOT_ELBOW_IMU_GYRO = 0x540,
  CAN_ID_ROBOT_ELBOW_IMU_ACCEL = 0x541,
  CAN_ID_ROBOT_HAND_IMU_GYRO = 0x580,
  CAN_ID_ROBOT_HAND_IMU_ACCEL = 0x581,

  // Pressure sensor messages.
  CAN_ID_ROBOT_THUMB_PRESSURE = 0x780,
  CAN_ID_ROBOT_INDEX_PRESSURE = 0x781,
  CAN_ID_ROBOT_MIDDLE_PRESSURE = 0x782,
  CAN_ID_ROBOT_RING_PRESSURE = 0x783,
  CAN_ID_ROBOT_PINKY_PRESSURE = 0x784,

  // Human EMG message.
  CAN_ID_HUMAN_UPPER_ARM_EMG = 0x3C0,

  // Human IMU messages.
  CAN_ID_HUMAN_UPPER_ARM_IMU_GYRO = 0x5C0,
  CAN_ID_HUMAN_UPPER_ARM_IMU_ACCEL = 0x5C1,
} CanMessageId;

// Allows for filtering which messages to accept from the CAN bus.
// It follows the same format as in the [ESP-IDF documentation](
// https://docs.espressif.com/projects/esp-idf/en/v5.4.3/esp32c3/api-reference/peripherals/twai.html#acceptance-filter
// ), but it only filters on the id, not any other parts of the frame.
// 
// For example, if the ID is 0b000'0000'1000 and the mask is 0b000'0000'0111,
// then all IDs that match 0b000'0000'1xxx will be allowed and all other ids
// will be filtered.
typedef struct {
    uint32_t id;
    // Set bits are ignored. Unset bits MUST match.
    uint32_t ignore_mask;
} CanMsgFilter;

// A convenience function for creating a combined filter mask from the
// different types of filter mask available.
uint16_t create_filter_mask(CanMessageTypeFilterMask msg_type_filter_mask,
                            CanRecipientNodeFilterMask recipient_node_filter_mask,
                            CanGenericFilterMask generic_filter_mask);

// Initialize the CAN bus
esp_err_t can_init(int tx_pin, int rx_pin, int baudrate, const CanMsgFilter* filter);

// Send a CAN message
esp_err_t can_send(uint32_t id, const uint8_t *data, uint8_t len);

// Try to receive a message (blocking for timeout_ms)
esp_err_t can_receive(uint32_t *id, uint8_t *data, uint8_t *len, int timeout_ms);

// Stop and deinitialize the node
void can_deinit(void);
