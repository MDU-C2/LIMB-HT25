#pragma once

#include <stdint.h>

#include "esp_err.h"

// Values with CAN ID bits set that represent a certain message type.
typedef enum {
  CAN_MESSAGE_TYPE_STOP = 0x100,           // 0b001'0000'0000
  CAN_MESSAGE_TYPE_ACTUATION = 0x200,      // 0b010'0000'0000
  CAN_MESSAGE_TYPE_POTENTIOMETER = 0x400,  // 0b100'0000'0000
  CAN_MESSAGE_TYPE_IMU = 0x500,            // 0b101'0000'0000
  CAN_MESSAGE_TYPE_PRESSURE = 0x700,       // 0b111'0000'0000
} CanMessageType;

// Mask with CAN ID bits set to filter certain message types.
typedef enum {
  CAN_MESSAGE_TYPE_FILTER_ANY = 0x700,    // 0b111'0000'0000
  CAN_MESSAGE_TYPE_FILTER_EXACT = 0x000,  // 0b000'0000'0000
} CanMessageTypeFilterMask;

// Values with CAN ID bits set that represent a certain CAN recipient.
typedef enum {
  CAN_RECIPIENT_ROBOT_SHOULDER = 0x020,       // 0b000'0010'0000
  CAN_RECIPIENT_ROBOT_ELBOW = 0x040,          // 0b000'0100'0000
  CAN_RECIPIENT_ROBOT_WRIST = 0x060,          // 0b000'0110'0000
  CAN_RECIPIENT_ROBOT_HAND = 0x080,           // 0b000'1000'0000
  CAN_RECIPIENT_ROBOT_MAIN_COMPUTER = 0x0A0,  // 0b000'1010'0000
} CanRecipientNode;

// Mask with CAN ID bits set to filter certain CAN recipients.
typedef enum {
  CAN_RECIPIENT_FILTER_ANY = 0x0E0,    // 0b000'1110'0000
  CAN_RECIPIENT_FILTER_EXACT = 0x000,  // 0b000'0000'0000
} CanRecipientNodeFilterMask;

// Mask with CAN ID bits set to filter the generic ID part of the CAN ID.
typedef enum {
  CAN_GENERIC_FILTER_ANY = 0x01F,    // 0b000'0001'1111
  CAN_GENERIC_FILTER_EXACT = 0x000,  // 0b000'0000'0000
} CanGenericFilterMask;

// The CAN IDs that correspond to the different messages we use for the LIMB
// project.
typedef enum {
  // Stop messages.
  CAN_ID_ROBOT_SHOULDER_UP_DOWN_STOP = 0x120,     // 0b001'0010'0000
  CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_STOP = 0x121,  // 0b001'0010'0001
  CAN_ID_ROBOT_UPPER_ARM_ROTATION_STOP = 0x122,   // 0b001'0010'0010
  CAN_ID_ROBOT_ELBOW_UP_DOWN_STOP = 0x140,        // 0b001'0100'0000
  CAN_ID_ROBOT_LOWER_ARM_ROTATION_STOP = 0x160,   // 0b001'0110'0000
  CAN_ID_ROBOT_FINGERS_STOP = 0x161,              // 0b001'0110'0001
  CAN_ID_ROBOT_THUMB_STOP = 0x162,                // 0b001'0110'0010
  CAN_ID_ROBOT_INDEX_STOP = 0x163,                // 0b001'0110'0011
  CAN_ID_ROBOT_MIDDLE_STOP = 0x164,               // 0b001'0110'0100
  CAN_ID_ROBOT_RING_STOP = 0x165,                 // 0b001'0110'0101
  CAN_ID_ROBOT_PINKY_STOP = 0x166,                // 0b001'0110'0110

  // Actuation messages.
  CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION = 0x220,     // 0b010'0010'0000
  CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION = 0x221,  // 0b010'0010'0001
  CAN_ID_ROBOT_UPPER_ARM_ROTATION_ACTUATION = 0x222,   // 0b010'0010'0010
  CAN_ID_ROBOT_ELBOW_UP_DOWN_ACTUATION = 0x240,        // 0b010'0100'0000
  CAN_ID_ROBOT_LOWER_ARM_ROTATION_ACTUATION = 0x260,   // 0b010'0110'0000
  CAN_ID_ROBOT_THUMB_ACTUATION = 0x261,                // 0b010'0110'0001
  CAN_ID_ROBOT_INDEX_ACTUATION = 0x262,                // 0b010'0110'0010
  CAN_ID_ROBOT_MIDDLE_ACTUATION = 0x263,               // 0b010'0110'0011
  CAN_ID_ROBOT_RING_ACTUATION = 0x264,                 // 0b010'0110'0100
  CAN_ID_ROBOT_PINKY_ACTUATION = 0x265,                // 0b010'0110'0101
  CAN_ID_ROBOT_HAND_SET_GRIP_STATE = 0x266,            // 0b010'0110'0110

  // Potentiometer messages.
  CAN_ID_ROBOT_ELBOW_UP_DOWN_POTENTIOMETER = 0x4A0,        // 0b100'1010'0000
  CAN_ID_ROBOT_UPPER_ARM_ROTATION_POTENTIOMETER = 0x4A1,   // 0b100'1010'0001
  CAN_ID_ROBOT_SHOULDER_UP_DOWN_POTENTIOMETER = 0x4A2,     // 0b100'1010'0010
  CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_POTENTIOMETER = 0x4A3,  // 0b100'1010'0011

  // IMU messages.
  CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_PITCH = 0x5A0,  // 0b101'1010'0000
  CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_ROLL = 0x5A1,   // 0b101'1010'0001
  CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_YAW = 0x5A2,    // 0b101'1010'0010
  CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_X = 0x5A3,     // 0b101'1010'0011
  CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Y = 0x5A4,     // 0b101'1010'0100
  CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Z = 0x5A5,     // 0b101'1010'0101
  CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_PITCH = 0x5A6,     // 0b101'1010'0110
  CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_ROLL = 0x5A7,      // 0b101'1010'0111
  CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_YAW = 0x5A8,       // 0b101'1010'1000
  CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_X = 0x5A9,        // 0b101'1010'1001
  CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_Y = 0x5AA,        // 0b101'1010'1010
  CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_Z = 0x5AB,        // 0b101'1010'1011
  CAN_ID_ROBOT_HAND_IMU_GYRO_PITCH = 0x5AC,      // 0b101'1010'1100
  CAN_ID_ROBOT_HAND_IMU_GYRO_ROLL = 0x5AD,       // 0b101'1010'1101
  CAN_ID_ROBOT_HAND_IMU_GYRO_YAW = 0x5AE,        // 0b101'1010'1110
  CAN_ID_ROBOT_HAND_IMU_ACCEL_X = 0x5AF,         // 0b101'1010'1111
  CAN_ID_ROBOT_HAND_IMU_ACCEL_Y = 0x5B0,         // 0b101'1011'0000
  CAN_ID_ROBOT_HAND_IMU_ACCEL_Z = 0x5B1,         // 0b101'1011'0001

  // Pressure sensor messages.
  CAN_ID_ROBOT_THUMB_PRESSURE = 0x7A0,   // 0b111'1010'0000
  CAN_ID_ROBOT_INDEX_PRESSURE = 0x7A1,   // 0b111'1010'0001
  CAN_ID_ROBOT_MIDDLE_PRESSURE = 0x7A2,  // 0b111'1010'0010
  CAN_ID_ROBOT_RING_PRESSURE = 0x7A3,    // 0b111'1010'0011
  CAN_ID_ROBOT_PINKY_PRESSURE = 0x7A4,   // 0b111'1010'0100
} CanMessageId;

enum {
  CAN_MAX_MESSAGE_SIZE = 8,
};

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
uint16_t create_filter_mask(
    CanMessageTypeFilterMask msg_type_filter_mask,
    CanRecipientNodeFilterMask recipient_node_filter_mask,
    CanGenericFilterMask generic_filter_mask);

esp_err_t can_automatically_reenable_on_bus_off(void);

// Initialize the CAN bus
esp_err_t can_init(int tx_pin, int rx_pin, int baudrate,
                   const CanMsgFilter* filter);

// Send a CAN message
esp_err_t can_send(uint32_t id, const uint8_t* data, uint8_t len,
                   uint32_t ms_to_wait);

// Try to receive a message (blocking for timeout_ms)
esp_err_t can_receive(uint32_t* id, uint8_t* data, uint8_t* len,
                      int timeout_ms);

// Stop and deinitialize the node
void can_deinit(void);
