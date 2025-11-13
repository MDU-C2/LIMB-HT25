#pragma once

/**
 * @file app_config.h
 * @brief Centralized configuration for hardware mappings and application parameters.
 */

#include "sdkconfig.h"

#include "driver/gpio.h"
#include "driver/i2c.h"

// -------------------- CAN Interface --------------------
#define CAN_TX_GPIO                    (GPIO_NUM_4)
#define CAN_RX_GPIO                    (GPIO_NUM_5)

#define CAN_BAUD_RATE_DEFAULT          (200000)  // 200 kbps default
#define CAN_BAUD_RATE_MIN              (100000)
#define CAN_BAUD_RATE_MAX              (500000)

// -------------------- I2C / IMU --------------------
#define IMU_I2C_PORT                   (I2C_NUM_0)
#define IMU_I2C_SDA_GPIO               (GPIO_NUM_8)
#define IMU_I2C_SCL_GPIO               (GPIO_NUM_9)
#define IMU_I2C_FREQ_HZ                (400000)

// LSM6DSO32 specific constants
#define LSM6DSO32_I2C_ADDR             (0x6A)

// -------------------- Stepper / Motion --------------------
#define STEPPER_STEP_GPIO              (GPIO_NUM_6)
#define STEPPER_DIR_GPIO               (GPIO_NUM_7)
#define STEPPER_ENABLE_GPIO            (GPIO_NUM_10)

#define MOTION_CONTROL_PERIOD_MS       (10)       // Control loop period

#define DEFAULT_MICROSTEPPING          (16)
#define DEFAULT_STEPS_PER_REV          (200)      // full-step steps per revolution
#define DEFAULT_GEAR_RATIO             (50.0f)    // gear ratio (motor revs per joint rev)

// Motion profile parameters
#define DEFAULT_MAX_VELOCITY_DPS      (30.0f)     // Maximum velocity in degrees per second
#define DEFAULT_MIN_VELOCITY_DPS       (2.0f)      // Minimum reliable velocity while moving
#define DEFAULT_MAX_ACCEL_DPS2         (100.0f)    // Maximum acceleration in degrees per second squared
#define DEFAULT_DEADBAND_DEG           (2.0f)      // Stop window in degrees

#define MAX_JOINT_ANGLE_DEG            (180.0f)

// -------------------- Task Configuration --------------------
#define CAN_RX_TASK_STACK              (4096)
#define CAN_RX_TASK_PRIO               (5)

#define CAN_TX_TASK_STACK              (4096)
#define CAN_TX_TASK_PRIO               (4)

#define IMU_TASK_STACK                 (4096)
#define IMU_TASK_PRIO                  (5)

#define MOTION_TASK_STACK              (4096)
#define MOTION_TASK_PRIO               (6)

#define IMU_UPDATE_HZ                  (100)
#define CAN_STATUS_HZ                  (20)

// -------------------- CAN Identifiers --------------------
#define CAN_ID_IMU_ORIENT              (0x101)
#define CAN_ID_ARM_CMD                 (0x201)
#define CAN_ID_ARM_ESTOP               (0x202)
#define CAN_ID_ARM_STATUS              (0x181)

// -------------------- Safety Parameters --------------------
#define EMERGENCY_STOP_TIMEOUT_MS      (500)

