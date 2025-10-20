/*
 * imu_driver.h
 *
 * Public interface for the LSM6DSO32 IMU sensor driver.
 *
 * This file defines the public functions and data structures that
 * other parts of the application (like main.c or other tasks)
 * can use to initialize and read data from the sensor.
 */
#ifndef IMU_DRIVER_H
#define IMU_DRIVER_H

#include "esp_system.h" // Required for the 'esp_err_t' type

// --- Public Data Structures ---
// These structs are defined here so that any file including this
// header can create variables of this type to store sensor data.

/**
 * @brief A generic structure to hold 3-axis floating-point data.
 */
typedef struct {
    float x; /**< X-axis value */
    float y; /**< Y-axis value */
    float z; /**< Z-axis value */
} imu_axis_data_t;

/**
 * @brief Holds a complete data packet from the LSM6DSO32.
 *
 * All data is converted to standard physical units.
 */
typedef struct {
    imu_axis_data_t accel;      /**< Accelerometer data in meters/second^2 (m/s^2) */
    imu_axis_data_t gyro;       /**< Gyroscope data in radians/second (rad/s) */
    float temperature;          /**< Temperature data in degrees Celsius (°C) */
} lsm6dso32_data_t;


// --- Public Function Prototypes ---
// These are the "buttons" the main application can press.
// The internal "how" is hidden in imu_driver.c.

/**
 * @brief Initializes the I2C bus and configures the LSM6DSO32 sensor.
 *
 * This function handles I2C driver installation, checks the sensor's
 * WHO_AM_I register to ensure communication, and configures the
 * accelerometer and gyroscope with default settings:
 * - 104Hz Output Data Rate (ODR)
 * - ±4g accelerometer range
 * - ±250 dps (degrees per second) gyroscope range
 *
 * @return ESP_OK on success.
 * @return ESP_FAIL if the I2C driver fails to install or if the
 * sensor is not found at the expected address.
 */
esp_err_t imu_init(void);

/**
 * @brief Reads the latest sensor data (Temp, Gyro, Accel) from the IMU.
 *
 * Fetches the raw 16-bit integer data from the sensor via I2C
 * and converts it into floating-point values with standard units
 * (m/s^2, rad/s, °C).
 *
 * @param[out] data A pointer to a lsm6dso32_data_t struct that will be
 * filled with the new sensor data.
 *
 * @return ESP_OK on success.
 * @return ESP_FAIL if the I2C read operation fails.
 */
esp_err_t imu_read_data(lsm6dso32_data_t *data);


#endif // IMU_DRIVER_H