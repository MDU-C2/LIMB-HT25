/*
 * imu_driver.h
 *
 * Public interface for the LSM6DSO32 IMU sensor driver.
 * This file defines the functions and data structures that
 * other parts of the application (like main.c) can use.
 */
#ifndef IMU_DRIVER_H
#define IMU_DRIVER_H

#include "esp_system.h" // Required for the 'esp_err_t' type

// --- Public Data Structures ---
// These structs are defined here so that main.c can create
// variables of this type to store the sensor data.

typedef struct {
    float x;
    float y;
    float z;
} imu_axis_data_t;

typedef struct {
    imu_axis_data_t accel;      // Acceleration data in m/s^2
    imu_axis_data_t gyro;       // Gyroscope data in rad/s
    float temperature;          // Temperature data in °C
} lsm6dso32_data_t;


// --- Public Function Prototypes ---
// These are the only two functions the main application
// will ever need to call.

/**
 * @brief Initializes the I2C bus and configures the LSM6DSO32 sensor.
 *
 * This function handles I2C driver installation, checks the sensor's
 * WHO_AM_I register, and configures the accelerometer and gyroscope
 * with default settings (104Hz ODR, ±4g, ±250 dps).
 *
 * @return ESP_OK on success, or ESP_FAIL if the sensor isn't found
 * or I2C fails.
 */
esp_err_t imu_init(void);

/**
 * @brief Reads the latest sensor data from the IMU.
 *
 * Fetches the raw temperature, gyroscope, and accelerometer data and
 * converts it into floating-point values with standard units.
 *
 * @param[out] data A pointer to a lsm6dso32_data_t struct that will be
 * filled with the new sensor data.
 * @return ESP_OK on success, or an error code on I2C read failure.
 */
esp_err_t imu_read_data(lsm6dso32_data_t *data);


#endif // IMU_DRIVER_H