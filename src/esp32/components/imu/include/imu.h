#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "driver/i2c.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief IMU data structure
 */
typedef struct {
    float x;
    float y;
    float z;
} imu_vector_t;

/**
 * @brief Complete IMU sensor data
 */
typedef struct {
    imu_vector_t accel;      // Accelerometer data in m/s²
    imu_vector_t gyro;       // Gyroscope data in rad/s
} imu_data_t;

/**
 * @brief IMU configuration structure
 */
typedef struct {
    i2c_port_t i2c_port;          // I2C port number
    uint8_t sda_pin;              // GPIO number for SDA
    uint8_t scl_pin;              // GPIO number for SCL
    uint32_t i2c_freq_hz;         // I2C clock frequency in Hz
    uint8_t sensor_addr;          // I2C address of the sensor (default: 0x6A)
    uint8_t accel_range;          // Accelerometer range (2, 4, 8, or 16 g)
    uint8_t gyro_range;           // Gyroscope range (125, 250, 500, 1000, or 2000 dps)
    uint8_t accel_odr;            // Accelerometer output data rate (0-7, see datasheet)
    uint8_t gyro_odr;             // Gyroscope output data rate (0-7, see datasheet)
} imu_config_t;

/**
 * @brief Default IMU configuration
 */
#define IMU_CONFIG_DEFAULT() { \
    .i2c_port = I2C_NUM_0, \
    .sda_pin = 4, \
    .scl_pin = 5, \
    .i2c_freq_hz = 400000, \
    .sensor_addr = 0x6A, \
    .accel_range = 4, \
    .gyro_range = 250, \
    .accel_odr = 0x50, \
    .gyro_odr = 0x50, \
}

/**
 * @brief Initialize the IMU component
 *
 * @param config Configuration structure for the IMU
 * @return esp_err_t ESP_OK on success
 */
esp_err_t imu_init(const imu_config_t *config);

/**
 * @brief Deinitialize the IMU component
 *
 * @return esp_err_t ESP_OK on success
 */
esp_err_t imu_deinit(void);

/**
 * @brief Read IMU sensor data
 *
 * @param data Pointer to imu_data_t structure to store the data
 * @return esp_err_t ESP_OK on success
 */
esp_err_t imu_read_data(imu_data_t *data);

/**
 * @brief Check if IMU sensor is present and responding
 *
 * @return true if sensor is detected, false otherwise
 */
bool imu_is_present(void);

