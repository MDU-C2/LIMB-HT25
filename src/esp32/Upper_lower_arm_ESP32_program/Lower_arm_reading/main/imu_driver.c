/*
 * imu_driver.c
 *
 * Private implementation of the LSM6DSO32 driver.
 * (See imu_driver.h for public interface details).
 * * Contains all the static helper functions and low-level I2C logic
 * required to operate the sensor.
 */
#include "imu_driver.h"     // Include our own public header
#include "esp_log.h"

// --- Private Definitions ---
// These are only used inside this .c file

static const char *TAG = "IMU_DRIVER";

// --- Public Function Implementations ---
// These are the functions defined in the .h file

/**
 * @brief Initializes the I2C bus and the LSM6DSO32 sensor.
 */
esp_err_t imu_init(void)
{
    ESP_LOGI(TAG, "IMU initialization stub");
    return ESP_OK;
}

/**
 * @brief Reads a full block of sensor data (Temp, Gyro, Accel)
 * and converts it to standard physical units.
 */
esp_err_t imu_read_data(lsm6dso32_data_t *data)
{
    if (!data) return ESP_ERR_INVALID_ARG;
    
    // Placeholder: retourne des zéros
    data->accel.x = 0.0f;
    data->accel.y = 0.0f;
    data->accel.z = 0.0f;
    data->gyro.x = 0.0f;
    data->gyro.y = 0.0f;
    data->gyro.z = 0.0f;
    data->pitch = 0.0f;
    data->roll = 0.0f;
    
    return ESP_OK;
}