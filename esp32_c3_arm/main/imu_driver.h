#pragma once

/**
 * @file imu_driver.h
 * @brief LSM6DSO32 IMU helper functions.
 */

#include "freertos/FreeRTOS.h"
#include "esp_err.h"
#include "driver/i2c.h"

#include "app_types.h"

typedef struct {
    i2c_port_t i2c_port;
    gpio_num_t sda_gpio;
    gpio_num_t scl_gpio;
    uint32_t clk_speed_hz;
    uint8_t i2c_addr;
} imu_driver_config_t;

typedef struct {
    int16_t accel_raw[3];
    int16_t gyro_raw[3];
    int16_t temperature_raw;
} imu_raw_sample_t;

esp_err_t imu_driver_init(const imu_driver_config_t *config);

esp_err_t imu_driver_read_raw(imu_raw_sample_t *sample);

esp_err_t imu_driver_read_orientation(float dt_seconds, imu_orientation_t *orientation_out);

esp_err_t imu_driver_read_temperature(float *temperature_celsius);

