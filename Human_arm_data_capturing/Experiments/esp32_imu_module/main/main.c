/*
 * main.c
 *
 * Main application file. This file is responsible for
 * initializing and using the IMU driver.
 *
 * The application logic is simple:
 * 1. Initialize the IMU driver.
 * 2. In a loop, read data from the driver.
 * 3. Print the data to the console (using printf).
 */
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "imu_driver.h"    

static const char *TAG = "IMU_APP";

void app_main(void)
{
    // Step 1: Initialize the IMU driver.
    // This one function handles I2C setup, sensor check, and config.
    if (imu_init() != ESP_OK) {
        ESP_LOGE(TAG, "IMU initialization failed. Halting application.");
        return; // Stop here if the sensor failed
    }

    ESP_LOGI(TAG, "IMU driver initialized. Starting data loop...");

    // Step 2: Create a variable to hold the data
    lsm6dso32_data_t imu_data;

    // Step 3: Main application loop
    while (1) {
        // Ask the driver to fill our data struct
        if (imu_read_data(&imu_data) == ESP_OK) {
            
            // This is the application logic.
            // We replace the JSON formatting with a simple printf.
            printf("Accel: x=%.2f, y=%.2f, z=%.2f | Gyro: x=%.2f, y=%.2f, z=%.2f\n",
                   imu_data.accel.x, imu_data.accel.y, imu_data.accel.z,
                   imu_data.gyro.x, imu_data.gyro.y, imu_data.gyro.z);

        } else {
            ESP_LOGW(TAG, "Failed to read data from IMU");
        }

        // Wait 100ms before reading again (for a ~10Hz sample rate)
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}