#include <stdio.h>
#include <math.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "imu.h"

static const char *TAG = "IMU_MOVEMENT";

void app_main(void)
{
    // Disable most ESP-IDF logs, but keep errors for IMU component
    esp_log_level_set("*", ESP_LOG_ERROR);
    esp_log_level_set("IMU", ESP_LOG_INFO);
    
    // Initialize IMU with default config
    imu_config_t config = IMU_CONFIG_DEFAULT();
    config.sda_pin = 7;
    config.scl_pin = 6;
    
    printf("Initializing IMU: SDA=%d, SCL=%d, addr=0x%02X\n", 
           config.sda_pin, config.scl_pin, config.sensor_addr);
    
    esp_err_t ret = imu_init(&config);
    if (ret != ESP_OK) {
        printf("ERROR: IMU init failed: %s (0x%x)\n", esp_err_to_name(ret), ret);
        printf("Check:\n");
        printf("  1. I2C pins are correct (SDA=%d, SCL=%d)\n", config.sda_pin, config.scl_pin);
        printf("  2. IMU sensor is connected\n");
        printf("  3. Pull-up resistors are present\n");
        return;
    }
    
    printf("IMU initialized successfully\n");
    
    // Small delay for sensor to stabilize
    vTaskDelay(pdMS_TO_TICKS(100));

    imu_data_t data;
    
    while (1) {
        ret = imu_read_data(&data);
        if (ret == ESP_OK) {
            // Convert raw int16_t to physical units
            // Gyro: ±250 dps -> rad/s
            float gx = (float)data.gyro.x * 250.0f / 32768.0f * (M_PI / 180.0f);
            float gy = (float)data.gyro.y * 250.0f / 32768.0f * (M_PI / 180.0f);
            float gz = (float)data.gyro.z * 250.0f / 32768.0f * (M_PI / 180.0f);
            
            // Accel: ±4g -> m/s²
            float ax = (float)data.accel.x * 4.0f / 32768.0f * 9.81f;
            float ay = (float)data.accel.y * 4.0f / 32768.0f * 9.81f;
            float az = (float)data.accel.z * 4.0f / 32768.0f * 9.81f;
            
            // Send JSON via serial (printf goes to UART)
            // Use proper format specifiers to ensure negative numbers print correctly
            printf("{\"accel\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f},\"gyro\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f}}\n",
                   (double)ax, (double)ay, (double)az, (double)gx, (double)gy, (double)gz);
            fflush(stdout); // Ensure data is sent immediately
        } else {
            printf("ERROR: IMU read failed: %d\n", ret);
        }
        
        vTaskDelay(pdMS_TO_TICKS(10)); // 100Hz
    }
}

