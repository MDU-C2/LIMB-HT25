#include <stdio.h>
#include <math.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "imu.h"
#include "limb_ble_periph.h"
#include "sensors_service.h"

static const char *TAG = "IMU_MOVEMENT";

// Sequence number for BLE packets
static uint32_t g_imu_sequence_number = 0;

/**
 * @brief Task that reads IMU data and sends it via BLE
 */
void SendImuDataTask(void* arg)
{
    ESP_LOGI(TAG, "IMU data sending task started");
    
    // Small delay to ensure BLE is initialized
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    imu_data_t data;
    const TickType_t xDelay = pdMS_TO_TICKS(10); // 100Hz = 10ms delay
    
    while (1) {
        esp_err_t ret = imu_read_data(&data);
        if (ret == ESP_OK) {
            // Get IMU buffer from BLE service
            CharacteristicBuffer imu_buf = get_imu_buf();
            
            // Format: [4 bytes sequence number][sensor data]
            // Sensor data format: For each sample, for each sensor:
            //   gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z (each int16_t, little-endian)
            
            // Write sequence number (uint32_t, little-endian)
            memcpy(imu_buf.data, &g_imu_sequence_number, 4);
            
            // According to sensors_service.h:
            // - kImuSamplesToSend = 1 (one sample per packet)
            // - kImuSensorCount = 2 (but we only have 1 sensor)
            // - kImuValuesPerSample = 6 (gyro x,y,z + accel x,y,z)
            // - kImuBytesPerValue = 2 (int16_t)
            
            // For now, we'll send data for 1 sensor and duplicate it for sensor 2
            // The data layout for 1 sample with 2 sensors:
            // Sensor 1: gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z (12 bytes)
            // Sensor 2: gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z (12 bytes)
            
            uint8_t* sensor_data = imu_buf.data + 4; // Skip sequence number
            
            // Write sensor 1 data (gyro then accel)
            memcpy(sensor_data + 0, &data.gyro.x, 2);  // gyro.x (int16_t, little-endian)
            memcpy(sensor_data + 2, &data.gyro.y, 2);  // gyro.y
            memcpy(sensor_data + 4, &data.gyro.z, 2);  // gyro.z
            memcpy(sensor_data + 6, &data.accel.x, 2); // accel.x
            memcpy(sensor_data + 8, &data.accel.y, 2); // accel.y
            memcpy(sensor_data + 10, &data.accel.z, 2); // accel.z
            
            // Duplicate for sensor 2 (since kImuSensorCount = 2)
            // memcpy(sensor_data + 12, sensor_data, 12);
            
            // Send notification to BLE subscribers
            bool sent = TryNotifyImuSubscribers();
            if (sent) {
                ESP_LOGD(TAG, "IMU data sent, sequence: %lu", g_imu_sequence_number);
            }
            
            g_imu_sequence_number++;
        } else {
            ESP_LOGW(TAG, "IMU read failed: %s", esp_err_to_name(ret));
        }
        
        vTaskDelay(xDelay);
    }
    
    vTaskDelete(NULL);
}

void app_main(void)
{
    // Set log levels
    esp_log_level_set("*", ESP_LOG_WARN);
    esp_log_level_set("IMU", ESP_LOG_INFO);
    esp_log_level_set("IMU_MOVEMENT", ESP_LOG_INFO);
    esp_log_level_set("LIMB BLE Periph", ESP_LOG_INFO);
    
    ESP_LOGI(TAG, "Starting IMU Movement with BLE");
    
    // Initialize IMU with default config
    imu_config_t config = IMU_CONFIG_DEFAULT();
    config.sda_pin = 5;
    config.scl_pin = 4;
    
    ESP_LOGI(TAG, "Initializing IMU: SDA=%d, SCL=%d, addr=0x%02X", 
             config.sda_pin, config.scl_pin, config.sensor_addr);
    
    esp_err_t ret = imu_init(&config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "IMU init failed: %s (0x%x)", esp_err_to_name(ret), ret);
        ESP_LOGE(TAG, "Check:");
        ESP_LOGE(TAG, "  1. I2C pins are correct (SDA=%d, SCL=%d)", config.sda_pin, config.scl_pin);
        ESP_LOGE(TAG, "  2. IMU sensor is connected");
        ESP_LOGE(TAG, "  3. Pull-up resistors are present");
        return;
    }
    
    ESP_LOGI(TAG, "IMU initialized successfully");
    
    // Small delay for sensor to stabilize
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Start BLE task
    ESP_LOGI(TAG, "Starting BLE task...");
    xTaskCreate(BleTask, "BleTask", 4096, NULL, 5, NULL);
    
    // Start IMU data sending task
    ESP_LOGI(TAG, "Starting IMU data sending task...");
    xTaskCreate(SendImuDataTask, "SendImuDataTask", 4096, NULL, 5, NULL);
    
    ESP_LOGI(TAG, "System started. IMU data will be sent via BLE.");
    ESP_LOGI(TAG, "Device name: LIMBServer");
    ESP_LOGI(TAG, "Waiting for BLE connection...");
}

