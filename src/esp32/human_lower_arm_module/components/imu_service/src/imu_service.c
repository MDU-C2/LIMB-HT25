#include "imu_service.h"
#include "imu.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <string.h>
#include "esp_timer.h"

static const char *TAG = "IMU_SERVICE_STREAM";

// --- Internal State & Sync ---
static EventGroupHandle_t s_imu_event_group;
static imu_micro_packet_t s_imu_packet;
static uint32_t s_imu_seq = 0;
static portMUX_TYPE s_imu_mux = portMUX_INITIALIZER_UNLOCKED;

static uint8_t s_imu_active_mask = 0;
#define IMU1_ACTIVE (1 << 0)
#define IMU2_ACTIVE (1 << 1)

/**
 * @brief Periodic task for IMU data acquisition.
 * Runs at 100Hz (10ms interval) to provide smooth motion tracking.
 */
static void imu_task(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1000 / IMU_SAMPLE_RATE_HZ); 

    // Calibration/Conversion factors to map raw readings to fixed-point integers
    const float ACCEL_FIXED_FACTOR = 1.19641f;
    const float GYRO_FIXED_FACTOR = 0.15271f;

    ESP_LOGI(TAG, "IMU Streaming Task Started at 100Hz");

    while (1) {
        bool data_collected = false;
        imu_data_t raw;

        // Initialize packet metadata
        s_imu_packet.header = IMU_HEADER_MAGIC;
        s_imu_packet.seq = s_imu_seq;

        // --- Read IMU 1 (Primary) ---
        if (s_imu_active_mask & IMU1_ACTIVE) {
            if (imu_read_data(IMU_ADDR_1, &raw) == ESP_OK) {
                s_imu_packet.imu1_data[0] = (int16_t)(raw.accel.x * ACCEL_FIXED_FACTOR);
                s_imu_packet.imu1_data[1] = (int16_t)(raw.accel.y * ACCEL_FIXED_FACTOR);
                s_imu_packet.imu1_data[2] = (int16_t)(raw.accel.z * ACCEL_FIXED_FACTOR);
                s_imu_packet.imu1_data[3] = (int16_t)(raw.gyro.x  * GYRO_FIXED_FACTOR);
                s_imu_packet.imu1_data[4] = (int16_t)(raw.gyro.y  * GYRO_FIXED_FACTOR);
                s_imu_packet.imu1_data[5] = (int16_t)(raw.gyro.z  * GYRO_FIXED_FACTOR);
                data_collected = true;
            }
        }

        // --- Read IMU 2 (Secondary) ---
        if (s_imu_active_mask & IMU2_ACTIVE) {
            if (imu_read_data(IMU_ADDR_2, &raw) == ESP_OK) {
                s_imu_packet.imu2_data[0] = (int16_t)(raw.accel.x * ACCEL_FIXED_FACTOR);
                s_imu_packet.imu2_data[1] = (int16_t)(raw.accel.y * ACCEL_FIXED_FACTOR);
                s_imu_packet.imu2_data[2] = (int16_t)(raw.accel.z * ACCEL_FIXED_FACTOR);
                s_imu_packet.imu2_data[3] = (int16_t)(raw.gyro.x  * GYRO_FIXED_FACTOR);
                s_imu_packet.imu2_data[4] = (int16_t)(raw.gyro.y  * GYRO_FIXED_FACTOR);
                s_imu_packet.imu2_data[5] = (int16_t)(raw.gyro.z  * GYRO_FIXED_FACTOR);
                data_collected = true;
            }
        }

        // If at least one sensor was read, timestamp and notify the BLE service
        if (data_collected) {
            s_imu_seq++;
            s_imu_packet.timestamp = esp_timer_get_time();
            xEventGroupSetBits(s_imu_event_group, IMU_STREAM_BIT);
        }

        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

/**
 * @brief Initializes I2C hardware and triggers the IMU sampling task.
 */
esp_err_t imu_service_start(EventGroupHandle_t event_group, imu_service_config_t config) {
    if (event_group == NULL) return ESP_ERR_INVALID_ARG;
    
    s_imu_event_group = event_group;
    s_imu_active_mask = 0;

    // Hardware I2C Initialization
    imu_config_t imu_hw_config = IMU_CONFIG_DEFAULT();
    if (imu_init(&imu_hw_config) != ESP_OK) return ESP_FAIL;

    // Check presence and initialize IMU 1
    if (config.enable_imu1 && imu_is_present(IMU_ADDR_1)) {
        s_imu_active_mask |= IMU1_ACTIVE;
        ESP_LOGI(TAG, "IMU 1 (0x6A) detected and initialized.");
    }

    // Check presence and initialize IMU 2
    if (config.enable_imu2 && imu_is_present(IMU_ADDR_2)) {
        imu_configure_sensor(IMU_ADDR_2);
        s_imu_active_mask |= IMU2_ACTIVE;
        ESP_LOGI(TAG, "IMU 2 (0x6B) detected and initialized.");
    }

    if (s_imu_active_mask == 0) {
        ESP_LOGE(TAG, "No IMU sensors found. Task not started.");
        return ESP_ERR_NOT_FOUND;
    }

    // Launch task on Core 0 (Shared with ADC processing)
    return xTaskCreatePinnedToCore(imu_task, "imu_task", 4096, NULL, 5, NULL, 0) == pdPASS ? ESP_OK : ESP_FAIL;
}

size_t imu_service_get_micropacket(void *dest) {
    taskENTER_CRITICAL(&s_imu_mux);
    memcpy(dest, &s_imu_packet, sizeof(imu_micro_packet_t));
    taskEXIT_CRITICAL(&s_imu_mux);
    return sizeof(imu_micro_packet_t);
}