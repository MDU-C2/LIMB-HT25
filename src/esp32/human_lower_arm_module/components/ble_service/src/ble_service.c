#include "ble_service.h"
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_timer.h"

#include "limb_ble_periph.h"
#include "sensors_service.h"
#include "adc_service.h"
#include "imu_service.h"

static const char *TAG = "BLE_SERVICE_STREAM";

// --- Local Temporary Buffers ---
// Used to fetch data from sensor services before copying to BLE GATT buffers
static emg_micro_packet_t   s_temp_emg;
static piezo_micro_packet_t s_temp_piezo;
static imu_micro_packet_t   s_temp_imu;
static_assert(sizeof(s_temp_imu) == kImuBufSize, "We expect that the IMU micro packet and the BLE IMU characteristic have the same size.");

/**
 * @brief Main BLE synchronization and dispatch task.
 * This task waits for event bits from ADC and IMU services.
 * It ensures that data is sent as soon as it is ready, maintaining high throughput.
 */
static void ble_sync_send_task(void *pvParameters) {
    ble_task_params_t *params = (ble_task_params_t *)pvParameters;
    EventGroupHandle_t sync_group = params->group;
    
    // Bits to monitor: EMG, Piezo, and IMU ready signals
    const EventBits_t STREAM_BITS = (ADC_EMG_STREAM_BIT | ADC_PIEZO_STREAM_BIT | IMU_STREAM_BIT);

    ESP_LOGI(TAG, "BLE Streaming Task: Dispatcher started.");

    // System Health Monitoring variables
    uint32_t emg_count = 0, piezo_count = 0, imu_count = 0;
    uint64_t last_log_time = esp_timer_get_time();

    while (1) {
        // Wait for any of the sensor services to set their "Ready" bit
        EventBits_t bits = xEventGroupWaitBits(
            sync_group, 
            STREAM_BITS,
            pdTRUE,        // Clear bits on exit
            pdFALSE,       // Wait for ANY bit
            pdMS_TO_TICKS(1000)
        );

        uint64_t now = esp_timer_get_time();

        // --- EMG Dispatch ---
        if (bits & ADC_EMG_STREAM_BIT) {
            size_t sz = adc_service_get_emg_micropacket(&s_temp_emg);
            CharacteristicBuffer master = get_emg_buf(); // Get pointer to GATT characteristic buffer
            
            if (sz <= master.size) {
                memcpy(master.data, &s_temp_emg, sz);
                TryNotifyEmgSubscribers(); // Trigger BLE Notification
                emg_count++;
            }
        }

        // --- Piezo Dispatch ---
        if (bits & ADC_PIEZO_STREAM_BIT) {
            size_t sz = adc_service_get_piezo_micropacket(&s_temp_piezo);
            CharacteristicBuffer master = get_piezo_buf();
            
            if (sz <= master.size) {
                memcpy(master.data, &s_temp_piezo, sz);
                TryNotifyPiezoSubscribers();
                piezo_count++;
            }
        }

        // --- IMU Dispatch ---
        if (bits & IMU_STREAM_BIT) {
            size_t sz = imu_service_get_micropacket(&s_temp_imu);
            CharacteristicBuffer master = get_imu_buf();
            
            if (sz <= master.size) {
                memcpy(master.data, &s_temp_imu, sz);
                TryNotifyImuSubscribers();
                imu_count++;
            }
        }

        // --- System Health & Diagnostic Report (Every 2 seconds) ---
        if (now - last_log_time >= 5000000) {
            ESP_LOGI("DIAG", "*******************");
            // Values should be approx 200 (100 PPS * 2 seconds)
            ESP_LOGI("DIAG", "Health [pps] -> EMG: %lu | IMU: %lu | PIEZO: %lu", 
                     emg_count, imu_count, piezo_count);
            ESP_LOGI("DIAG", "---------------------");
            // Log last samples to verify signal integrity
            ESP_LOGI("DIAG", "Last Data -> EMG1:%u | PIEZO:%u",
                     s_temp_emg.data[39], s_temp_piezo.data[9]);

            ESP_LOGI("DIAG", "IMU -> AccZ:%f | GyroZ:%f", s_temp_imu.imu_data[2], s_temp_imu.imu_data[5]);
            ESP_LOGI("DIAG", "*******************");
            
            // Reset counters for the next window
            emg_count = 0;
            piezo_count = 0;
            imu_count = 0;
            last_log_time = now;
        }
    }
}

/**
 * @brief Initializes and starts the BLE services and synchronization tasks.
 */
esp_err_t ble_service_start(EventGroupHandle_t event_group, EventBits_t bits_to_wait) {
    static ble_task_params_t params;
    params.group = event_group;
    params.mask = bits_to_wait;

    // Start NimBLE/Stack core task
    xTaskCreatePinnedToCore(BleTask, "ble_stack", 8192, NULL, 10, NULL, 0);

    // Start our custom Synchronization Dispatcher task
    // Assigned to Core 0 (or Core 1 depending on workload) with high priority
    BaseType_t ret = xTaskCreatePinnedToCore(
        ble_sync_send_task,
        "ble_sync_task",
        4096,
        &params,
        15, // High priority to avoid latency in radio dispatch
        NULL,
        0   // Core 0
    );

    return (ret == pdPASS) ? ESP_OK : ESP_FAIL;
}
