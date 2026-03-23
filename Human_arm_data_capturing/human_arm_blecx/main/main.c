/**
 * @file main.c
 * @brief Main application file for 'human_arm' project.
 *
 * This main function only starts the specialized components.
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h" 
#include "esp_log.h"
#include "nvs_flash.h" 

// --- Project Component Includes ---
#include "emg_service.h"
#include "imu_service.h"
#include "ble_service.h" 

// --- Global Variables ---
static const char *TAG_MAIN = "APP_MAIN";

// Synchronization Primitives
static EventGroupHandle_t s_sync_event_group;
const int EMG_DATA_READY_BIT = BIT0;
const int IMU_DATA_READY_BIT = BIT1;


// =========================================================================
// == APP_MAIN: Initialization
// =========================================================================
void app_main(void)
{
    ESP_LOGI(TAG_MAIN, "Starting APP_MAIN...");

    // Initialize Non-Volatile Storage (required by BLE)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize RTOS Synchronization Primitives
    s_sync_event_group = xEventGroupCreate();
    if (!s_sync_event_group) {
        ESP_LOGE(TAG_MAIN, "Failed to create sync event group!"); 
        return;
    }
    ESP_LOGI(TAG_MAIN, "Sync Event Group created.");

    // ===================================================
    // == Start Component Services
    // ===================================================

    // --- Start EMG Service---
    if (emg_service_start_task(s_sync_event_group, EMG_DATA_READY_BIT) != ESP_OK) {
        ESP_LOGE(TAG_MAIN, "Failed to start EMG service!");
    } else {
        ESP_LOGI(TAG_MAIN, "EMG Service started successfully.");
    }
    
    // // --- Start IMU Service ---
    // if (imu_service_start_task(s_sync_event_group, IMU_DATA_READY_BIT) != ESP_OK) {
    //     ESP_LOGE(TAG_MAIN, "Failed to start IMU service!");
    // } else {
    //     ESP_LOGI(TAG_MAIN, "IMU Service started successfully.");
    // }

    // --- Start BLE Service ---
    // const EventBits_t all_bits = (EMG_DATA_READY_BIT | IMU_DATA_READY_BIT);
    const EventBits_t all_bits = (EMG_DATA_READY_BIT);
    if (ble_service_start(s_sync_event_group, all_bits) != ESP_OK) {
        ESP_LOGE(TAG_MAIN, "Failed to start BLE service!");
    } else {
        ESP_LOGI(TAG_MAIN, "BLE Service started successfully.");
    }

    ESP_LOGI(TAG_MAIN, "app_main finished setup. All service tasks are running.");
}