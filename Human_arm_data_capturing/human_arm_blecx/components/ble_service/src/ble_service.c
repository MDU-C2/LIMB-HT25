/*
 * ble_service.c
 *
 * Implements the Consumer task: sync_send_task.
 * - Depends on limb_emg_adc (Producer)
 * - Depends on limb_imu (Producer)
 * - Depends on limb_ble_periph (The "Tool" to send data)
 */

#include "ble_service.h" 

// --- Includes needed by sync_send_task ---
#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"

// --- DEPENDENCIES  ---
#include "limb_ble_periph.h"
#include "sensors_service.h"

// 2. The "Producers" 
#include "emg_service.h"
#include "imu_service.h"

// --- Private Definitions ---
static const char *TAG = "BLE_SERVICE"; 

// This counter is now private to this component
static uint32_t g_chunk_seq_num = 0;
#define SEQ_NUM_RESET_VALUE 300000

static EventGroupHandle_t s_sync_event_group;
static EventBits_t s_bits_to_wait_for;

/**
 * @brief This is the local data structure used to hold a
 * copy of the data before sending.
 */
typedef struct {
    emg_data_packet_t emg_data; 
    imu_data_window_t imu_data; 
} combined_packet_t;


// --- Task Prototype ---
static void sync_send_task(void *pvParameters);


// --- Public Function Implementation ---

esp_err_t ble_service_start(EventGroupHandle_t event_group, EventBits_t bits_to_wait) {
    ESP_LOGI(TAG, "Starting BLE Service...");

    // Store the event group handles passed in from main
    s_sync_event_group = event_group;
    s_bits_to_wait_for = bits_to_wait;

    // 1. Start the the BLE peripheral driver
    // This creates the BleTask
    xTaskCreate(BleTask, "ble_task", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "BLE Peripheral Task (BleTask) created.");

    // 2. Start the consumer task
    xTaskCreate(sync_send_task, "sync_task", 4096, NULL, 4, NULL);
    ESP_LOGI(TAG, "Sync/Send Task (sync_send_task) created.");

    return ESP_OK;
}

// =========================================================================
// == TASK 3: CONSUMER (Synchronization & Send Task)
// (This is your logic from main.c, but MODULARIZED)
// =========================================================================
static void sync_send_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Sync Task started, waiting for producers...");

    // --- Local packet buffer ---
    static combined_packet_t local_packet;

    // --- Get handles to the BLE "Tool" buffers ---
    CharacteristicBuffer emg_buffer = get_emg_buf(); 
    // CharacteristicBuffer imu_buffer = get_imu_buf(); 

    // --- Helper constants  ---
    // These must match the Python receiver
    const int emg_samples_per_chunk = kEmgNewSamplesPerWindow / kPartOfWindowPerSend; 
    const int emg_chunk_bytes = emg_samples_per_chunk * kEmgBytesPerSample * kEmgSensorCount;
    const int imu_samples_per_chunk = kImuNewSamplesPerWindow / kPartOfWindowPerSend;
    const int imu_chunk_bytes = imu_samples_per_chunk * kImuBytesPerSample * kImuSensorCount;

    // --- Get handles to the Producer Mutexes ---
    SemaphoreHandle_t emg_mutex = emg_service_get_buffer_mutex();
    // SemaphoreHandle_t imu_mutex = imu_service_get_buffer_mutex();

    while (1) {
        // 1. Wait until both producers signal data is ready
        EventBits_t uxBits = xEventGroupWaitBits(s_sync_event_group, 
                                                s_bits_to_wait_for, 
                                                pdTRUE, // Clear bits on exit
                                                pdTRUE, // Wait for ALL bits
                                                portMAX_DELAY);

        if((uxBits & s_bits_to_wait_for) == s_bits_to_wait_for) {

            // 2. Copy data from component buffers into local_packet
            
            // --- Copy EMG data  ---
            if (xSemaphoreTake(emg_mutex, portMAX_DELAY) == pdTRUE) {
                // We no longer access g_emg_buffer
                // We ASK the service for the packet
                emg_service_get_packet(&local_packet.emg_data);
                xSemaphoreGive(emg_mutex);
            } else {
                ESP_LOGE(TAG, "Failed to take EMG mutex! Skipping packet.");
                continue;
            }

            // // --- Copy IMU data ---
            // if (xSemaphoreTake(imu_mutex, portMAX_DELAY) == pdTRUE) {
            //     // We ASK the service for the packet
            //     imu_service_get_window_packet(&local_packet.imu_data);
            //     xSemaphoreGive(imu_mutex);
            // } else {
            //     ESP_LOGE(TAG, "Failed to take IMU mutex! Skipping packet.");
            //     continue;
            // }

            // 3. --- CHUNKING AND SENDING LOGIC  ---
            
            // ESP_LOGI(TAG, "Sending window: %d chunks", kPartOfWindowPerSend);

            for (int i = 0; i < kPartOfWindowPerSend; i++) { 
                //## -- create temporal vars for stack windows in EMG and IMU so BLE can send same as it does now
                //## -- also size constants could remain the same
                
                // --- Prepare EMG Chunk ---
                uint8_t *emg_chunk_ptr = (uint8_t *) &local_packet.emg_data.emg_ch0_window[i * emg_samples_per_chunk];
                memcpy(emg_buffer.data, emg_chunk_ptr, emg_chunk_bytes);
                memcpy(emg_buffer.data + emg_chunk_bytes, &g_chunk_seq_num, sizeof(uint32_t));

                // // --- Prepare IMU Chunk ---
                // // We use .samples because imu_data is now an imu_data_window_t struct
                // uint8_t *imu_chunk_ptr = (uint8_t *) &local_packet.imu_data.samples[i * imu_samples_per_chunk];
                // memcpy(imu_buffer.data, imu_chunk_ptr, imu_chunk_bytes);
                // memcpy(imu_buffer.data + imu_chunk_bytes, &g_chunk_seq_num, sizeof(uint32_t));

                // --- Send Notifications  ---
                TryNotifyEmgSubscribers();
                // TryNotifyImuSubscribers();
                
                // --- Increment global counter ---
                g_chunk_seq_num++;
            }
            
            if (g_chunk_seq_num >= SEQ_NUM_RESET_VALUE) {
                ESP_LOGW(TAG, "Sequence number counter reset!");
                g_chunk_seq_num = 0;
            }
        }
    } 
} 