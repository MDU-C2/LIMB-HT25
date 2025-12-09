/*
 * emg_service.c
 *
 * Implements the EMG processing task (the "service").
 * This task uses the 'emg_driver' to get data and manages
 * the global data buffer and synchronization primitives.
 */

#include "emg_service.h" 
#include "emg_driver.h"  

#include "esp_log.h"
#include <string.h>

// --- Private Component State Variables ---
static const char *TAG = "EMG_SERVICE";

// Synchronization Primitives (NOW PRIVATE TO THIS COMPONENT)
static EventGroupHandle_t s_sync_event_group; // Handle to the event group
static EventBits_t s_emg_data_ready_bit;      // Bit to set
static SemaphoreHandle_t g_emg_buffer_mutex;  // Mutex for the buffer
static emg_data_packet_t g_emg_buffer;        // Global buffer

// --- Task Prototype ---
static void emg_task(void *pvParameters);

// --- Public Function Implementation ---

SemaphoreHandle_t emg_service_get_buffer_mutex(void) {
    return g_emg_buffer_mutex;
}

void emg_service_get_packet(emg_data_packet_t *destination_buffer) {
    // The consumer (e.g., sync_send_task) will call this function.
    // Assumes mutex is already taken by the caller.
    memcpy(destination_buffer, &g_emg_buffer, sizeof(emg_data_packet_t));
}

esp_err_t emg_service_start_task(EventGroupHandle_t event_group, EventBits_t emg_ready_bit) {
    ESP_LOGI(TAG, "Initializing EMG Module...");

    // Store synchronization primitives
    s_sync_event_group = event_group;
    s_emg_data_ready_bit = emg_ready_bit;

    // Initialize the Hardware Driver
    if (emg_driver_init() != ESP_OK) { 
        ESP_LOGE(TAG, "EMG driver init failed!"); 
        return ESP_FAIL; 
    }

    // Create the Mutex (now owned by this component)
    g_emg_buffer_mutex = xSemaphoreCreateMutex();
    if (!g_emg_buffer_mutex) {
        ESP_LOGE(TAG, "Failed to create EMG mutex!"); 
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "EMG Mutex created.");

    // Create EMG Task
    TaskHandle_t emg_task_handle = NULL;
    xTaskCreate(emg_task, "emg_task", 4096, NULL, 6, &emg_task_handle);
    
    if (!emg_task_handle) { 
        ESP_LOGE(TAG, "Failed to create EMG task!"); 
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "EMG task created.");

    // Connect the Driver to the Task
    emg_driver_set_notify_task(emg_task_handle);
    ESP_LOGI(TAG, "EMG Task Handle configured in driver.");
    
    // Start the Hardware
    if (emg_driver_start() != ESP_OK) { 
        ESP_LOGE(TAG, "Failed to start EMG driver!"); 
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "EMG Driver started.");
    
    return ESP_OK;
}


// =========================================================================
// == EMG TASK
// =========================================================================
static void emg_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Task started, waiting for data...");

    while (1) {
        
        // 1. Sleep and wait for notification from driver ISR
        emg_driver_process_data();

        // 2. Check if the driver raised the flag
        if (emg_driver_is_window_ready()) {
            
            // 3. Take the mutex to update the global buffer
            if (xSemaphoreTake(g_emg_buffer_mutex, portMAX_DELAY) == pdTRUE) {
                
                // 4. Ask the driver to fill the global buffer
                emg_driver_get_packet(&g_emg_buffer);
                
                // 5. Release the mutex
                xSemaphoreGive(g_emg_buffer_mutex);

                // 6. Notify the consumer (e.g., sync_task) that data is ready!
                if (s_sync_event_group) {
                    xEventGroupSetBits(s_sync_event_group, s_emg_data_ready_bit);
                }

                //  --- DEBUG LOG ---
                // ESP_LOGI(TAG, "last value: %d", g_emg_buffer.emg_ch0_window[EMG_WINDOW_SIZE-1]);
                

            } else {
                ESP_LOGE(TAG, "Failed to take EMG mutex!");
            }
        }
    }
}