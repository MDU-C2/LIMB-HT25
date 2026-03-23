/*
 * emg_service.h
 *
 * Public interface for the EMG Processing Service.
 */
#ifndef EMG_SERVICE_H
#define EMG_SERVICE_H

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"

#include "emg_driver.h"

// --- Public Function Prototypes ---

/**
 * @brief Starts the EMG processing task.
 *
 * This function initializes the EMG driver, creates the mutexes
 * and event groups, and spawns the 'emg_task' to run in the background.
 *
 * @param event_group Handle to the main synchronization event group (if any).
 * @param emg_ready_bit The bit to set when an EMG window is ready.
 * @return ESP_OK on success.
 */
esp_err_t emg_service_start_task(EventGroupHandle_t event_group, EventBits_t emg_ready_bit);

/**
 * @brief Gets the mutex handle for the global EMG data buffer.
 * The consumer task (e.g., sync_send_task) must use this.
 *
 * @return Handle to the EMG buffer mutex.
 */
SemaphoreHandle_t emg_service_get_buffer_mutex(void);

/**
 * @brief Copies the latest complete EMG window from the buffer.
 *
 * @param[out] destination_buffer Pointer to the emg_data_packet_t to fill.
 */
void emg_service_get_packet(emg_data_packet_t *destination_buffer);


#endif // EMG_SERVICE_H