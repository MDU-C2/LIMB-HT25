#ifndef BLE_SERVICE_H
#define BLE_SERVICE_H

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

/**
 * @brief Parameters for the BLE Synchronization Task.
 * Contains the event group and bitmask required for multi-sensor coordination.
 */
typedef struct {
    EventGroupHandle_t group; // Shared event group for task notification
    EventBits_t mask;         // Mask defining which sensors are active for streaming
} ble_task_params_t;

/**
 * @brief Starts the BLE Dispatcher Task.
 * * This task waits for synchronization bits from the ADC and IMU services.
 * When data is ready, it fetches the micro-packets and triggers BLE notifications.
 * * @param event_group The RTOS event group where ADC and IMU services report ready status.
 * @param bits_to_wait Bitmask of the streams to monitor (e.g., EMG_STREAM_BIT | IMU_STREAM_BIT).
 * @return ESP_OK if the task was successfully created, ESP_FAIL otherwise.
 */
esp_err_t ble_service_start(EventGroupHandle_t event_group, EventBits_t bits_to_wait);

#endif // BLE_SERVICE_H