/*
 * ble_service.h
 *
 * Public interface for the BLE Tx Service.
 */
#ifndef BLE_SERVICE_H
#define BLE_SERVICE_H

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

/**
 * @brief Starts the BLE service and the synchronization task.
 *
 * This function initializes the BLE peripheral (limb_ble_periph)
 * and spawns the 'sync_send_task' to run in the background.
 *
 * @param event_group Handle to the main synchronization event group.
 * @param bits_to_wait The bits (EMG + IMU) to wait for.
 * @return ESP_OK on success.
 */
esp_err_t ble_service_start(EventGroupHandle_t event_group, EventBits_t bits_to_wait);


#endif // BLE_SERVICE_H