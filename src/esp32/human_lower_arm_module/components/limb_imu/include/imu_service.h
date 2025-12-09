/*
 * imu_service.h
 *
 * Public interface for the IMU Processing Service.
 */
#ifndef IMU_SERVICE_H
#define IMU_SERVICE_H

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"
#include "imu_driver.h" 

// --- Public Data Structures ---

/**
 * @brief Defines the data structure for one IMU window.
 */
#define IMU_WINDOW_SIZE 10
typedef struct {
    lsm6dso32_data_t samples[IMU_WINDOW_SIZE]; //## -- add second array samples2 for the second sensor
} imu_data_window_t;


// --- Public Function Prototypes ---

/**
 * @brief Starts the IMU processing task.
 *
 * @param event_group Handle to the main synchronization event group.
 * @param imu_ready_bit The bit to set when an IMU window is ready.
 * @return ESP_OK on success.
 */
esp_err_t imu_service_start_task(EventGroupHandle_t event_group, EventBits_t imu_ready_bit);

/**
 * @brief Gets the mutex handle for the global IMU data buffer.
 *
 * @return Handle to the IMU buffer mutex.
 */
SemaphoreHandle_t imu_service_get_buffer_mutex(void);

/**
 * @brief Copies the latest complete IMU window from the buffer.
 *
 * @param[out] destination_buffer Pointer to the imu_data_window_t to fill.
 */
void imu_service_get_window_packet(imu_data_window_t *destination_buffer);


#endif // IMU_SERVICE_H