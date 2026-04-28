#ifndef IMU_SERVICE_H
#define IMU_SERVICE_H

#include <stdint.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "sensors_service.h"

// Event bit to notify the BLE Dispatcher that new IMU data is ready
#define IMU_STREAM_BIT          (1 << 2) 

/**
 * @brief IMU Micro-packet structure (Streaming Protocol)
 * Contains synchronized samples from the IMU.
 * Total size: 4 (seq) + 24 (IMU1) = 28 bytes.
 */
typedef struct {
    uint32_t seq;       // Packet sequence counter
    float imu_data[kImuSamplesToSend * kImuValuesPerSample]; // IMU: accel_x, y, z | gyro_pitch, roll, yaw
} __attribute__((packed)) imu_micro_packet_t;

// --- Public API ---

/**
 * @brief Initializes IMU hardware (I2C) and starts the periodic sampling task.
 * @param event_group Reference to the event group for synchronization.
 * @return ESP_OK on success, ESP_FAIL if sensor is not found.
 */
esp_err_t imu_service_start(EventGroupHandle_t event_group);

/**
 * @brief Thread-safe copy of the latest IMU micro-packet.
 * @param dest Destination buffer to copy the packet into.
 * @return Number of bytes copied.
 */
size_t imu_service_get_micropacket(void *dest);

#endif
