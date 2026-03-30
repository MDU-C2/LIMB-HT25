#ifndef IMU_SERVICE_H
#define IMU_SERVICE_H

#include <stdint.h>
#include "esp_err.h"
#include "imu.h" 
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

// --- Streaming Configuration ---
// The IMU task runs at 100Hz, providing a fresh sample every 10ms
#define IMU_SAMPLE_RATE_HZ      100   
#define IMU_HEADER_MAGIC        0xCCDD

// Event bit to notify the BLE Dispatcher that new IMU data is ready
#define IMU_STREAM_BIT          (1 << 2) 

/**
 * @brief IMU Micro-packet structure (Streaming Protocol)
 * Contains synchronized samples from the IMU.
 * Total size: 2 (header) + 4 (seq) + 8 (timestamp) + 12 (IMU1) = 26 bytes.
 * Values are sent as int16_t (scaled by 1000) for transmission efficiency.
 */
typedef struct {
    uint16_t header;    // Synchronization Magic Number (0xCCDD)
    uint32_t seq;       // Packet sequence counter
    uint64_t timestamp; // System time in microseconds
    int16_t imu_data[6]; // IMU: accel_x, y, z | gyro_x, y, z
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