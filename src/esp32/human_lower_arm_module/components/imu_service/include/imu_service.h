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

// Hardware I2C Addresses for the dual IMU setup
#define IMU_ADDR_1   0x6A
#define IMU_ADDR_2   0x6B

/**
 * @brief IMU Micro-packet structure (Streaming Protocol)
 * Contains synchronized samples from both IMUs.
 * Total size: 2 (header) + 4 (seq) + 8 (timestamp) + 12 (IMU1) + 12 (IMU2) = 38 bytes.
 * Values are sent as int16_t (scaled by 1000) for transmission efficiency.
 */
typedef struct {
    uint16_t header;    // Synchronization Magic Number (0xCCDD)
    uint32_t seq;       // Packet sequence counter
    uint64_t timestamp; // System time in microseconds
    int16_t imu1_data[6]; // IMU1: accel_x, y, z | gyro_x, y, z
    int16_t imu2_data[6]; // IMU2: accel_x, y, z | gyro_x, y, z
} __attribute__((packed)) imu_micro_packet_t;

/**
 * @brief Configuration structure for IMU service initialization
 */
typedef struct {
    bool enable_imu1;
    bool enable_imu2;
} imu_service_config_t;

// --- Public API ---

/**
 * @brief Initializes IMU hardware (I2C) and starts the periodic sampling task.
 * @param event_group Reference to the event group for synchronization.
 * @param config Enable/Disable flags for each IMU sensor.
 * @return ESP_OK on success, ESP_FAIL if sensors are not found.
 */
esp_err_t imu_service_start(EventGroupHandle_t event_group, imu_service_config_t config);

/**
 * @brief Thread-safe copy of the latest IMU micro-packet.
 * @param dest Destination buffer to copy the packet into.
 * @return Number of bytes copied.
 */
size_t imu_service_get_micropacket(void *dest);

#endif