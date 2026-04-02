#ifndef ADC_SERVICE_H
#define ADC_SERVICE_H

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "sensors_service.h"

// --- Event Group Bits for BLE Dispatch ---
// Notify the BLE task that a new micro-packet is ready to be streamed
#define ADC_EMG_STREAM_BIT    (1 << 0)
#define ADC_PIEZO_STREAM_BIT  (1 << 1)

// --- Packed Data Structures (Wire Protocol) ---
// __attribute__((packed)) ensures no compiler padding, matching the Python receiver's struct format.

/**
 * @brief EMG Micro-packet structure
 */
typedef struct {
    uint32_t seq;       // Sequence counter for packet loss detection
    uint16_t data[kEmgSamplesToSend * kEmgValuesPerSample];
} __attribute__((packed)) emg_micro_packet_t;

/**
 * @brief Piezoelectric Micro-packet structure
 */
typedef struct {
    uint32_t seq;       
    uint16_t data[kPiezoSamplesToSend * kPiezoValuesPerSample];
} __attribute__((packed)) piezo_micro_packet_t;

// --- Public Interface ---

/**
 * @brief Initializes ADC hardware, internal buffers, and the continuous sampling task.
 * @param event_group Reference to the event group for BLE task synchronization.
 * @param config Structure specifying which channels to activate.
 * @return ESP_OK on success, or appropriate error code.
 */
esp_err_t adc_service_init(EventGroupHandle_t event_group);

/**
 * @brief Copies the latest processed EMG micro-packet to the destination buffer.
 * @param dest Pointer to the destination memory.
 * @return Size of the copied data in bytes.
 */
size_t adc_service_get_emg_micropacket(void *dest);

/**
 * @brief Copies the latest processed Piezo micro-packet to the destination buffer.
 * @param dest Pointer to the destination memory.
 * @return Size of the copied data in bytes.
 */
size_t adc_service_get_piezo_micropacket(void *dest);

#endif
