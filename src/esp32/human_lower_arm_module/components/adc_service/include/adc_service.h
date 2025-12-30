#ifndef ADC_SERVICE_H
#define ADC_SERVICE_H

#include "esp_err.h"
#include "hal/adc_types.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

// --- Timing & Rate Definitions ---
// Defines the time window for each data burst (10ms = 100Hz packet rate)
#define MICRO_BATCH_MS 10 

// Sampling rates for biopotential and mechanical signals
#define ADC_EMG_SAMPLE_RATE_HZ 4000
#define ADC_EMG_MICRO_SIZE (ADC_EMG_SAMPLE_RATE_HZ * MICRO_BATCH_MS / 1000) // 40 samples/channel

#define ADC_PIEZO_SAMPLE_RATE_HZ 1000
#define ADC_PIEZO_MICRO_SIZE (ADC_PIEZO_SAMPLE_RATE_HZ * MICRO_BATCH_MS / 1000) // 10 samples

#define ADC_SERVICE_CHANNEL_COUNT 3

// --- Event Group Bits for BLE Dispatch ---
// Notify the BLE task that a new micro-packet is ready to be streamed
#define ADC_EMG_STREAM_BIT    (1 << 0)
#define ADC_PIEZO_STREAM_BIT  (1 << 1)

// Initialization masks to verify hardware readiness
#define ADC_EMG1_READY_BIT (1 << 0)
#define ADC_EMG2_READY_BIT (1 << 1)
#define ADC_PIEZO_READY_BIT (1 << 2)

// --- Packed Data Structures (Wire Protocol) ---
// __attribute__((packed)) ensures no compiler padding, matching the Python receiver's struct format.

/**
 * @brief EMG Micro-packet structure
 * Header: 0xAABB for frame synchronization
 */
typedef struct {
    uint16_t header;    // Frame start identifier
    uint32_t seq;       // Sequence counter for packet loss detection
    uint64_t timestamp; // System time in microseconds (esp_timer_get_time)
    uint16_t data[ADC_EMG_MICRO_SIZE * 2]; // Interleaved EMG1 and EMG2 data (80 samples total)
} __attribute__((packed)) emg_micro_packet_t;

/**
 * @brief Piezoelectric Micro-packet structure
 * Header: 0xEEFF for frame synchronization
 */
typedef struct {
    uint16_t header;    
    uint32_t seq;       
    uint64_t timestamp; 
    uint16_t data[ADC_PIEZO_MICRO_SIZE];   // Vibration/Pressure data (10 samples)
} __attribute__((packed)) piezo_micro_packet_t;

// Channel enablement configuration for ADC Manager
typedef struct {
    bool enable_emg1;
    bool enable_emg2;
    bool enable_piezo;
} adc_service_config_t;

// --- Public Interface ---

/**
 * @brief Initializes ADC hardware, internal buffers, and the continuous sampling task.
 * @param event_group Reference to the event group for BLE task synchronization.
 * @param config Structure specifying which channels to activate.
 * @return ESP_OK on success, or appropriate error code.
 */
esp_err_t adc_service_init(EventGroupHandle_t event_group, adc_service_config_t config);

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