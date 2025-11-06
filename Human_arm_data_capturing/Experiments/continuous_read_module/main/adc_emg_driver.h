/*
 * adc_emg_driver.h
 *
 * Public interface for the EMG data acquisition driver.
 * This file defines the functions and data structures that the main application
 * can use to initialize, start, and get data from the ADC system.
 */
#ifndef EMG_DRIVER_H
#define EMG_DRIVER_H

#include "esp_system.h" // Required for the 'esp_err_t' type

#define DUAL_CHANNEL_MODE   0
// --- Public Configuration ---
// These parameters define the structure of the data this driver produces.
#define EMG_WINDOW_SIZE 800
#define EMG_STEP_SIZE   400

// --- Public Data Structures ---
// This is the "product" that the driver delivers: a complete data packet
// ready for transmission or further processing.
typedef struct {
    int16_t emg_ch0_window[EMG_WINDOW_SIZE];
#if DUAL_CHANNEL_MODE
    int16_t emg_ch1_window[EMG_WINDOW_SIZE];
#endif
} emg_data_packet_t;


// --- Public Function Prototypes ---
// These are the "buttons" the main application can press to control the driver.

/**
 * @brief Initializes the continuous ADC, calibration, and all necessary buffers.
 * This must be called once before starting the acquisition.
 * @return ESP_OK on success.
 */
esp_err_t emg_driver_init(void);

/**
 * @brief Starts the ADC continuous conversion.
 * After this is called, the hardware will start capturing data in the background.
 * @return ESP_OK on success.
 */
esp_err_t emg_driver_start(void);

/**
 * @brief Checks if a new, complete data window is ready.
 * This is a non-blocking function. The main application should call this
 * repeatedly in its loop to check for new data.
 * @return true if a new window is available, false otherwise.
 */
bool emg_driver_is_window_ready(void);

/**
 * @brief Copies the latest complete data window into the provided packet structure.
 *
 * After calling this function, the "window ready" flag is reset.
 * @param[out] packet A pointer to a data packet structure that will be filled
 * with the latest window data.
 */
void emg_driver_get_packet(emg_data_packet_t *packet);

/**
 * @brief The core processing function for the driver.
 * This function must be called continuously in the main application loop.
 * It waits for new data from the ADC/DMA and processes it to build the sliding window.
 */
void emg_driver_process_data(void);

#endif // EMG_DRIVER_H