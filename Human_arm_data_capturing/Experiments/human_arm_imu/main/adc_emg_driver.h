/*
 * adc_emg_driver.h
 *
 * Public interface for the EMG (ADC) data acquisition driver.
 *
 * This file defines the public functions and data structures that
 * the main application (or other tasks) can use to initialize, 
 * start, and get data from the ADC system.
 *
 * This driver is responsible for configuring the ESP32's continuous ADC,
 * sampling data via DMA, managing an internal circular buffer, and 
 * reconstructing sliding data windows.
 */
#ifndef EMG_DRIVER_H
#define EMG_DRIVER_H

#include "esp_system.h" 
#include "freertos/FreeRTOS.h" 
#include "freertos/task.h"     

/**
 * @brief Set to 1 to enable two-channel sampling, 0 for single-channel.
 * This flag controls the data structures and driver configuration.
 */
#define DUAL_CHANNEL_MODE   0

// --- Public Configuration ---

/**
 * @brief The total number of samples in one complete data window.
 * (e.g., 800 samples)
 */
#define EMG_WINDOW_SIZE 800

/**
 * @brief The number of *new* samples to acquire before creating a new window.
 * This defines the "overlap".
 * (e.g., 400 new samples @ 800 window size = 50% overlap).
 * This value determines the rate at which new packets are generated.
 */
#define EMG_STEP_SIZE   400

// --- Public Data Structures ---

/**
 * @brief Structure to hold one complete window of EMG data.
 *
 * This is the "product" that the driver delivers.
 * The data is stored as raw millivolts (int16_t) to save memory
 * and optimize for faster data transfer (e.g., via Bluetooth).
 */
typedef struct {
    /** @brief Array holding the window data for Channel 0 (in millivolts). */
    int16_t emg_ch0_window[EMG_WINDOW_SIZE];
#if DUAL_CHANNEL_MODE
    /** @brief Array holding the window data for Channel 1 (in millivolts). */
    int16_t emg_ch1_window[EMG_WINDOW_SIZE];
#endif
} emg_data_packet_t;


// --- Public Function Prototypes ---
// These are the "buttons" the main application can press to control the driver.

/**
 * @brief Initializes the continuous ADC, calibration, and all necessary buffers.
 * This must be called once before starting the acquisition.
 * @return ESP_OK on success, or an error code on failure.
 */
esp_err_t emg_driver_init(void);

/**
 * @brief Starts the ADC continuous conversion.
 * After this is called, the hardware will start capturing data 
 * via DMA in the background.
 * @return ESP_OK on success.
 */
esp_err_t emg_driver_start(void);

/**
 * @brief Checks if a new, complete data window is ready to be collected.
 *
 * This is a non-blocking function. The main application (or task) 
 * should call this repeatedly in its loop (typically after 
 * emg_driver_process_data() returns) to check for new data.
 *
 * @return true if a new window is available, false otherwise.
 */
bool emg_driver_is_window_ready(void);

/**
 * @brief Copies the latest complete data window into the provided packet structure.
 *
 * This function performs the reconstruction of the window from the internal
 * circular buffer. It is a critical section and is protected by a spinlock.
 * After calling this function, the "window ready" flag is automatically reset.
 *
 * @param[out] packet A pointer to a data packet structure that will be filled
 * with the latest window data.
 */
void emg_driver_get_packet(emg_data_packet_t *packet);

/**
 * @brief The core processing "engine" of the driver.
 *
 * This function must be called continuously in the application's main loop (or task).
 * It blocks (sleeps) the calling task until a new batch of data is
 * available from the ADC's DMA interrupt.
 * * When awakened, it processes the new samples, fills the internal
 * circular buffer, and sets the "window ready" flag if EMG_STEP_SIZE
 * new samples have been acquired.
 */
void emg_driver_process_data(void);

/**
 * @brief Sets the task handle to be notified by the ADC ISR.
 * MUST be called after the processing task is created and before emg_driver_start().
 * @param task_handle The handle of the task waiting in emg_driver_process_data().
 */
void emg_driver_set_notify_task(TaskHandle_t task_handle);

#endif // EMG_DRIVER_H