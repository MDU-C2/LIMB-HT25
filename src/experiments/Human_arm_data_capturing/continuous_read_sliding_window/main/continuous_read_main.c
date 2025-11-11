/*
 * SPDX-FileCopyrightText: 2021-2022 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string.h>
#include <stdio.h>
#include "sdkconfig.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_adc/adc_continuous.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

// =========================================================================
// == 1. SYSTEM CONFIGURATION: ADC AND SLIDING WINDOW PARAMETERS
// =========================================================================
// This section defines the core parameters of our data acquisition system.

// --- Sliding Window Parameters ---
#define WINDOW_SIZE         800  // The total number of samples in each data window sent for processing.
#define STEP_SIZE           400  // The number of NEW samples to acquire before creating the next window.
// This results in a 50% overlap (400 samples) between consecutive windows.

// --- ADC & DMA Parameters ---
#define ADC_SAMPLING_FREQ   4000 // The speed at which the ADC will capture samples, in Hertz (Hz).
#define DMA_TRANSFER_LEN    256  // The size of each data block transferred from the ADC to memory, in bytes.
                                 // Since each 12-bit sample requires 2 bytes, this block holds 128 samples.
                                 // A smaller size reduces latency but increases CPU overhead slightly.

// --- Generic ADC Driver Settings ---
#define EXAMPLE_ADC_UNIT        ADC_UNIT_1
#define EXAMPLE_ADC_CONV_MODE   ADC_CONV_SINGLE_UNIT_1
#define EXAMPLE_ADC_ATTEN       ADC_ATTEN_DB_12         // Use the widest voltage measurement range (~0-3.0V).
#define EXAMPLE_ADC_BIT_WIDTH   SOC_ADC_DIGI_MAX_BITWIDTH // Use the highest hardware-supported resolution (12-bit).

// --- Data Format Helpers (for ESP32-C3) ---
#define EXAMPLE_ADC_OUTPUT_TYPE ADC_DIGI_OUTPUT_FORMAT_TYPE2
#define EXAMPLE_ADC_GET_CHANNEL(p_data) ((p_data)->type2.channel)
#define EXAMPLE_ADC_GET_DATA(p_data)    ((p_data)->type2.data)

// --- Static Global Variables ---
static adc_channel_t channel[1] = {ADC_CHANNEL_2}; // The ADC channel (GPIO pin) to read from.
static TaskHandle_t s_task_handle;                 // A handle to the main task, used for notifications from the ISR.
static const char *TAG = "EMG_ACQUISITION";        // Tag for logging messages.

// =========================================================================
// == 2. DATA BUFFERS AND STATE VARIABLES
// =========================================================================
// These variables manage the state of our data acquisition pipeline. They are
// declared as 'static' to limit their scope to this file.

// The circular buffer acts as the system's short-term memory. It continuously
// stores the most recent `WINDOW_SIZE` voltage samples.
static float s_emg_circular_buffer[WINDOW_SIZE];

// The linear buffer is a temporary container. It holds a clean, ordered
// "snapshot" of the circular buffer, ready to be sent over Bluetooth.
static float s_window_to_send[WINDOW_SIZE];

// The write index tracks the next position to write to in the circular buffer.
// It wraps around to the beginning when it reaches the end.
static int s_write_index = 0;

// This counter tracks how many new samples have arrived since the last window
// was created. It acts as the trigger for our `STEP_SIZE`.
static int s_new_samples_count = 0;

// --- Function Prototypes ---
// Declaring functions before they are used allows us to keep `app_main` at the
// top for better readability. The actual function bodies are defined at the end.
static bool adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle);
static void adc_calibration_deinit(adc_cali_handle_t handle);
static void continuous_adc_init(adc_channel_t *channel, uint8_t channel_num, adc_continuous_handle_t *out_handle);

// The ADC Interrupt Service Routine (ISR) callback.
// every time the DMA has filled one transfer block (`DMA_TRANSFER_LEN`).
// NOTE: ISRs must be extremely fast. Do not perform complex calculations or blocking operations here.
static bool IRAM_ATTR s_conv_done_cb(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data)
{
    BaseType_t mustYield = pdFALSE;
    // Notify the main task that new data is available. This is a very fast, non-blocking operation.
    vTaskNotifyGiveFromISR(s_task_handle, &mustYield);
    return (mustYield == pdTRUE);
}

// =========================================================================
// == 3. MAIN APPLICATION LOGIC (app_main)
// =========================================================================

void app_main(void)
{
    esp_err_t ret;
    uint32_t ret_num = 0;
    // This buffer will receive the small blocks of raw data from the DMA.
    uint8_t dma_result_buffer[DMA_TRANSFER_LEN] = {0};

    // Get a handle to the current task so the ISR knows who to notify.
    s_task_handle = xTaskGetCurrentTaskHandle();

    // --- Initialization Phase ---
    // Setup the ADC for continuous reading.
    adc_continuous_handle_t handle = NULL;
    continuous_adc_init(channel, sizeof(channel) / sizeof(adc_channel_t), &handle);

    // Setup the calibration scheme to convert raw ADC values to voltage.
    adc_cali_handle_t cali_handle = NULL;
    bool do_calibration = adc_calibration_init(EXAMPLE_ADC_UNIT, channel[0], EXAMPLE_ADC_ATTEN, &cali_handle);

    // --- Start the Acquisition Engine ---
    // Register the callback function that will be triggered when data is ready.
    adc_continuous_evt_cbs_t cbs = { .on_conv_done = s_conv_done_cb };
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(handle, &cbs, NULL));
    
    // Start the ADC. From this point on, the hardware will be capturing data in the background.
    ESP_ERROR_CHECK(adc_continuous_start(handle));

    // --- Main Processing Loop ---
    while (1) {
        // Wait here indefinitely until notified by the ISR. This consumes zero CPU while waiting.
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // This inner loop ensures we process all data that may have accumulated
        // in the DMA buffer while the main task was busy.
        while (1) {
            // Try to read a block of data from the DMA's internal buffer.
            ret = adc_continuous_read(handle, dma_result_buffer, DMA_TRANSFER_LEN, &ret_num, 0);

            if (ret == ESP_OK) {
                // If data was successfully read, process each sample in the block.
                for (int i = 0; i < ret_num; i += SOC_ADC_DIGI_RESULT_BYTES) {
                    // Unpack the raw 12-bit data from its 2-byte container.
                    adc_digi_output_data_t *p = (adc_digi_output_data_t*)&dma_result_buffer[i];
                    uint32_t raw_data = EXAMPLE_ADC_GET_DATA(p);

                    // --- Circular Buffer Logic ---
                    if (do_calibration) {
                        int voltage_mv = 0;
                        adc_cali_raw_to_voltage(cali_handle, raw_data, &voltage_mv);
                        
                        // 1. Store the new voltage sample in the circular buffer.
                        s_emg_circular_buffer[s_write_index] = (float)voltage_mv / 1000.0;
                    } else {
                        // If calibration is not available, store a default value.
                        s_emg_circular_buffer[s_write_index] = 0.0;
                    }

                    // 2. Advance the write index, wrapping around if it reaches the end.
                    // The modulo operator (%) is key to making the buffer circular.
                    s_write_index = (s_write_index + 1) % WINDOW_SIZE;
                    s_new_samples_count++;

                    // 3. Check if we have gathered enough new samples to create a new window.
                    if (s_new_samples_count >= STEP_SIZE) {
                        s_new_samples_count = 0; // Reset the counter for the next step.

                        // --- Window Reconstruction ---
                        // This logic creates a flat, linear copy of the data from the circular buffer.
                        // It correctly handles the "wrap-around" case where the window is split in two parts.
                        int start_index = (s_write_index - WINDOW_SIZE + WINDOW_SIZE) % WINDOW_SIZE;
                        
                        if (start_index + WINDOW_SIZE <= WINDOW_SIZE) {
                            // Simple case: The window is in one contiguous block.
                            memcpy(s_window_to_send, &s_emg_circular_buffer[start_index], WINDOW_SIZE * sizeof(float));
                        } else {
                            // Complex case: The window wraps around. It must be copied in two chunks.
                            int first_chunk_size = WINDOW_SIZE - start_index;
                            memcpy(s_window_to_send, &s_emg_circular_buffer[start_index], first_chunk_size * sizeof(float));
                            memcpy(&s_window_to_send[first_chunk_size], s_emg_circular_buffer, (WINDOW_SIZE - first_chunk_size) * sizeof(float));
                        }

                        // --- Debugging Output ---
                        ESP_LOGI(TAG, "Window created. Start index: %d. Size: %d. Step: %d.", 
                                 start_index, WINDOW_SIZE, STEP_SIZE);

                        // =========================================================================
                        // == 4. PLACEHOLDER FOR BLUETOOTH DATA TRANSMISSION
                        // =========================================================================
                        // At this point, `s_window_to_send` contains the complete 800-sample
                        // voltage window, perfectly ordered and ready for transmission.
                        //
                        // You would call your Bluetooth sending function here, e.g.:
                        // ble_send_emg_data(s_window_to_send, sizeof(s_window_to_send));
                        // =========================================================================
                    }
                }
            } else if (ret == ESP_ERR_TIMEOUT) {
                // If adc_continuous_read returns a timeout, it means the DMA buffer is empty.
                // We can break from this inner loop and go back to sleep, waiting for the next notification.
                break;
            }
        }
    }

    // --- Teardown Phase (Unreachable Code) ---
    // In this specific application, the main loop runs forever. However, it's good practice
    // to include cleanup code for more complex applications where tasks might be stopped and started.
    ESP_ERROR_CHECK(adc_continuous_stop(handle));
    if (do_calibration) {
        adc_calibration_deinit(cali_handle);
    }
}

// =========================================================================
// == 5. HELPER FUNCTION DEFINITIONS
// =========================================================================

// This function configures and initializes the ADC for continuous (DMA-based) reading.
static void continuous_adc_init(adc_channel_t *channel, uint8_t channel_num, adc_continuous_handle_t *out_handle)
{
    adc_continuous_handle_t handle = NULL;

    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 4096, // Increased to handle high-frequency data without overflow.
        .conv_frame_size = DMA_TRANSFER_LEN,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, &handle));

    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = ADC_SAMPLING_FREQ,
        .conv_mode = EXAMPLE_ADC_CONV_MODE,
        .format = EXAMPLE_ADC_OUTPUT_TYPE,
    };

    adc_digi_pattern_config_t adc_pattern[SOC_ADC_PATT_LEN_MAX] = {0};
    dig_cfg.pattern_num = channel_num;
    for (int i = 0; i < channel_num; i++) {
        adc_pattern[i].atten = EXAMPLE_ADC_ATTEN;
        adc_pattern[i].channel = channel[i] & 0x7;
        adc_pattern[i].unit = EXAMPLE_ADC_UNIT;
        adc_pattern[i].bit_width = EXAMPLE_ADC_BIT_WIDTH;
    }
    dig_cfg.adc_pattern = adc_pattern;
    ESP_ERROR_CHECK(adc_continuous_config(handle, &dig_cfg));
    *out_handle = handle;
}

// This function sets up the ADC calibration based on values stored in the chip's eFuse.
static bool adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle)
{
    adc_cali_handle_t handle = NULL;
    esp_err_t ret = ESP_FAIL;
    bool calibrated = false;
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    if (!calibrated) {
        ESP_LOGI(TAG, "Calibration scheme version is %s", "Curve Fitting");
        adc_cali_curve_fitting_config_t cali_config = {
            .unit_id = unit,
            .chan = channel,
            .atten = atten,
            .bitwidth = ADC_BITWIDTH_DEFAULT,
        };
        ret = adc_cali_create_scheme_curve_fitting(&cali_config, &handle);
        if (ret == ESP_OK) {
            calibrated = true;
        }
    }
#endif
    *out_handle = handle;
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Calibration Success");
    } else if (ret == ESP_ERR_NOT_SUPPORTED || !calibrated) {
        ESP_LOGW(TAG, "eFuse not burnt, skip software calibration");
    } else {
        ESP_LOGE(TAG, "Invalid arg or no memory");
    }
    return calibrated;
}

// This function releases the memory used by the calibration scheme.
static void adc_calibration_deinit(adc_cali_handle_t handle)
{
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    ESP_LOGI(TAG, "deregister %s calibration scheme", "Curve Fitting");
    ESP_ERROR_CHECK(adc_cali_delete_scheme_curve_fitting(handle));
#endif
}