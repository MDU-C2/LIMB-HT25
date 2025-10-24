/*
 * adc_emg_driver.c
 *
 * Private implementation of the EMG (ADC) data acquisition driver.
 * (See adc_emg_driver.h for public interface details).
 */
#include "adc_emg_driver.h"
#include <string.h>
#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"         // For task notifications, handles, and critical sections
#include "esp_adc/adc_continuous.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

// --- Private Driver Configuration ---

/** @brief ADC sample rate in Hz (e.g., 4000 Hz). */
#define ADC_SAMPLING_FREQ   4000
/** @brief Size of a single DMA buffer transfer. */
#define DMA_TRANSFER_LEN    256

// --- ADC hardware-specific settings ---
#define ADC_UNIT        ADC_UNIT_1
#define ADC_CONV_MODE   ADC_CONV_SINGLE_UNIT_1
#define ADC_ATTEN       ADC_ATTEN_DB_12
#define ADC_BIT_WIDTH   SOC_ADC_DIGI_MAX_BITWIDTH

// --- ADC data format helpers (for IDF v5.x Type 2 format) ---
#define ADC_OUTPUT_TYPE ADC_DIGI_OUTPUT_FORMAT_TYPE2
#define ADC_GET_CHANNEL(p_data) ((p_data)->type2.channel)
#define ADC_GET_DATA(p_data)    ((p_data)->type2.data)

// --- ADC physical pin mapping ---
#define ADC_PIN_CH0 ADC_CHANNEL_2
#define ADC_PIN_CH1 ADC_CHANNEL_3

// --- Private State Variables ---
static const char *TAG = "EMG_DRIVER";

/** * @brief Task handle for the *processing* task (e.g., emg_task).
 * This handle is captured by emg_driver_process_data() on its first run.
 * It is used by the ISR to notify the correct task when data is ready.
 */
static TaskHandle_t s_task_handle = NULL;

/** @brief Handle for the ADC continuous driver instance. */
static adc_continuous_handle_t s_adc_handle = NULL;
/** @brief Handle for ADC calibration scheme (Channel 0). */
static adc_cali_handle_t s_cali_handle_ch0 = NULL;
#if DUAL_CHANNEL_MODE
/** @brief Handle for ADC calibration scheme (Channel 1). */
static adc_cali_handle_t s_cali_handle_ch1 = NULL;
#endif

/** @brief Volatile flag set by ISR, cleared by application. */
static volatile bool s_window_is_ready = false;
/** @brief Spinlock to protect circular buffer access during packet reconstruction. */
static portMUX_TYPE s_packet_spinlock = portMUX_INITIALIZER_UNLOCKED;

/** @brief The main circular buffer(s) for storing raw millivolt samples. */
#if DUAL_CHANNEL_MODE
    static int16_t s_circular_buffers[2][EMG_WINDOW_SIZE];
    /** @brief Write pointers for each channel's circular buffer. */
    static int s_write_indices[2] = {0, 0};
#else
    static int16_t s_circular_buffers[1][EMG_WINDOW_SIZE];
    static int s_write_indices[1] = {0};
#endif
/** @brief Counter for new samples since the last window was created. Resets every STEP_SIZE. */
static int s_new_samples_count = 0;

// --- Private Function Prototypes ---
static bool adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle);
static void continuous_adc_init(adc_channel_t *channel_array, uint8_t channel_num, adc_continuous_handle_t *out_handle);

// --- ISR Callback ---

/**
 * @brief ADC conversion complete interrupt callback.
 * This function is called from an ISR context when a DMA transfer is done.
 * Its only job is to notify the processing task (s_task_handle) that new data is available.
 */
static bool IRAM_ATTR s_conv_done_cb(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data) {
    BaseType_t mustYield = pdFALSE;
    // Notify the task that is waiting in emg_driver_process_data()
    vTaskNotifyGiveFromISR(s_task_handle, &mustYield);
    // Return true if the notification caused a context switch (i.e., a higher priority task was unblocked)
    return (mustYield == pdTRUE);
}

// --- Public Function Implementations ---

esp_err_t emg_driver_init(void) {
    // Note: The task handle (s_task_handle) is NOT captured here.
    // It will be captured by emg_driver_process_data() on its first call
    // to ensure it gets the handle of the *processing task*, not the *init task*.

#if DUAL_CHANNEL_MODE
    static adc_channel_t channels[2] = {ADC_PIN_CH0, ADC_PIN_CH1};
    continuous_adc_init(channels, 2, &s_adc_handle);
#else
    static adc_channel_t channels[1] = {ADC_PIN_CH0};
    continuous_adc_init(channels, 1, &s_adc_handle);
#endif

    // Initialize calibration for Channel 0
    bool cali_ch0_ok = adc_calibration_init(ADC_UNIT, ADC_PIN_CH0, ADC_ATTEN, &s_cali_handle_ch0);
#if DUAL_CHANNEL_MODE
    // Initialize calibration for Channel 1
    bool cali_ch1_ok = adc_calibration_init(ADC_UNIT, ADC_PIN_CH1, ADC_ATTEN, &s_cali_handle_ch1);
    if (!cali_ch0_ok || !cali_ch1_ok) {
        ESP_LOGE(TAG, "ADC Calibration Failed");
        return ESP_FAIL;
    }
#else
    if (!cali_ch0_ok) {
        ESP_LOGE(TAG, "ADC Ch0 Calibration Failed");
        return ESP_FAIL;
    }
#endif

    // Register the DMA interrupt callback function
    adc_continuous_evt_cbs_t cbs = { .on_conv_done = s_conv_done_cb };
    return adc_continuous_register_event_callbacks(s_adc_handle, &cbs, NULL);
}

esp_err_t emg_driver_start(void) {
    if (s_adc_handle == NULL) return ESP_FAIL;
    // Start the ADC continuous conversion
    return adc_continuous_start(s_adc_handle);
}

bool emg_driver_is_window_ready(void) {
    return s_window_is_ready;
}

void emg_driver_get_packet(emg_data_packet_t *packet) {
    // 1. Enter critical section (protects against other tasks/interrupts)
    taskENTER_CRITICAL(&s_packet_spinlock);
    
    // 2. Clear the flag immediately to avoid race conditions
    s_window_is_ready = false; 

    // 3. Calculate the start index of the *oldest* sample in the window
    // (s_write_indices points to the *next* slot to be written)
    int start_idx_ch0 = (s_write_indices[0] - EMG_WINDOW_SIZE + EMG_WINDOW_SIZE) % EMG_WINDOW_SIZE;

    // 4. Reconstruct the window, handling the circular buffer wrap-around
    if (start_idx_ch0 + EMG_WINDOW_SIZE <= EMG_WINDOW_SIZE) {
        // Case 1: The window is in a single contiguous block (e.g., start_idx is 0)
        memcpy(packet->emg_ch0_window, &s_circular_buffers[0][start_idx_ch0], EMG_WINDOW_SIZE * sizeof(int16_t));
    } else {
        // Case 2: The window is wrapped around the buffer end
        // Copy the first part (from start_idx to the end of the buffer)
        int chunk_size = EMG_WINDOW_SIZE - start_idx_ch0;
        memcpy(packet->emg_ch0_window, &s_circular_buffers[0][start_idx_ch0], chunk_size * sizeof(int16_t));
        
        // Copy the second part (from the beginning of the buffer)
        memcpy(&packet->emg_ch0_window[chunk_size], s_circular_buffers[0], (EMG_WINDOW_SIZE - chunk_size) * sizeof(int16_t));
    }

#if DUAL_CHANNEL_MODE
    // Repeat the exact same reconstruction logic for Channel 1
    int start_idx_ch1 = (s_write_indices[1] - EMG_WINDOW_SIZE + EMG_WINDOW_SIZE) % EMG_WINDOW_SIZE;
    if (start_idx_ch1 + EMG_WINDOW_SIZE <= EMG_WINDOW_SIZE) {
        memcpy(packet->emg_ch1_window, &s_circular_buffers[1][start_idx_ch1], EMG_WINDOW_SIZE * sizeof(int16_t));
    } else {
        int chunk_size = EMG_WINDOW_SIZE - start_idx_ch1;
        memcpy(packet->emg_ch1_window, &s_circular_buffers[1][start_idx_ch1], chunk_size * sizeof(int16_t));
        memcpy(&packet->emg_ch1_window[chunk_size], s_circular_buffers[1], (EMG_WINDOW_SIZE - chunk_size) * sizeof(int16_t));
    }
#endif

    // 5. Exit critical section
    taskEXIT_CRITICAL(&s_packet_spinlock);
}

void emg_driver_process_data(void) {
    // --- Multi-Tasking Safety ---
    // On the first call, capture the handle of the *calling* task.
    // This ensures the ISR notifies the correct task (the 'emg_task')
    // instead of the task that called emg_driver_init() (the 'main' task).

    // Sleep this task until the ISR (s_conv_done_cb) gives a notification
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    // --- Task is now awake, data is ready in DMA buffer ---
    uint8_t dma_buffer[DMA_TRANSFER_LEN] = {0};
    uint32_t bytes_read = 0;
    
    // Read the data from the ADC's internal buffer into our local dma_buffer
    esp_err_t ret = adc_continuous_read(s_adc_handle, dma_buffer, DMA_TRANSFER_LEN, &bytes_read, 0);

    if (ret != ESP_OK) {
        if (ret == ESP_ERR_TIMEOUT) {
            // This can happen, just wait for next notification
            return; 
        }
        ESP_LOGE(TAG, "ADC Read Error: %s", esp_err_to_name(ret));
        return;
    }

    // Process every sample received in the DMA buffer
    for (int i = 0; i < bytes_read; i += SOC_ADC_DIGI_RESULT_BYTES) {
        adc_digi_output_data_t *p = (adc_digi_output_data_t*)&dma_buffer[i];
        uint32_t chan_num = ADC_GET_CHANNEL(p);
        uint32_t raw_data = ADC_GET_DATA(p);
        
        int voltage_mv = 0;
        int channel_index = -1; // Index for our circular buffer array

        // Convert raw ADC data to millivolts using the correct calibration handle
        if (chan_num == ADC_PIN_CH0 && s_cali_handle_ch0) {
            adc_cali_raw_to_voltage(s_cali_handle_ch0, raw_data, &voltage_mv);
            channel_index = 0;
        }
#if DUAL_CHANNEL_MODE
        else if (chan_num == ADC_PIN_CH1 && s_cali_handle_ch1) {
            adc_cali_raw_to_voltage(s_cali_handle_ch1, raw_data, &voltage_mv);
            channel_index = 1;
        }
#endif
        
        // --- Circular Buffer Logic ---
        if (channel_index != -1) {
            // 1. Store the new millivolt sample (as int16_t)
            s_circular_buffers[channel_index][s_write_indices[channel_index]] = (int16_t)voltage_mv;
            // 2. Advance the write index for that channel (with wrap-around)
            s_write_indices[channel_index] = (s_write_indices[channel_index] + 1) % EMG_WINDOW_SIZE;
        }
        
        // 3. Use Channel 0 as the "master clock" to count new samples
        if (chan_num == ADC_PIN_CH0) {
            s_new_samples_count++;
        }

        // 4. Check if we've gathered enough new samples (STEP_SIZE)
        if (s_new_samples_count >= EMG_STEP_SIZE) {
            s_new_samples_count = 0; // Reset for the next step
            
            // Set the flag to notify the main loop that a new window can be reconstructed
            s_window_is_ready = true; 
        }
    }
}

void emg_driver_set_notify_task(TaskHandle_t task_handle) {
    s_task_handle = task_handle;
}

// --- Private Helper Function Implementations ---

/**
 * @brief Configures and initializes the ADC continuous mode handle and DMA.
 */
static void continuous_adc_init(adc_channel_t *channel, uint8_t channel_num, adc_continuous_handle_t *out_handle)
{
    adc_continuous_handle_t handle = NULL;

    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 4096, // Internal DMA buffer size
        .conv_frame_size = DMA_TRANSFER_LEN,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, &handle));

    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = ADC_SAMPLING_FREQ,
        .conv_mode = ADC_CONV_MODE,
        .format = ADC_OUTPUT_TYPE,
    };

    // This pattern configuration array *must* be static.
    // The driver holds a pointer to it, so it cannot be a local variable
    // that gets destroyed when this function exits.
    static adc_digi_pattern_config_t adc_pattern[SOC_ADC_PATT_LEN_MAX] = {0};
    dig_cfg.pattern_num = channel_num;
    for (int i = 0; i < channel_num; i++) {
        adc_pattern[i].atten = ADC_ATTEN;
        adc_pattern[i].channel = channel[i] & 0x7;
        adc_pattern[i].unit = ADC_UNIT;
        adc_pattern[i].bit_width = ADC_BIT_WIDTH;
    }
    dig_cfg.adc_pattern = adc_pattern;
    ESP_ERROR_CHECK(adc_continuous_config(handle, &dig_cfg));
    *out_handle = handle;
}

/**
 * @brief Initializes the ADC calibration scheme for a single channel.
 */
static bool adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle)
{
    adc_cali_handle_t handle = NULL;
    esp_err_t ret = ESP_FAIL;
    bool calibrated = false;
    
// Check if the ESP-IDF version supports the Curve Fitting calibration scheme
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
// Note: Other calibration schemes (like LINE_FITTING) could be added here
// with #if ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED

    *out_handle = handle;
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Calibration Success");
    } else if (ret == ESP_ERR_NOT_SUPPORTED || !calibrated) {
        ESP_LOGW(TAG, "eFuse not burnt, skip software calibration");
    } else if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Invalid arg or no memory for calibration");
    }
    
    return calibrated;
}