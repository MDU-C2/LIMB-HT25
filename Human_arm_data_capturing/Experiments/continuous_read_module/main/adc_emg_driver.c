/*
 * emg_driver.c
 *
 * Private implementation of the EMG data acquisition driver.
 * This contains all the low-level logic for ADC, DMA, calibration,
 * and sliding window management.
 */
#include "adc_emg_driver.h"
#include <string.h>
#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

// --- Private Driver Configuration ---

#define ADC_SAMPLING_FREQ   4000
#define DMA_TRANSFER_LEN    256

#define ADC_UNIT        ADC_UNIT_1
#define ADC_CONV_MODE   ADC_CONV_SINGLE_UNIT_1
#define ADC_ATTEN       ADC_ATTEN_DB_12
#define ADC_BIT_WIDTH   SOC_ADC_DIGI_MAX_BITWIDTH

#define ADC_OUTPUT_TYPE ADC_DIGI_OUTPUT_FORMAT_TYPE2
#define ADC_GET_CHANNEL(p_data) ((p_data)->type2.channel)
#define ADC_GET_DATA(p_data)    ((p_data)->type2.data)

#define ADC_PIN_CH0 ADC_CHANNEL_2
#define ADC_PIN_CH1 ADC_CHANNEL_3

// --- Private State Variables ---
static const char *TAG = "EMG_DRIVER";
static TaskHandle_t s_task_handle;
static adc_continuous_handle_t s_adc_handle = NULL;
static adc_cali_handle_t s_cali_handle_ch0 = NULL;
#if DUAL_CHANNEL_MODE
static adc_cali_handle_t s_cali_handle_ch1 = NULL;
#endif

static volatile bool s_window_is_ready = false;
static portMUX_TYPE s_packet_spinlock = portMUX_INITIALIZER_UNLOCKED;

#if DUAL_CHANNEL_MODE
    static int16_t s_circular_buffers[2][EMG_WINDOW_SIZE];
    static int s_write_indices[2] = {0, 0};
#else
    static int16_t s_circular_buffers[1][EMG_WINDOW_SIZE];
    static int s_write_indices[1] = {0};
#endif
static int s_new_samples_count = 0;

// --- Private Function Prototypes ---
static bool adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle);
static void continuous_adc_init(adc_channel_t *channel_array, uint8_t channel_num, adc_continuous_handle_t *out_handle);

// --- ISR Callback ---
static bool IRAM_ATTR s_conv_done_cb(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data) {
    BaseType_t mustYield = pdFALSE;
    vTaskNotifyGiveFromISR(s_task_handle, &mustYield);
    return (mustYield == pdTRUE);
}

// --- Public Function Implementations ---

esp_err_t emg_driver_init(void) {
    s_task_handle = xTaskGetCurrentTaskHandle();

#if DUAL_CHANNEL_MODE
    static adc_channel_t channels[2] = {ADC_PIN_CH0, ADC_PIN_CH1};
    continuous_adc_init(channels, 2, &s_adc_handle);
#else
    static adc_channel_t channels[1] = {ADC_PIN_CH0};
    continuous_adc_init(channels, 1, &s_adc_handle);
#endif

    bool cali_ch0_ok = adc_calibration_init(ADC_UNIT, ADC_PIN_CH0, ADC_ATTEN, &s_cali_handle_ch0);
#if DUAL_CHANNEL_MODE
    bool cali_ch1_ok = adc_calibration_init(ADC_UNIT, ADC_PIN_CH1, ADC_ATTEN, &s_cali_handle_ch1);
    if (!cali_ch0_ok || !cali_ch1_ok) return ESP_FAIL;
#else
    if (!cali_ch0_ok) return ESP_FAIL;
#endif

    adc_continuous_evt_cbs_t cbs = { .on_conv_done = s_conv_done_cb };
    return adc_continuous_register_event_callbacks(s_adc_handle, &cbs, NULL);
}

esp_err_t emg_driver_start(void) {
    return adc_continuous_start(s_adc_handle);
}

bool emg_driver_is_window_ready(void) {
    return s_window_is_ready;
}

void emg_driver_get_packet(emg_data_packet_t *packet) {

    taskENTER_CRITICAL(&s_packet_spinlock);
    s_window_is_ready = false; 

    int start_idx_ch0 = (s_write_indices[0] - EMG_WINDOW_SIZE + EMG_WINDOW_SIZE) % EMG_WINDOW_SIZE;

    if (start_idx_ch0 + EMG_WINDOW_SIZE <= EMG_WINDOW_SIZE) {
        memcpy(packet->emg_ch0_window, &s_circular_buffers[0][start_idx_ch0], EMG_WINDOW_SIZE * sizeof(int16_t));
    } else {
        int chunk_size = EMG_WINDOW_SIZE - start_idx_ch0;
        memcpy(packet->emg_ch0_window, &s_circular_buffers[0][start_idx_ch0], chunk_size * sizeof(int16_t));
        memcpy(&packet->emg_ch0_window[chunk_size], s_circular_buffers[0], (EMG_WINDOW_SIZE - chunk_size) * sizeof(int16_t));
    }

#if DUAL_CHANNEL_MODE
    int start_idx_ch1 = (s_write_indices[1] - EMG_WINDOW_SIZE + EMG_WINDOW_SIZE) % EMG_WINDOW_SIZE;
    if (start_idx_ch1 + EMG_WINDOW_SIZE <= EMG_WINDOW_SIZE) {
        memcpy(packet->emg_ch1_window, &s_circular_buffers[1][start_idx_ch1], EMG_WINDOW_SIZE * sizeof(int16_t));
    } else {
        int chunk_size = EMG_WINDOW_SIZE - start_idx_ch1;
        memcpy(packet->emg_ch1_window, &s_circular_buffers[1][start_idx_ch1], chunk_size * sizeof(int16_t));
        memcpy(&packet->emg_ch1_window[chunk_size], s_circular_buffers[1], (EMG_WINDOW_SIZE - chunk_size) * sizeof(int16_t));
    }
#endif

    taskEXIT_CRITICAL(&s_packet_spinlock);
}

void emg_driver_process_data(void) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    uint8_t dma_buffer[DMA_TRANSFER_LEN] = {0};
    uint32_t bytes_read = 0;
    esp_err_t ret = adc_continuous_read(s_adc_handle, dma_buffer, DMA_TRANSFER_LEN, &bytes_read, 0);

    if (ret != ESP_OK) {
        return; // No data or error, wait for next notification
    }

    for (int i = 0; i < bytes_read; i += SOC_ADC_DIGI_RESULT_BYTES) {
        adc_digi_output_data_t *p = (adc_digi_output_data_t*)&dma_buffer[i];
        uint32_t chan_num = ADC_GET_CHANNEL(p);
        uint32_t raw_data = ADC_GET_DATA(p);
        
        int voltage_mv = 0;
        int channel_index = -1;

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
        if (channel_index != -1) {
            s_circular_buffers[channel_index][s_write_indices[channel_index]] = (int16_t)voltage_mv;
            s_write_indices[channel_index] = (s_write_indices[channel_index] + 1) % EMG_WINDOW_SIZE;
        }
        
        if (chan_num == ADC_PIN_CH0) {
            s_new_samples_count++;
        }

        if (s_new_samples_count >= EMG_STEP_SIZE) {
            s_new_samples_count = 0;

            s_window_is_ready = true;
        }
    }
}

// --- Private Helper Function Implementations ---
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
        .conv_mode = ADC_CONV_MODE,
        .format = ADC_OUTPUT_TYPE,
    };

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

