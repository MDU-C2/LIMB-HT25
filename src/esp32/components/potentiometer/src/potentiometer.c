#include "potentiometer.h"

#include <string.h>
#include "esp_log.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

static const char *TAG = "POTENTIOMETER";

// Static state
static potentiometer_config_t s_config;
static adc_oneshot_unit_handle_t s_adc_handle = NULL;
static adc_cali_handle_t s_cali_handle = NULL;
static bool s_initialized = false;

/**
 * @brief Initialize ADC calibration
 */
static bool adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle)
{
    adc_cali_handle_t handle = NULL;
    esp_err_t ret = ESP_FAIL;
    bool calibrated = false;

#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    if (!calibrated) {
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
        ESP_LOGI(TAG, "ADC calibration initialized successfully");
    } else {
        ESP_LOGW(TAG, "ADC calibration not available, using raw values");
    }
    return calibrated;
}

esp_err_t potentiometer_init(const potentiometer_config_t *config)
{
    if (s_initialized) {
        ESP_LOGW(TAG, "Potentiometer already initialized");
        return ESP_ERR_INVALID_STATE;
    }

    // Use default config if none provided
    if (config == NULL) {
        s_config = POTENTIOMETER_CONFIG_DEFAULT();
        ESP_LOGI(TAG, "Using default configuration");
    } else {
        s_config = *config;
    }

    // Initialize ADC unit
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = s_config.adc_unit,
    };
    esp_err_t ret = adc_oneshot_new_unit(&init_config, &s_adc_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize ADC unit: %s", esp_err_to_name(ret));
        return ret;
    }

    // Configure ADC channel
    adc_oneshot_chan_cfg_t chan_config = {
        .bitwidth = s_config.adc_bitwidth,
        .atten = s_config.adc_atten,
    };
    ret = adc_oneshot_config_channel(s_adc_handle, s_config.adc_channel, &chan_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure ADC channel: %s", esp_err_to_name(ret));
        adc_oneshot_del_unit(s_adc_handle);
        s_adc_handle = NULL;
        return ret;
    }

    // Initialize calibration (optional, but recommended)
    adc_calibration_init(s_config.adc_unit, 
                        s_config.adc_channel, 
                        s_config.adc_atten, 
                        &s_cali_handle);

    s_initialized = true;
    ESP_LOGI(TAG, "Potentiometer initialized on GPIO%d (ADC Channel %d)", 
             s_config.gpio_pin, s_config.adc_channel);
    
    return ESP_OK;
}

esp_err_t potentiometer_read_raw(int *raw_value)
{
    if (!s_initialized || s_adc_handle == NULL) {
        ESP_LOGE(TAG, "Potentiometer not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (raw_value == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    return adc_oneshot_read(s_adc_handle, s_config.adc_channel, raw_value);
}

esp_err_t potentiometer_read_voltage(int *voltage_mv)
{
    if (!s_initialized) {
        ESP_LOGE(TAG, "Potentiometer not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (voltage_mv == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    int raw_value;
    esp_err_t ret = potentiometer_read_raw(&raw_value);
    if (ret != ESP_OK) {
        return ret;
    }

    // Convert to voltage if calibration is available
    if (s_cali_handle != NULL) {
        ret = adc_cali_raw_to_voltage(s_cali_handle, raw_value, voltage_mv);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Failed to convert to voltage: %s", esp_err_to_name(ret));
            *voltage_mv = 0;
        }
    } else {
        // No calibration available, return 0
        *voltage_mv = 0;
        ESP_LOGW(TAG, "No calibration available, cannot convert to voltage");
    }

    return ESP_OK;
}

esp_err_t potentiometer_read_normalized(uint16_t *normalized_value)
{
    if (!s_initialized) {
        ESP_LOGE(TAG, "Potentiometer not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (normalized_value == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    int raw_value;
    esp_err_t ret = potentiometer_read_raw(&raw_value);
    if (ret != ESP_OK) {
        return ret;
    }

    // Normalize to 0-1000 range based on ADC bitwidth
    // For 12-bit ADC: max value is 4095
    // For 11-bit ADC: max value is 2047
    // For 10-bit ADC: max value is 1023
    int max_value = 0;
    switch ((int)s_config.adc_bitwidth) {
        case (int)ADC_BITWIDTH_12:
            max_value = 4095;
            break;
        case (int)ADC_BITWIDTH_11:
            max_value = 2047;
            break;
        case (int)ADC_BITWIDTH_10:
            max_value = 1023;
            break;
        case (int)ADC_BITWIDTH_9:
            max_value = 511;
            break;
        default:
            max_value = 4095; // Default to 12-bit
            break;
    }

    if (max_value > 0) {
        *normalized_value = (uint16_t)((raw_value * 1000) / max_value);
        if (*normalized_value > 1000) {
            *normalized_value = 1000;
        }
    } else {
        *normalized_value = 0;
    }

    return ESP_OK;
}

esp_err_t potentiometer_deinit(void)
{
    if (!s_initialized) {
        return ESP_OK;
    }

    // Delete calibration handle
    if (s_cali_handle != NULL) {
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
        adc_cali_delete_scheme_curve_fitting(s_cali_handle);
#endif
        s_cali_handle = NULL;
    }

    // Delete ADC unit
    if (s_adc_handle != NULL) {
        adc_oneshot_del_unit(s_adc_handle);
        s_adc_handle = NULL;
    }

    s_initialized = false;
    ESP_LOGI(TAG, "Potentiometer deinitialized");
    
    return ESP_OK;
}

