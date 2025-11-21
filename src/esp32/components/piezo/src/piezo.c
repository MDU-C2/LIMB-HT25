#include "piezo.h"
#include "adc_manager.h"

#include <string.h>
#include "esp_log.h"

static const char *TAG = "PIEZO";

// Static state
static piezo_config_t s_config;
static adc_mgr_handle_t s_adc_handle = -1;
static bool s_initialized = false;

esp_err_t piezo_init(const piezo_config_t *config)
{
    if (s_initialized) {
        ESP_LOGW(TAG, "Piezo already initialized");
        return ESP_ERR_INVALID_STATE;
    }

    // Use default config if none provided
    if (config == NULL) {
        piezo_config_t default_config = PIEZO_CONFIG_DEFAULT();
        s_config = default_config;
        ESP_LOGI(TAG, "Using default configuration");
    } else {
        s_config = *config;
    }

    // Register channel with ADC manager
    adc_oneshot_chan_cfg_t chan_config = {
        .bitwidth = s_config.adc_bitwidth,
        .atten = s_config.adc_atten,
    };
    
    s_adc_handle = adc_mgr_register_channel(s_config.adc_channel, &chan_config);
    if (s_adc_handle < 0) {
        ESP_LOGE(TAG, "Failed to register ADC channel with ADC manager");
        return ESP_FAIL;
    }

    s_initialized = true;
    ESP_LOGI(TAG, "Piezo sensor initialized on GPIO%d (ADC Channel %d)", 
             s_config.gpio_pin, s_config.adc_channel);
    
    return ESP_OK;
}

esp_err_t piezo_read_raw(int *raw_value)
{
    if (!s_initialized || s_adc_handle < 0) {
        ESP_LOGE(TAG, "Piezo sensor not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (raw_value == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    return adc_mgr_read(s_adc_handle, raw_value);
}

esp_err_t piezo_deinit(void)
{
    if (!s_initialized) {
        return ESP_OK;
    }

    // Note: We don't unregister from ADC manager here because other sensors
    // might be using the same channel. The ADC manager handles cleanup.
    s_adc_handle = -1;

    s_initialized = false;
    ESP_LOGI(TAG, "Piezo sensor deinitialized");
    
    return ESP_OK;
}

