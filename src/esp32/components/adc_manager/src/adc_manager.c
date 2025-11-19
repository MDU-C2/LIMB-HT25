
#include "adc_manager.h"

#include <string.h>
#include "esp_log.h"
#include "esp_adc/adc_oneshot.h"

static const char *TAG = "adc_mgr";

static adc_oneshot_unit_handle_t    s_unit = NULL;
static bool                         s_inited = false;
static bool                         s_channels_used[SOC_ADC_MAX_CHANNELS];

esp_err_t adc_mgr_init(void)
{
    if (s_inited) {
        return ESP_OK;
    }

    // Configure ADC1 in oneshot mode
    adc_oneshot_unit_init_cfg_t unit_cfg = {
        .unit_id = ADC_UNIT_1,
    };

    esp_err_t err = adc_oneshot_new_unit(&unit_cfg, &s_unit);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "adc_oneshot_new_unit failed : %s", esp_err_to_name(err));
        return err;
    }

    memset(s_channels_used, 0, sizeof(s_channels_used));
    s_inited = true;
    ESP_LOGI(TAG, "ADC manager initialized successfully");
    return ESP_OK;
}

esp_err_t adc_mgr_deinit(void)
{
    if (!s_inited) {
        return ESP_OK;
    }

    // Release ADC unit
    esp_err_t err = adc_oneshot_del_unit(s_unit);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "adc_oneshot_del_unit failed : %s", esp_err_to_name(err));
    }
    s_unit = NULL;
    s_inited = false;

    return ESP_OK;
}

adc_mgr_handle_t adc_mgr_register_channel(adc_channel_t channel, const adc_oneshot_chan_cfg_t *cfg)
{
    if (!s_inited) {
        esp_err_t err = adc_mgr_init();
        if (err != ESP_OK) return -1;
    }

    if (channel < 0 || channel >= SOC_ADC_MAX_CHANNELS) {
        return -1;
    }

    adc_oneshot_chan_cfg_t local_cfg;
    if (cfg) {
        local_cfg = *cfg;
    } else {
        local_cfg.bitwidth = ADC_BITWIDTH_DEFAULT;
        local_cfg.atten = ADC_ATTEN_DB_11;
    }

    // Check if this channel is already registered (only one sensor per channel)
    if (s_channels_used[channel]) {
        return channel;  // Use channel number as handle
    }

    // Configure the channel in the underlying driver
    esp_err_t err = adc_oneshot_config_channel(s_unit, channel, &local_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "adc_oneshot_config_channel failed : %s", esp_err_to_name(err));
        return -1;
    }

    s_channels_used[channel] = true;
    ESP_LOGI(TAG, "Registered channel %d", channel);
    return channel;  // Use channel number as handle
}

esp_err_t adc_mgr_read(adc_mgr_handle_t handle, int *out_raw)
{
    if (!s_inited || !out_raw) {
        return ESP_ERR_INVALID_ARG;
    }

    if (handle < 0 || handle >= SOC_ADC_MAX_CHANNELS || !s_channels_used[handle]) {
        return ESP_ERR_INVALID_STATE;
    }

    // Handle is the channel number
    return adc_oneshot_read(s_unit, (adc_channel_t)handle, out_raw);
}

esp_err_t adc_mgr_get_channel(adc_mgr_handle_t handle, adc_channel_t *out_channel)
{
    if (!out_channel) return ESP_ERR_INVALID_ARG;
    if (!s_inited) return ESP_ERR_INVALID_STATE;
    if (handle < 0 || handle >= SOC_ADC_MAX_CHANNELS || !s_channels_used[handle]) return ESP_ERR_INVALID_STATE;

    // Handle is the channel number
    *out_channel = (adc_channel_t)handle;
    return ESP_OK;
}

