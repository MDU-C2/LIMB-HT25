
#pragma once

#include "esp_err.h"
#include "esp_adc/adc_oneshot.h"
#include "hal/adc_types.h"

typedef int adc_mgr_handle_t;

// Call once at startup (before any register/read operations)
esp_err_t adc_mgr_init(void);

// Call once at shutdown (after all register/read operations)
esp_err_t adc_mgr_deinit(void);

// Register a new channel (returns handle for subsequent read operations, or -1 on error)
adc_mgr_handle_t adc_mgr_register_channel(adc_channel_t channel, const adc_oneshot_chan_cfg_t *cfg);

// Read the raw value from a registered channel
esp_err_t adc_mgr_read(adc_mgr_handle_t handle, int *out_raw);

// Get the channel associated with a registered handle
esp_err_t adc_mgr_get_channel(adc_mgr_handle_t handle, adc_channel_t *out_channel);