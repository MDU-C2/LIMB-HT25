
#pragma once

#include "esp_err.h"
#include "hal/adc_types.h"

// The ADC configuration for a single channel.
typedef struct {
  adc_channel_t channel;
  uint32_t sample_rate;
} AdcMgrChannelConfig;

// ADC configurations for multiple channels.
typedef struct {
  AdcMgrChannelConfig *channel_configs;
  uint8_t channel_configs_len;
  uint16_t ms_worth_of_buffer_size;
} AdcMgrConfig;

// A simple buffer struct for 16-bit values.
typedef struct {
  uint16_t *data;
  uint16_t capacity;
  uint16_t length;
} AdcMgrChannelBuffer;

// A struct containing value buffers for all available channels. The index of
// each element corresponds to the channel's ID (0th index is the buffer for
// channel 0, 1st for channel 1, etc.).
typedef struct {
  AdcMgrChannelBuffer channel_buffers[SOC_ADC_MAX_CHANNEL_NUM];
} AdcMgrReadResults;

// Call once at startup (before any register/read operations)
esp_err_t adc_mgr_init(AdcMgrConfig config);

// Call once at shutdown (after all register/read operations)
esp_err_t adc_mgr_deinit(void);

// Read raw values from registered channels.
// The caller should provide an AdcMgrChannelBuffer for all channels registered
// in adc_mgr_init.
esp_err_t adc_mgr_read(AdcMgrReadResults *inout_results, uint32_t timeout_ms);
