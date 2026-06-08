
#include "adc_manager.h"

#include <string.h>
#include <sys/param.h>

#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_continuous.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal/adc_types.h"
#include "soc/soc_caps.h"

// --- Private function declarations ---
static bool adc_mgr_channel_buffer_push(AdcMgrChannelBuffer* buf,
                                        uint16_t value);
static bool write_results_to_channel_buffers(
    const adc_digi_output_data_t* outputs, uint32_t output_count,
    AdcMgrReadResults* inout_results);
static esp_err_t adc_mgr_init_handle_preconditions(AdcMgrConfig config);
static void adc_mgr_reset_global_values(void);

static_assert(ADC_CHANNEL_0 == 0,
              "We assume that the channel enum values correspond to their IDs "
              "when doing array indexing.");

// Only one instance of the ADC manager can be running at the same time, so
// globals it is.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static adc_continuous_handle_t s_handle = NULL;
static bool s_inited = false;
static uint8_t s_channels_count = 0;
static uint32_t s_sample_rate = 0;
static uint32_t s_desired_sample_rates_per_channel[SOC_ADC_MAX_CHANNEL_NUM];
// The period is the ratio between the channel's desired sample rate and the
// ADC's actual sample rate. Every first value of a channel's period is then the
// value that should be used to achieve its desired sample rate.
static uint32_t s_values_in_channel_periods[SOC_ADC_MAX_CHANNEL_NUM] = {0};
static uint32_t s_values_read_already_in_period[SOC_ADC_MAX_CHANNEL_NUM];

static adc_cali_handle_t s_calibration_handle;

enum {
  kAdcReadBufLen = SOC_ADC_DIGI_DATA_BYTES_PER_CONV * 5 * 40,
};

static uint8_t s_adc_read_buf[kAdcReadBufLen];
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static const char* const TAG = "adc_mgr";  //

esp_err_t adc_mgr_init(const AdcMgrConfig adc_mgr_config) {
  esp_err_t err = adc_mgr_init_handle_preconditions(adc_mgr_config);
  if (err != ESP_OK) {
    return err;
  }

  adc_cali_curve_fitting_config_t cali_cfg = {
      .unit_id = ADC_UNIT_1,
      // The channel doesn't make a difference, so we just default to 0.
      // https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32c3/api-reference/peripherals/adc_calibration.html#adc-calibration-curve-fitting-scheme
      .chan = ADC_CHANNEL_0,
      .atten = ADC_ATTEN_DB_12,
      .bitwidth = SOC_ADC_DIGI_MAX_BITWIDTH,
  };

  ESP_RETURN_ON_ERROR(
      adc_cali_create_scheme_curve_fitting(&cali_cfg, &s_calibration_handle),
      TAG, "Couldn't create adc calibration scheme");

  // Since every channel has the same sample rate internally, we set the ADC
  // to the largest and then during reads make sure to downsample by throwing
  // away values for channels with lower sample rates.
  uint32_t largest_desired_sample_rate =
      adc_mgr_config.channel_configs[0].sample_rate;
  for (int i = 1; i < adc_mgr_config.channel_configs_len; ++i) {
    largest_desired_sample_rate =
        MAX(adc_mgr_config.channel_configs[i].sample_rate,
            largest_desired_sample_rate);
  }

  // If the requested sample rate is lower than the minimum supported sample
  // rate, we instead find the lowest multiple of the requested sample rate
  // that is above the minimum supported sample rate.
  if (largest_desired_sample_rate < SOC_ADC_SAMPLE_FREQ_THRES_LOW) {
    // + 1 since we want the ceiling.
    const uint32_t multiple =
        (SOC_ADC_SAMPLE_FREQ_THRES_LOW / largest_desired_sample_rate) + 1;
    const uint32_t new_largest_desired_sample_rate =
        largest_desired_sample_rate * multiple;

    ESP_LOGW(TAG,
             "The largest desired sample rate (%u) is lower than the lowest "
             "supported ADC sample rate (%u). Setting the ADC's sample rate to "
             "the nearest multiple that it supports (calling adc_mgr_read will "
             "still provide you values at the correct sample rate).",
             largest_desired_sample_rate, SOC_ADC_SAMPLE_FREQ_THRES_LOW,
             largest_desired_sample_rate * multiple);

    largest_desired_sample_rate = new_largest_desired_sample_rate;
  }

  // We don't support using channel sampling rates that aren't divisible by the
  // actual sampling rate.
  for (int i = 0; i < adc_mgr_config.channel_configs_len; ++i) {
    const AdcMgrChannelConfig channel_config =
        adc_mgr_config.channel_configs[i];
    if ((channel_config.sample_rate % largest_desired_sample_rate) != 0) {
      ESP_LOGW(TAG,
               "The desired sample rate for channel %d (%u) isn't evenly "
               "divisible by the ADC's sample rate (%u). The closest higher "
               "sample rate that's evenly divisible by the actual sample rate "
               "will be used instead.",
               channel_config.channel, channel_config.sample_rate,
               largest_desired_sample_rate);
    }
  }

  const uint32_t buffer_ms_in_hz =
      1000 / adc_mgr_config.ms_worth_of_buffer_size;

  // Configure ADC in continuous mode.
  const adc_continuous_handle_cfg_t handle_cfg = {
      .max_store_buf_size = SOC_ADC_DIGI_RESULT_BYTES *
                            adc_mgr_config.channel_configs_len *
                            largest_desired_sample_rate / buffer_ms_in_hz,
      .conv_frame_size =
          SOC_ADC_DIGI_DATA_BYTES_PER_CONV * adc_mgr_config.channel_configs_len,
      // If we end up filling the buffer pool, we care more about the most
      // recent values. However, the caller should be setting the buffer pool to
      // be large enough for this to not happen in the first place.
      .flags.flush_pool = true,
  };

  ESP_LOGI(TAG, "Initializing ADC with max buf size = %uB",
           handle_cfg.max_store_buf_size);

  err = adc_continuous_new_handle(&handle_cfg, &s_handle);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "adc_continuous_new_handle failed : %s",
             esp_err_to_name(err));
    adc_cali_delete_scheme_curve_fitting(s_calibration_handle);
    return err;
  }

  s_sample_rate = largest_desired_sample_rate;
  s_channels_count = adc_mgr_config.channel_configs_len;

  // Save the sample rates.
  for (int i = 0; i < adc_mgr_config.channel_configs_len; ++i) {
    const AdcMgrChannelConfig channel_cfg = adc_mgr_config.channel_configs[i];
    s_desired_sample_rates_per_channel[channel_cfg.channel] =
        channel_cfg.sample_rate;
  }

  // Save the amount of values in each channel's periods.
  for (int i = 0; i < SOC_ADC_MAX_CHANNEL_NUM; ++i) {
    s_values_in_channel_periods[i] =
        s_sample_rate / s_desired_sample_rates_per_channel[i];
  }

  // Turn our channel configs into esp_adc's channel configs.
  adc_digi_pattern_config_t channel_configs[SOC_ADC_MAX_CHANNEL_NUM] = {0};
  for (int i = 0; i < adc_mgr_config.channel_configs_len; ++i) {
    channel_configs[i] = (adc_digi_pattern_config_t){
        .channel = adc_mgr_config.channel_configs[i].channel,
        .atten = ADC_ATTEN_DB_12,
        .bit_width = SOC_ADC_DIGI_MAX_BITWIDTH,
        // ADC1 is the only one supported.
        .unit = ADC_UNIT_1,
    };
  }

  const adc_continuous_config_t config = {
      .adc_pattern = channel_configs,
      .pattern_num = adc_mgr_config.channel_configs_len,
      .sample_freq_hz = s_sample_rate,
      .conv_mode = ADC_CONV_SINGLE_UNIT_1,
      .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
  };

  err = adc_continuous_config(s_handle, &config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "adc_continuous_config failed : %s", esp_err_to_name(err));
    goto adc_mgr_init_cleanup;
  }

  err = adc_continuous_start(s_handle);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "adc_continuous_start failed : %s", esp_err_to_name(err));
    goto adc_mgr_init_cleanup;
  }

  s_inited = true;
  ESP_LOGI(TAG, "ADC manager initialized successfully");

  for (int i = 0; i < adc_mgr_config.channel_configs_len; ++i) {
    const adc_channel_t channel = adc_mgr_config.channel_configs[i].channel;
    ESP_LOGI(TAG, "Registered channel %d", channel);
  }

  return ESP_OK;

adc_mgr_init_cleanup:
  adc_continuous_deinit(s_handle);
  adc_cali_delete_scheme_curve_fitting(s_calibration_handle);
  adc_mgr_reset_global_values();
  return err;
}

esp_err_t adc_mgr_deinit(void) {
  if (!s_inited) {
    return ESP_OK;
  }

  {
    const esp_err_t err = adc_continuous_stop(s_handle);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "adc_continuous_stop failed : %s", esp_err_to_name(err));
    }
  }

  {
    const esp_err_t err = adc_continuous_deinit(s_handle);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "adc_continuous_deinit failed : %s", esp_err_to_name(err));
    }
  }
  adc_mgr_reset_global_values();

  return ESP_OK;
}

esp_err_t adc_mgr_read(AdcMgrReadResults* inout_results,
                       const uint32_t timeout_ms) {
  if (!s_inited || inout_results == NULL) {
    ESP_LOGW(TAG, "Invalid arg to adc_mgr_read: inited=%d, inoutptr=%p",
             s_inited, inout_results);
    return ESP_ERR_INVALID_ARG;
  }

  const TickType_t starting_tick = xTaskGetTickCount();

  esp_err_t err = ESP_OK;
  bool wrote_value = false;
  uint32_t read_bytes = 0;
  do {
    // We want to make sure the overall timeout is `timeout_ms`, not use
    // `timeout_ms` every new call to `adc_continuous_read`.
    const uint32_t ms_since_start =
        pdTICKS_TO_MS(xTaskGetTickCount() - starting_tick);
    const uint32_t remaining_timeout_ms =
        MAX((int32_t)timeout_ms - (int32_t)ms_since_start, 0);

    err =
        adc_continuous_read(s_handle, (uint8_t*)s_adc_read_buf, kAdcReadBufLen,
                            &read_bytes, remaining_timeout_ms);
    switch (err) {
      case ESP_OK: {
        const adc_digi_output_data_t* outputs =
            (adc_digi_output_data_t*)s_adc_read_buf;
        const uint32_t output_count = read_bytes / sizeof(*outputs);

        wrote_value = write_results_to_channel_buffers(outputs, output_count,
                                                       inout_results);
        break;
      }
      case ESP_ERR_INVALID_STATE: {
        ESP_LOGW(TAG,
                 "Invalid state while reading from ADC, internal buffer "
                 "probably filled up.");
        return ESP_ERR_INVALID_STATE;
      }
      case ESP_ERR_TIMEOUT: {
        if (wrote_value) {
          // Timing out while having written a value means we've finished
          // reading all buffered values in cases where the ADC's buffer pool is
          // larger than the caller's provided buffers.
          return ESP_OK;
        }

        // If we got a timeout without writing a value, we treat it as an actual
        // timeout.
        return ESP_ERR_TIMEOUT;
      }
      default: {
        return err;
      }
    }
  } while (read_bytes >= kAdcReadBufLen);

  return err;
}

// --- Private implementations ---

static void adc_mgr_reset_global_values(void) {
  s_handle = NULL;
  s_inited = false;
  s_sample_rate = 0;
  s_channels_count = 0;
  memset(s_values_read_already_in_period, 0,
         sizeof(s_values_read_already_in_period));
  memset(s_desired_sample_rates_per_channel, 0,
         sizeof(s_desired_sample_rates_per_channel));
  memset(s_values_in_channel_periods, 0, sizeof(s_values_in_channel_periods));
}

// Pushes a value to the buffer.
// Returns true if successful and false if the buffer is full.
static bool adc_mgr_channel_buffer_push(AdcMgrChannelBuffer* buf,
                                        const uint16_t value) {
  if (buf->length >= buf->capacity) {
    return false;
  }

  buf->data[buf->length++] = value;
  return true;
}

// Pushes the read ADC values to the corresponding channel buffers.
static bool write_results_to_channel_buffers(
    const adc_digi_output_data_t* outputs, const uint32_t output_count,
    AdcMgrReadResults* inout_results) {
  bool wrote_value = false;
  for (size_t i = 0; i < output_count; ++i) {
    const adc_channel_t channel = outputs[i].type2.channel;
    int millivolts = 0;
    esp_err_t err = adc_cali_raw_to_voltage(s_calibration_handle,
                                            outputs[i].type2.data, &millivolts);
    ESP_ERROR_CHECK_WITHOUT_ABORT(err);
    const uint32_t values_in_period = s_values_in_channel_periods[channel];

    // Only write the first value in a channel's period, ignore the rest.
    if (++s_values_read_already_in_period[channel] == values_in_period) {
      const bool success = adc_mgr_channel_buffer_push(
          &inout_results->channel_buffers[channel], millivolts);
      if (!success) {
        ESP_LOGW(TAG, "Channel buffer for ADC values is full.");
        // In a debug build we don't want this situation to be missed.
        assert(false && "Channel buffer for ADC values is full.");
      } else {
        wrote_value = true;
      }
      s_values_read_already_in_period[channel] = 0;
    }
  }
  return wrote_value;
}

static esp_err_t adc_mgr_init_handle_preconditions(
    const AdcMgrConfig adc_mgr_config) {
  if (s_inited) {
    ESP_LOGE(TAG, "The ADC manager has already been initialized.");
    return ESP_ERR_NOT_ALLOWED;
  }

  if (adc_mgr_config.channel_configs == NULL) {
    ESP_LOGE(TAG, "The channel configs array passed to adc_mgr_init was NULL.");
    return ESP_ERR_INVALID_ARG;
  }

  if (adc_mgr_config.channel_configs_len > SOC_ADC_MAX_CHANNEL_NUM) {
    ESP_LOGE(
        TAG,
        "%d channels were passed to adc_mgr_init, but only %d are supported",
        adc_mgr_config.channel_configs_len, SOC_ADC_MAX_CHANNEL_NUM);
    return ESP_ERR_INVALID_SIZE;
  }

  if (adc_mgr_config.channel_configs_len == 0) {
    ESP_LOGE(TAG, "No channels were passed to adc_mgr_init.");
    return ESP_ERR_INVALID_SIZE;
  }

  if (adc_mgr_config.ms_worth_of_buffer_size == 0) {
    ESP_LOGE(TAG, "No ms_worth_of_buffer_size was provided to adc_mgr_init.");
    return ESP_ERR_INVALID_SIZE;
  }

  // Check that multiple configs for the same channel aren't provided.
  {
    bool channel_is_initialized[SOC_ADC_MAX_CHANNEL_NUM] = {false};

    for (int i = 0; i < adc_mgr_config.channel_configs_len; ++i) {
      const adc_channel_t channel = adc_mgr_config.channel_configs[i].channel;
      if (channel_is_initialized[channel]) {
        ESP_LOGE(TAG,
                 "Multiple configs for channel %d were passed to adc_mgr_init.",
                 channel);
        return ESP_ERR_INVALID_ARG;
      }
      channel_is_initialized[channel] = true;
    }
  }

  // Make sure that we actually support the requested sample rates for the
  // different channels.
  for (int i = 0; i < adc_mgr_config.channel_configs_len; ++i) {
    const uint32_t desired_sample_rate =
        adc_mgr_config.channel_configs[i].sample_rate;
    if (desired_sample_rate == 0 ||
        desired_sample_rate > SOC_ADC_SAMPLE_FREQ_THRES_HIGH) {
      ESP_LOGE(TAG,
               "The desired sample rate of %u for channel %d is not supported.",
               desired_sample_rate, adc_mgr_config.channel_configs[i].channel);
      return ESP_ERR_INVALID_ARG;
    }
  }

  return ESP_OK;
}
