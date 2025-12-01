#include "adc_manager.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "portmacro.h"

static const char* const TAG = "adc_mgr_example";

#define LIMB_ARR_LEN(x) (sizeof(x) / sizeof(*(x)))

enum {
  kEmgChannel = ADC_CHANNEL_0,
  kPiezoChannel = ADC_CHANNEL_1,
  kEmgSampleRate = 4000,
  kPiezoSampleRate = 100,
};

void read_adc_task([[maybe_unused]] void* unused) {
  ESP_LOGI(TAG, "Started adc reading task");

  // 1. First you set up which channels and sample rates you want to use.
  AdcMgrChannelConfig channel_configs[] = {
      {
          .channel = kEmgChannel,
          .sample_rate = kEmgSampleRate,
      },
      {
          .channel = kPiezoChannel,
          .sample_rate = kPiezoSampleRate,
      },
  };

  AdcMgrConfig mgr_config = {
      .channel_configs = channel_configs,
      .channel_configs_len = LIMB_ARR_LEN(channel_configs),
      .ms_worth_of_buffer_size = 100,
  };

  // 2. Then you initialize the ADC manager.
  esp_err_t err = adc_mgr_init(mgr_config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Error initializing ADC manager: %s", esp_err_to_name(err));
    vTaskDelete(NULL);
    return;
  }

  // 3. Then you provide the buffers that the ADC manager should write its
  // results into.
  enum {
    kEmgBufCapacity = 1024,
    kPiezoBufCapacity = 1024,
  };

  uint16_t emg_underlying_buf[kEmgBufCapacity] = {0};
  uint16_t piezo_underlying_buf[kPiezoBufCapacity] = {0};

  AdcMgrReadResults results = {
      .channel_buffers =
          {
              // 4. The index corresponds to the channel, so make sure the 0th
              // index is given the buffer for ADC channel 0, etc.
              [kEmgChannel] =
                  {
                      .data = emg_underlying_buf,
                      .capacity = kEmgBufCapacity,
                  },
              [kPiezoChannel] =
                  {
                      .data = piezo_underlying_buf,
                      .capacity = kPiezoBufCapacity,
                  },
          },
  };

  while (true) {
    // 4. Then you tell the ADC manager to write any read values to the buffers
    // you provided in the AdcMgrReadResults variable.
    // NOTE: You will receive all available values from all registered channels,
    // so make sure you handle all of them.
    esp_err_t err = adc_mgr_read(&results, 0);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error reading from ADC manager: %s", esp_err_to_name(err));
      continue;
    }

    // 5. Use the values however you want. Here we just print them, but you
    // could directly send them over CAN, for example.
    AdcMgrChannelBuffer* emg_channel_buffer =
        &results.channel_buffers[kEmgChannel];
    AdcMgrChannelBuffer* piezo_channel_buffer =
        &results.channel_buffers[kPiezoChannel];

    ESP_LOGI(TAG, "Read %u piezo values and %u EMG values",
             piezo_channel_buffer->length, emg_channel_buffer->length);

    // Print piezo readings.
    for (int i = 0; i < piezo_channel_buffer->length; ++i) {
      uint16_t value = piezo_channel_buffer->data[i];
      // NOTE: Printing is slow, so it'll delay the next call to adc_mgr_read.
      ESP_LOGI(TAG, "Read piezo (%u)", value);
    }

    // Print EMG readings.
    for (int i = 0; i < emg_channel_buffer->length; ++i) {
      uint16_t value = emg_channel_buffer->data[i];
      // NOTE: Printing is slow, so it'll delay the next call to adc_mgr_read.
      ESP_LOGI(TAG, "Read EMG (%u)", value);
    }

    // 6. Make sure you update the length of the buffers after you have used the
    // values!
    piezo_channel_buffer->length = 0;
    emg_channel_buffer->length = 0;

    // Avoid the task watchdog by giving the idle task a chance to run. This
    // effectively limits the rate at which adc_mgr_read can be called. Since
    // the maximum tick rate is 1000 Hz, we potentially get an additional 1 ms
    // delay for each subsequent call. As long as we are able to process the
    // values fast enough, it should still be fine.
    vTaskDelay(1);
  }

  // 7. Deinitialize the ADC manager when you're done with it.
  ESP_ERROR_CHECK(adc_mgr_deinit());

  vTaskDelete(NULL);
}

void app_main(void) {
  BaseType_t err = xTaskCreate(read_adc_task, "read_adc_task",
                               1024 * 2 * 2 * 2 * 2, NULL, 5, NULL);
  if (err != pdPASS) {
    ESP_LOGE(TAG, "ERROR creating send task");
  }
}
