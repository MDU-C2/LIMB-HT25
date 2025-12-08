#include "adc_manager.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "potentiometer.h"

static const char* const TAG = "potentiometer_example";

#define LIMB_ARR_LEN(x) (sizeof(x) / sizeof(*(x)))

enum {
  kPotentiometerChannel = ADC_CHANNEL_2,
  kPotentiometerSampleRate = 1000,
  kPotentiometerBufCapacity = 1024,
};

uint16_t s_pot_underlying_buf[kPotentiometerBufCapacity] = {0};

void app_main(void) {
  // Setting up the ADC manager.
  AdcMgrChannelConfig channel_configs[] = {
      {
          .channel = kPotentiometerChannel,
          .sample_rate = kPotentiometerSampleRate,
      },
  };

  AdcMgrConfig mgr_config = {
      .channel_configs = channel_configs,
      .channel_configs_len = LIMB_ARR_LEN(channel_configs),
      .ms_worth_of_buffer_size = 100,
  };

  {
    esp_err_t err = adc_mgr_init(mgr_config);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error initializing ADC manager: %s", esp_err_to_name(err));
      return;
    }
  }

  AdcMgrReadResults results = {
      .channel_buffers =
          {
              [kPotentiometerChannel] =
                  {
                      .data = s_pot_underlying_buf,
                      .capacity = kPotentiometerBufCapacity,
                  },
          },
  };

  printf("pot degrees: %.2f", 1.f);

  // Wait to make sure we have ADC values to read.
  vTaskDelay(pdMS_TO_TICKS(10));

  // Read our potentiometer values.
  esp_err_t err = adc_mgr_read(&results, 0);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Error reading from ADC manager: %s", esp_err_to_name(err));
    return;
  }

  const AdcMgrChannelBuffer* pot_channel_buffer =
      &results.channel_buffers[kPotentiometerChannel];

  ESP_LOGI(TAG, "Read %u potentiometer values from ptr: %p with capacity: %u",
           pot_channel_buffer->length, pot_channel_buffer->data,
           pot_channel_buffer->capacity);

  for (int i = 0; i < pot_channel_buffer->length; ++i) {
    ESP_LOGI(TAG, "Reading value %d", i);
    uint16_t value = pot_channel_buffer->data[i];

    // Set the angle limits of your potentiometer and the ADC bitwidth used.
    const Potentiometer potentiometer = {
        .min_degree = -90.0F,
        .max_degree = 90.0F,
        .adc_bitwidth = SOC_ADC_DIGI_MAX_BITWIDTH,
    };

    // Then you can convert the ADC values to the corresponding degrees.
    float degrees = potentiometer_adc_to_degrees(potentiometer, value);
    ESP_LOGI(TAG, "Raw ADC value: %u, corresponding degree: %f", value,
             degrees);
  }
}
