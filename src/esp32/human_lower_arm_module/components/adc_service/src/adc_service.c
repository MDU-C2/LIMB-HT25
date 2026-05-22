#include "adc_service.h"

#include <string.h>

#include "adc_manager.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "hal/adc_types.h"
#include "limb_utils.h"
#include "sensors_service.h"

static const char* TAG = "ADC_SERVICE_STREAM";

// --- Synchronization & State ---
static EventGroupHandle_t s_adc_event_group;
static portMUX_TYPE s_adc_mux = portMUX_INITIALIZER_UNLOCKED;

// --- Channels & Calibration ---
typedef enum {
  kEmgChannel = ADC_CHANNEL_2,
  kPiezoChannel = ADC_CHANNEL_3,
} AdcChannel;

// --- Streaming Data (Micro-packets) ---
static emg_micro_packet_t s_emg_packet;
static piezo_micro_packet_t s_piezo_packet;
static uint32_t s_emg_seq = 0;
static uint32_t s_piezo_seq = 0;

// --- ADC config ---
const AdcMgrChannelConfig kAdcChannelConfigs[] = {
    {
        .channel = kEmgChannel,
        .sample_rate = kEmgFrequency,
    },
    {
        .channel = kPiezoChannel,
        .sample_rate = kPiezoFrequency,
    }};

const AdcMgrConfig kAdcMgrConfig = {
    .channel_configs = kAdcChannelConfigs,
    .channel_configs_len = LIMB_ARR_LEN(kAdcChannelConfigs),
    .ms_worth_of_buffer_size = 200};

// --- DMA Temporary Buffers (Reduced size to optimize RAM) ---
#define ADC_READ_TEMP_CAPACITY 512
static uint16_t s_temp_data_emg[ADC_READ_TEMP_CAPACITY];
static uint16_t s_temp_data_piezo[ADC_READ_TEMP_CAPACITY];

// --- Flow Control ---
typedef struct {
  uint16_t* data;
  uint16_t capacity;
  uint16_t length;
} SampleBuffer;

// Map from adc_channel_t to the corresponding channel's sample buffer.
static SampleBuffer s_channel_sample_buffers[SOC_ADC_MAX_CHANNEL_NUM] = {
    [kEmgChannel] =
        {
            .data = s_emg_packet.data,
            .capacity = LIMB_ARR_LEN(s_emg_packet.data),
        },
    [kPiezoChannel] =
        {
            .data = s_piezo_packet.data,
            .capacity = LIMB_ARR_LEN(s_piezo_packet.data),
        },
};

static void process_emg_sample(adc_channel_t channel, uint16_t millivolt) {
  SampleBuffer* sample_buffer = &s_channel_sample_buffers[channel];
  sample_buffer->data[sample_buffer->length++] = millivolt;

  if (sample_buffer->length >= sample_buffer->capacity) {
    sample_buffer->length = 0;

    s_emg_packet.seq = s_emg_seq++;
    xEventGroupSetBits(s_adc_event_group, ADC_EMG_STREAM_BIT);
  }
}

static void process_piezo_sample(adc_channel_t channel, uint16_t millivolt) {
  SampleBuffer* sample_buffer = &s_channel_sample_buffers[channel];
  sample_buffer->data[sample_buffer->length++] = millivolt;

  if (sample_buffer->length >= sample_buffer->capacity) {
    sample_buffer->length = 0;

    s_piezo_packet.seq = s_piezo_seq++;
    xEventGroupSetBits(s_adc_event_group, ADC_PIEZO_STREAM_BIT);
  }
}

/**
 * @brief Processes a single raw sample, applies calibration, and fills
 * micro-packets.
 * * For EMG: Handles dual-channel interleaving. Sets event bit only when BOTH
 * channels reach 40 samples.
 */
static void process_new_sample(AdcChannel channel, uint16_t value) {
  switch (channel) {
    case kEmgChannel: {
      process_emg_sample(channel, value);
      break;
    }
    case kPiezoChannel: {
      process_piezo_sample(channel, value);
      break;
    }
  }
}

/**
 * @brief High-priority task that fetches DMA data from ADC Manager every 10ms.
 */
static void adc_task(void* pvParameters) {
  // Map temporary buffers to active channels
  AdcMgrReadResults res = {
      .channel_buffers =
          {
              [kEmgChannel] =
                  {
                      .data = s_temp_data_emg,
                      .capacity = ADC_READ_TEMP_CAPACITY,
                  },
              [kPiezoChannel] =
                  {
                      .data = s_temp_data_piezo,
                      .capacity = ADC_READ_TEMP_CAPACITY,
                  },
          },
  };

  TickType_t xLastWakeTime = xTaskGetTickCount();
  static_assert(kEmgPacketSendRateHz == kImuPacketSendRateHz &&
                    kEmgPacketSendRateHz == kPiezoPacketSendRateHz,
                "We assume all sensors send at the same rate");
  const TickType_t xFrequency = pdMS_TO_TICKS(1000 / kEmgPacketSendRateHz);

  while (1) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    // Fetch results from DMA through ADC Manager
    esp_err_t err = adc_mgr_read(&res, 0);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Error reading from ADC: %s", esp_err_to_name(err));
      continue;
    }

    for (int i = 0; i < LIMB_ARR_LEN(kAdcChannelConfigs); i++) {
      const adc_channel_t channel = kAdcChannelConfigs[i].channel;
      AdcMgrChannelBuffer* buf = &res.channel_buffers[channel];

      for (uint32_t j = 0; j < buf->length; j++) {
        process_new_sample(channel, buf->data[j]);
      }
      buf->length = 0;  // Clear length after processing
    }
  }
}

esp_err_t adc_service_init(EventGroupHandle_t event_group) {
  s_adc_event_group = event_group;

  // Init the ADC Manager DMA engine
  esp_err_t err = adc_mgr_init(kAdcMgrConfig);
  if (err != ESP_OK) return err;

  // Make sure sample buffers are empty.
  for (int i = 0; i < LIMB_ARR_LEN(kAdcChannelConfigs); i++) {
    s_channel_sample_buffers[i].length = 0;
  }

  // Launch processing task on Core 0
  return xTaskCreatePinnedToCore(adc_task, "adc_task", 4096, NULL, 10, NULL,
                                 0) == pdPASS
             ? ESP_OK
             : ESP_FAIL;
}

size_t adc_service_get_emg_micropacket(void* dest) {
  taskENTER_CRITICAL(&s_adc_mux);
  memcpy(dest, &s_emg_packet, sizeof(emg_micro_packet_t));
  taskEXIT_CRITICAL(&s_adc_mux);
  return sizeof(emg_micro_packet_t);
}

size_t adc_service_get_piezo_micropacket(void* dest) {
  taskENTER_CRITICAL(&s_adc_mux);
  memcpy(dest, &s_piezo_packet, sizeof(piezo_micro_packet_t));
  taskEXIT_CRITICAL(&s_adc_mux);
  return sizeof(piezo_micro_packet_t);
}
