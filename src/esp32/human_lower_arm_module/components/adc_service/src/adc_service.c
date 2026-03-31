#include "adc_service.h"
#include "adc_manager.h" 
#include "esp_log.h"
#include "freertos/task.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include <string.h>
#include "esp_timer.h"
#include "hal/adc_types.h"
#include "limb_utils.h"

static const char *TAG = "ADC_SERVICE_STREAM";

// --- Synchronization & State ---
static EventGroupHandle_t s_adc_event_group;
static portMUX_TYPE s_adc_mux = portMUX_INITIALIZER_UNLOCKED; 

// --- Channels & Calibration ---
enum {
    kEmg1Channel = ADC_CHANNEL_2,
    kEmg2Channel = ADC_CHANNEL_0,
    kPiezoChannel = ADC_CHANNEL_3,
};
static const adc_channel_t s_physical_channels[ADC_SERVICE_CHANNEL_COUNT] = {kEmg1Channel, kEmg2Channel, kPiezoChannel};
static adc_cali_handle_t s_adc_cali_handle[ADC_SERVICE_CHANNEL_COUNT] = {NULL};

// --- Streaming Data (Micro-packets) ---
static emg_micro_packet_t s_emg_packet;
static piezo_micro_packet_t s_piezo_packet;
static uint32_t s_emg_seq = 0;
static uint32_t s_piezo_seq = 0;

// --- ADC config ---
const AdcMgrChannelConfig kAdcChannelConfigs[] = {
  {
    .channel = kEmg1Channel,
    .sample_rate = ADC_EMG_SAMPLE_RATE_HZ,
  },
  {
    .channel = kEmg2Channel,
    .sample_rate = ADC_EMG_SAMPLE_RATE_HZ,
  },
  {
    .channel = kPiezoChannel,
    .sample_rate = ADC_PIEZO_SAMPLE_RATE_HZ,
  }
};

const AdcMgrConfig kAdcMgrConfig = {
    .channel_configs = kAdcChannelConfigs, 
    .channel_configs_len = LIMB_ARR_LEN(kAdcChannelConfigs),
    .ms_worth_of_buffer_size = 200
};

// --- DMA Temporary Buffers (Reduced size to optimize RAM) ---
#define ADC_READ_TEMP_CAPACITY 512 
static uint16_t s_temp_data_emg[ADC_READ_TEMP_CAPACITY];
static uint16_t s_temp_data_emg_1[ADC_READ_TEMP_CAPACITY];
static uint16_t s_temp_data_piezo[ADC_READ_TEMP_CAPACITY];

// --- Flow Control ---
typedef struct {
    uint16_t current_index;
} stream_control_t;

static stream_control_t s_channel_controls[ADC_SERVICE_CHANNEL_COUNT];
static uint8_t s_piezo_sample_counter = 0;
static bool s_emg1_ready_flag = false;
static bool s_emg2_ready_flag = false;

static void process_emg_sample(stream_control_t *state, int local_index, uint16_t millivolt) {
    // Interleaving: EMG1 fills first half (0-39), EMG2 fills second half (40-79)
    if (local_index == 0) s_emg_packet.data[state->current_index] = millivolt;
    else s_emg_packet.data[state->current_index + ADC_EMG_MICRO_SIZE] = millivolt;

    state->current_index++;
    if (state->current_index >= ADC_EMG_MICRO_SIZE) {
        state->current_index = 0;
        if (local_index == 0) s_emg1_ready_flag = true;
        else s_emg2_ready_flag = true;

        // Trigger notification only when both EMG channels are synchronized
        if (s_emg1_ready_flag && s_emg2_ready_flag) {
            s_emg_packet.header = 0xAABB;
            s_emg_packet.seq = s_emg_seq++;
            xEventGroupSetBits(s_adc_event_group, ADC_EMG_STREAM_BIT);
            s_emg1_ready_flag = s_emg2_ready_flag = false;
        }
    }
}

static void process_piezo_sample(stream_control_t *state, uint16_t millivolt) {
    // Decimating 4kHz input to 1kHz output
    if (++s_piezo_sample_counter < 4) return;
    s_piezo_sample_counter = 0;

    s_piezo_packet.data[state->current_index++] = millivolt;
    if (state->current_index >= ADC_PIEZO_MICRO_SIZE) {
        state->current_index = 0;
        s_piezo_packet.header = 0xEEFF;
        s_piezo_packet.seq = s_piezo_seq++;
        xEventGroupSetBits(s_adc_event_group, ADC_PIEZO_STREAM_BIT);
    }
}

/**
 * @brief Processes a single raw sample, applies calibration, and fills micro-packets.
 * * For EMG: Handles dual-channel interleaving. Sets event bit only when BOTH channels reach 40 samples.
 * For PIEZO: Implements a 4:1 decimation (from 4kHz to 1kHz).
 */
static void process_new_sample(int local_index, uint16_t value) {
    stream_control_t *state = &s_channel_controls[local_index];
    
    // Apply ADC Calibration (Raw to Voltage mV)
    int voltage_mv = 0;
    if (s_adc_cali_handle[local_index] != NULL) {
        adc_cali_raw_to_voltage(s_adc_cali_handle[local_index], value, &voltage_mv);
    } else {
        voltage_mv = value;
    }
    uint16_t val = (uint16_t)voltage_mv;

    // --- EMG Logic (Channels 0 and 1) ---
    if (local_index == 0 || local_index == 1) {
        process_emg_sample(state, local_index, val);
    } 
    // --- PIEZO Logic (Channel 2 with 4:1 Decimation) ---
    else if (local_index == 2) {
        process_piezo_sample(state, val);
    }
}

/**
 * @brief High-priority task that fetches DMA data from ADC Manager every 10ms.
 */
static void adc_task(void *pvParameters) {
    // Map temporary buffers to active channels
    AdcMgrReadResults res = {
      .channel_buffers = {
        [kEmg1Channel] = {
          .data = s_temp_data_emg,
          .capacity = ADC_READ_TEMP_CAPACITY,
        },
        [kEmg2Channel] = {
          .data = s_temp_data_emg_1,
          .capacity = ADC_READ_TEMP_CAPACITY,
        },
        [kPiezoChannel] = {
          .data = s_temp_data_piezo,
          .capacity = ADC_READ_TEMP_CAPACITY,
        },
      },
    }; 

    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(10); // 100Hz processing rate

    while (1) {
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
        uint64_t now = esp_timer_get_time();

        // Stamp the packet with the system time at the start of a new batch
        if (s_channel_controls[0].current_index == 0 && s_channel_controls[1].current_index == 0) {
            s_emg_packet.timestamp = now;
        }
        if (s_channel_controls[2].current_index == 0) {
            s_piezo_packet.timestamp = now;
        }

        // Fetch results from DMA through ADC Manager
        if (adc_mgr_read(&res, 0) == ESP_OK) {
            for (int i = 0; i < ADC_SERVICE_CHANNEL_COUNT; i++) {
                AdcMgrChannelBuffer *buf = &res.channel_buffers[s_physical_channels[i]];
                
                for (uint32_t j = 0; j < buf->length; j++) {
                    process_new_sample(i, buf->data[j]);
                }
                buf->length = 0; // Clear length after processing
            }
        }
    }
}

esp_err_t adc_service_init(EventGroupHandle_t event_group) {
    s_adc_event_group = event_group;

    // Init the ADC Manager DMA engine
    esp_err_t err = adc_mgr_init(kAdcMgrConfig);
    if (err != ESP_OK) return err;

    // Create calibration handles and reset indices
    for (int i = 0; i < LIMB_ARR_LEN(kAdcChannelConfigs); i++) {
        // TODO(johan): The calibration scheme functionality should
        // probably be moved to the ADC manager component so the
        // settings are guaranteed to be consistent.
        adc_cali_curve_fitting_config_t cali_cfg = {
            .unit_id = ADC_UNIT_1,
            .chan = kAdcChannelConfigs[i].channel,
            .atten = ADC_ATTEN_DB_12,
            .bitwidth = SOC_ADC_DIGI_MAX_BITWIDTH,
        };
        adc_cali_create_scheme_curve_fitting(&cali_cfg, &s_adc_cali_handle[i]);
        s_channel_controls[i].current_index = 0;
    }

    // Launch processing task on Core 0
    return xTaskCreatePinnedToCore(adc_task, "adc_task", 4096, NULL, 10, NULL, 0) == pdPASS ? ESP_OK : ESP_FAIL;
}

size_t adc_service_get_emg_micropacket(void *dest) {
    taskENTER_CRITICAL(&s_adc_mux);
    memcpy(dest, &s_emg_packet, sizeof(emg_micro_packet_t));
    taskEXIT_CRITICAL(&s_adc_mux);
    return sizeof(emg_micro_packet_t);
}

size_t adc_service_get_piezo_micropacket(void *dest) {
    taskENTER_CRITICAL(&s_adc_mux);
    memcpy(dest, &s_piezo_packet, sizeof(piezo_micro_packet_t));
    taskEXIT_CRITICAL(&s_adc_mux);
    return sizeof(piezo_micro_packet_t);
}
