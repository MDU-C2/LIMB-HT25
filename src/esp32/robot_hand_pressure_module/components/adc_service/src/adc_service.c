#include "adc_service.h"

#include "adc_manager.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

static const char* TAG = "ADC_SERVICE";

static const adc_channel_t physical_channels[NUM_FINGERS] = {
    ADC_CHANNEL_0, ADC_CHANNEL_1, ADC_CHANNEL_2, ADC_CHANNEL_3, ADC_CHANNEL_4};

// Variable protegida para guardar el último valor leído
static float s_latest_pressure = 0;
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;

/**
 * @brief Task que vacía el DMA constantemente (como en tu proyecto viejo)
 */
static void adc_cleaner_task(void* pvParameters) {
  AdcMgrReadResults results;
  uint16_t temp_storage[NUM_FINGERS][10];  // Buffer para sacar basura

  while (1) {
    // Configuramos para succionar todo lo que haya
    for (int i = 0; i < NUM_FINGERS; i++) {
      adc_channel_t ch = physical_channels[i];
      results.channel_buffers[ch].data = temp_storage[i];
      results.channel_buffers[ch].capacity = 10;
      results.channel_buffers[ch].length = 0;
    }

    // Leemos con timeout pequeño para no bloquear
    if (adc_mgr_read(&results, 10) == ESP_OK) {
      float sum_all = 0;
      int valid_fingers = 0;

      for (int i = 0; i < NUM_FINGERS; i++) {
        adc_channel_t ch = physical_channels[i];
        if (results.channel_buffers[ch].length > 0) {
          // Tomamos el valor más reciente de este canal
          sum_all +=
              (float)temp_storage[i][results.channel_buffers[ch].length - 1];
          valid_fingers++;
        }
      }

      if (valid_fingers > 0) {
        portENTER_CRITICAL(&s_mux);
        s_latest_pressure = sum_all / valid_fingers;
        portEXIT_CRITICAL(&s_mux);
      }
    }
    // Pequeño respiro para que otras tareas respiren
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

esp_err_t init_adc_service(void) {
  AdcMgrChannelConfig configs[NUM_FINGERS];
  for (int i = 0; i < NUM_FINGERS; i++) {
    configs[i].channel = physical_channels[i];
    configs[i].sample_rate = 1000;
  }

  AdcMgrConfig mgr_config = {
      .channel_configs = configs,
      .channel_configs_len = NUM_FINGERS,
      .ms_worth_of_buffer_size = 500  // Medio segundo de margen
  };

  esp_err_t ret = adc_mgr_init(mgr_config);
  if (ret != ESP_OK) return ret;

  // Lanzamos la tarea de limpieza (La clave del éxito)
  xTaskCreatePinnedToCore(adc_cleaner_task, "adc_cleaner", 4096, NULL, 10, NULL,
                          0);

  return ESP_OK;
}

float get_instant_pressure(void) {
  float val;
  portENTER_CRITICAL(&s_mux);
  val = s_latest_pressure;
  portEXIT_CRITICAL(&s_mux);
  return val;
}

wstats_t get_window_stats(void) {
  wstats_t results = {0.0f, 0.0f};
  float readings[SAMPLES_PER_WIND];
  float sum = 0, sum_sq = 0;

  for (int i = 0; i < SAMPLES_PER_WIND; i++) {
    readings[i] = get_instant_pressure();
    sum += readings[i];
    vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD));
  }

  results.mean = sum / SAMPLES_PER_WIND;
  for (int i = 0; i < SAMPLES_PER_WIND; i++) {
    float diff = readings[i] - results.mean;
    sum_sq += diff * diff;
  }
  results.variance = sum_sq / SAMPLES_PER_WIND;
  return results;
}