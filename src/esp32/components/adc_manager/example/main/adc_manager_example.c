#include "adc_manager.h"
#include "esp_err.h"
#include "esp_log.h"
#include "hal/adc_types.h"

void app_main(void) {
  {
    esp_err_t err = adc_mgr_init();
    ESP_ERROR_CHECK(err);
  }

  adc_mgr_handle_t handle = adc_mgr_register_channel(ADC_CHANNEL_0, NULL);

  for (int i = 0; i < 100; ++i) {
    int value = 0;
    {
      esp_err_t err = adc_mgr_read(handle, &value);
      ESP_ERROR_CHECK(err);
    }
    ESP_LOGI("adc_manager example", "Read value: %d", value);
  }

  {
    esp_err_t err = adc_mgr_deinit();
    ESP_ERROR_CHECK(err);
  }
}
