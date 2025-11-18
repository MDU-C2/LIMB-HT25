#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "potentiometer.h"

static const char *TAG = "POTENTIOMETER_TEST";

void app_main(void)
{
    // Initialize potentiometer with default configuration (GPIO0)
    ESP_ERROR_CHECK(potentiometer_init(NULL));
    
    ESP_LOGI(TAG, "Potentiometer test started");

    // Main loop - read ADC values every 500ms
    while (1) {
        int raw_value;
        int voltage_mv;
        uint16_t normalized;

        // Read raw ADC value
        esp_err_t ret = potentiometer_read_raw(&raw_value);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "ADC Raw: %d", raw_value);
        } else {
            ESP_LOGE(TAG, "Failed to read raw value: %s", esp_err_to_name(ret));
        }

        // Read voltage (if calibration is available)
        ret = potentiometer_read_voltage(&voltage_mv);
        if (ret == ESP_OK && voltage_mv > 0) {
            ESP_LOGI(TAG, "Voltage: %d mV", voltage_mv);
        }

        // Read normalized value (0-1000)
        ret = potentiometer_read_normalized(&normalized);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "Normalized: %d/1000", normalized);
        }

        ESP_LOGI(TAG, "---");

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
