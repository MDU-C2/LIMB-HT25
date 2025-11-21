#include <stdio.h>
#include <stdlib.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "piezo.h"

static const char *TAG = "PIEZO_TEST";

void app_main(void)
{
    // Initialize piezo sensor with default configuration (GPIO1, ADC_CHANNEL_1)
    ESP_ERROR_CHECK(piezo_init(NULL));
    
    ESP_LOGI(TAG, "Piezo sensor test started");
    ESP_LOGI(TAG, "Reading ADC values every 100ms...");
    ESP_LOGI(TAG, "Tap or apply pressure to the piezo to see value changes");

    // Main loop - read ADC values every 100ms (fast enough to catch piezo spikes)
    while (1) {
        int raw_value;
        
        // Read raw ADC value from piezo sensor
        esp_err_t ret = piezo_read_raw(&raw_value);
        if (ret == ESP_OK) {
           
            ESP_LOGI(TAG, "Piezo Raw: %d ", raw_value);
        } else {
            ESP_LOGE(TAG, "Failed to read piezo value: %s", esp_err_to_name(ret));
        }

        vTaskDelay(pdMS_TO_TICKS(100));  // 100ms delay for 10 Hz sampling
    }
}

