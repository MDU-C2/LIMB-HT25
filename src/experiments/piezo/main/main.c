#include <stdio.h>
#include <stdlib.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "adc_manager.h"
#include "hal/adc_types.h"

static const char *TAG = "PIEZO_TEST";

void app_main(void)
{
    // Initialize ADC manager
    ESP_ERROR_CHECK(adc_mgr_init());
    ESP_LOGI(TAG, "ADC manager initialized");

    // Configure ADC channel for piezo sensor
    // Adjust ADC_CHANNEL_0 to match your GPIO pin
    // Common ESP32-C3 ADC channels: ADC_CHANNEL_0 (GPIO0), ADC_CHANNEL_1 (GPIO1), etc.
    adc_oneshot_chan_cfg_t adc_config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,  // 12-bit resolution
        .atten = ADC_ATTEN_DB_12,          // 0-3.3V range (good for piezo)
    };

    // Register the ADC channel
    // Change ADC_CHANNEL_0 to the appropriate channel for your piezo GPIO
    adc_mgr_handle_t piezo_handle = adc_mgr_register_channel(ADC_CHANNEL_1, &adc_config);
    
    if (piezo_handle < 0) {
        ESP_LOGE(TAG, "Failed to register ADC channel");
        return;
    }

    ESP_LOGI(TAG, "Piezo sensor test started on ADC Channel 0");
    ESP_LOGI(TAG, "Reading ADC values every 100ms...");
    ESP_LOGI(TAG, "Tap or apply pressure to the piezo to see value changes");

    int max_value = 0;
    int min_value = 4095;
    int sample_count = 0;

    // Main loop - read ADC values every 100ms (fast enough to catch piezo spikes)
    while (1) {
        int raw_value;
        
        // Read raw ADC value
        esp_err_t ret = adc_mgr_read(piezo_handle, &raw_value);
        if (ret == ESP_OK) {
            // Track min/max values to see piezo response range
            if (raw_value > max_value) {
                max_value = raw_value;
            }
            if (raw_value < min_value) {
                min_value = raw_value;
            }
            
            sample_count++;
            
            // Log every 10 samples (every second) or when value changes significantly
            
            ESP_LOGI(TAG, "ADC Raw: %d",raw_value);
            
        } else {
            ESP_LOGE(TAG, "Failed to read ADC value: %s", esp_err_to_name(ret));
        }

        vTaskDelay(pdMS_TO_TICKS(100));  // 100ms delay for 10 Hz sampling
    }
}

