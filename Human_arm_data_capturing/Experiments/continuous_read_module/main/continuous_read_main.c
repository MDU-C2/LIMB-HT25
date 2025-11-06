
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "adc_emg_driver.h" 

static const char *TAG = "EMG_APP";

void app_main(void)
{
    // Step 1: Initialize the EMG driver.
    if (emg_driver_init() != ESP_OK) {
        ESP_LOGE(TAG, "EMG driver initialization failed. Halting.");
        return;
    }

    // Step 2: Start the data acquisition.
    emg_driver_start();
    ESP_LOGI(TAG, "EMG driver started. Waiting for data...");

    // This variable will hold the complete data packet when it's ready.
    emg_data_packet_t data_packet;

    // Step 3: Main application loop.
while (1) {
        // This is the "engine" of the driver. It must be called repeatedly.
        // It will block and wait for new data internally.
        emg_driver_process_data();

        // After processing, check if a full window is ready.
        if (emg_driver_is_window_ready()) {
            
            // If ready, get the complete packet.
            emg_driver_get_packet(&data_packet);

            ESP_LOGI(TAG, "New data packet received!");

            ESP_LOGI(TAG, "  -> Last CH0 value: %dmV", data_packet.emg_ch0_window[EMG_WINDOW_SIZE - 1]);
        #if DUAL_CHANNEL_MODE
            ESP_LOGI(TAG, "  -> Last CH1 value: %dmV", data_packet.emg_ch1_window[EMG_WINDOW_SIZE - 1]);
        #endif
            
            // ==========================================================
            // == PLACEHOLDER FOR BLUETOOTH AND IMU LOGIC
            // ==========================================================
            // Here, you would read the IMU data and then send the
            // combined 'data_packet' over Bluetooth.
            // ==========================================================
        }
    }
}
