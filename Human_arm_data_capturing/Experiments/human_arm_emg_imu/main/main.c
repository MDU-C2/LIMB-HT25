/*
 * main.c
 *
 * Main application for multi-context data capture.
 *
 * This program implements a single-loop, synchronous data capture system.
 * It uses the EMG (ADC) driver as a "pacemaker" (Master) to trigger
 * instantaneous IMU (Slave) readings.
 *
 * The result is a single, combined data structure containing a full
 * window of EMG data and the corresponding IMU snapshot for that exact moment.
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

// 1. INCLUDE THE "MENUS" (APIs) FOR BOTH MODULES
#include "adc_emg_driver.h"
#include "imu_driver.h"

static const char *TAG = "APP_SYNC";

// 2. THE MULTI-CONTEXT (COMBINED) DATA STRUCTURE
typedef struct {
    emg_data_packet_t emg_data; // EMG data packet (contains the 800-sample int16_t window)
    lsm6dso32_data_t  imu_data; // IMU data packet (contains accel and gyro snapshot)
} combined_data_packet_t;


// =========================================================================
// == APP_MAIN: THE SINGLE, SYNCHRONIZED LOOP
// =========================================================================
void app_main(void)
{
    // --- 1. INITIALIZE BOTH MODULES ---
    ESP_LOGI(TAG, "Initializing IMU Module...");
    if (imu_init() != ESP_OK) {
        ESP_LOGE(TAG, "IMU initialization failed. Halting.");
        return; // Stop if sensor fails
    }

    ESP_LOGI(TAG, "Initializing EMG Module...");
    if (emg_driver_init() != ESP_OK) {
        ESP_LOGE(TAG, "EMG initialization failed. Halting.");
        return; // Stop if ADC fails
    }

    // Start the ADC's background sampling engine
    emg_driver_start();
    ESP_LOGI(TAG, "All modules initialized. Entering capture loop.");

    // --- 2. CREATE THE VARIABLE FOR OUR COMBINED PACKET ---
    // This local variable will live on the main task's stack
    // (We already increased the stack size in menuconfig for this)
    combined_data_packet_t my_packet;

    // --- 3. MAIN APPLICATION LOOP (The "Pacemaker" Loop) ---
    while (1)
    {
        // a) Block (sleep) this task. Wait for the ADC ISR to send a
        //    notification that a new batch of DMA data is ready.
        //    FreeRTOS ensures this is 0% CPU usage while sleeping.
        emg_driver_process_data();

        // b) We are now awake. Check if the driver has processed
        //    enough new samples (EMG_STEP_SIZE) to create a full window.
        if (emg_driver_is_window_ready()) 
        {
            // --- SYNCHRONIZATION EVENT! ---
            // The EMG window is ready. We must capture BOTH sensors NOW.

            // c) Get the complete 800-sample EMG window
            //    (This is just a fast internal memcpy)
            emg_driver_get_packet(&my_packet.emg_data);
            
            // d) Get the *most recent* instantaneous snapshot from the IMU
            //    (This is a quick I2C read)
            imu_read_data(&my_packet.imu_data);

            // e) PACKET READY! The 'my_packet' struct now contains
            //    data from both sensors, synchronized to this moment.
            //    (This is where you would send 'my_packet' over Bluetooth)

            // --- Verification Log ---
            ESP_LOGI(TAG, "== MULTI-CONTEXT PACKET CREATED ==");
            ESP_LOGI(TAG, "  EMG (last mV): %d", my_packet.emg_data.emg_ch0_window[EMG_WINDOW_SIZE - 1]);
            ESP_LOGI(TAG, "  IMU (Accel Z m/s²): %.2f", my_packet.imu_data.accel.z);
        }
    }
}

