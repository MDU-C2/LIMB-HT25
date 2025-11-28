/*
 * SPDX-FileCopyrightText: 2022-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"
#include "esp_log.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_timer.h"

const static char *TAG = "ADC_HAND";

/*---------------------------------------------------------------
 * ADC General Macros & Definitions
 *---------------------------------------------------------------*/
// Defines the number of FSR sensors used (currently 1, connected to ADC_CHANNEL_0).
#define NUM_FINGERS 1
#define EXAMPLE_ADC_ATTEN ADC_ATTEN_DB_12 // ADC attenuation (12dB = ~0V to 3.3V range)

// >>> CONTROL VARIABLES BASED ON VOLTS <<<
#define TARGET_PRESSURE_VOLTS 1.5 // (Setpoint) Target pressure value in Volts (e.g., 1.5V)
#define ALPHA_LPF             0.1 // Coefficient for the Exponential Low-Pass Filter (0 < ALPHA < 1)

// Control Gains (Initial test values)
// The controller output (angle change) is calculated based on the error in VOLTS.
#define KP_GAIN 5.0     // Proportional Gain (Output_Unit / Volt)
#define KD_GAIN 0.5     // Derivative Gain (Output_Unit / (Volt/second))


// Storage structures for ADC reading and calibration data.
static int adc_raw[NUM_FINGERS];
static int voltage_mV[NUM_FINGERS];
static adc_cali_handle_t adc1_cali_handles[NUM_FINGERS];
static bool do_calibration[NUM_FINGERS];

// Array of ADC channels to read.
const adc_channel_t adc_channels[NUM_FINGERS] = {
    ADC_CHANNEL_0 
};

// Function prototypes for ADC calibration.
static bool example_adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle);
static void example_adc_calibration_deinit(adc_cali_handle_t handle);


void app_main(void)
{
    //-------------ADC1 Init---------------//
    adc_oneshot_unit_handle_t adc1_handle;
    adc_oneshot_unit_init_cfg_t init_config1 = {
        .unit_id = ADC_UNIT_1,
    };
    // Initialize ADC unit 1
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config1, &adc1_handle));

    //-------------ADC1 Config & Calibration Loop---------------//
    adc_oneshot_chan_cfg_t config = {
        .atten = EXAMPLE_ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };

    for (int i = 0; i < NUM_FINGERS; i++) {
        // Configure ADC Channel
        ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, adc_channels[i], &config));
        
        // Initialize Calibration for the channel
        do_calibration[i] = example_adc_calibration_init(
            ADC_UNIT_1, 
            adc_channels[i], 
            EXAMPLE_ADC_ATTEN, 
            &adc1_cali_handles[i]
        );
    }
    
    char log_buffer[512]; // Buffer for the complete log line

    static float filtered_voltage_V = 0.0;
    static float control_error = 0.0;
    // Controller state variables
    static float previous_error = 0.0; // Error from the previous cycle e(k-1)
    static float control_output = 0.0; // Final controller output u(k)
    static int64_t last_time_us = 0; // Timestamp of the previous cycle (in microseconds)
    static float T_s_real = 0.01;    // Actual sampling time in seconds (initialized for safety)
    
    //------------------------- Fast Reading Loop -------------------------//
    while (1) {
        
        int len = 0; // Reset log buffer index at the start of each cycle

        for (int i = 0; i < NUM_FINGERS; i++) {
            // 1. Read the RAW ADC value
            ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, adc_channels[i], &adc_raw[i]));
            
            // 2. Convert to Calibrated Voltage
            float volt_V = 0;
            if (do_calibration[i]) {
                ESP_ERROR_CHECK(adc_cali_raw_to_voltage(adc1_cali_handles[i], adc_raw[i], &voltage_mV[i]));
                volt_V = (float)voltage_mV[i] / 1000.0; // Convert mV to V
            }

            // --- Control Logic (Only for Channel 0) ---
            if (i == 0) {
                
                // =================================================================
                // >> REAL SAMPLING TIME CALCULATION (T_s_real) <<
                // =================================================================
                int64_t current_time_us = esp_timer_get_time();
                
                if (last_time_us != 0) {
                    // T_s_real [seconds] = (Current Time - Previous Time) / 1,000,000
                    T_s_real = (float)(current_time_us - last_time_us) / 1000000.0; 
                    
                    // Safety check: prevent division by zero or erratic T_s values
                    if (T_s_real <= 0 || T_s_real > 0.1) T_s_real = 0.01; 
                }
                last_time_us = current_time_us; // Update timestamp for the next cycle
                
                
                // 1. Apply Exponential Low-Pass Filter (LPF)
                // LPF smoothes the noisy sensor readings.
                if (filtered_voltage_V == 0.0) {
                    // Initialize filter on the first run
                    filtered_voltage_V = volt_V;
                } else {
                    // LPF formula: y(k) = alpha * u(k) + (1 - alpha) * y(k-1)
                    // 

                    filtered_voltage_V = ALPHA_LPF * volt_V + (1.0 - ALPHA_LPF) * filtered_voltage_V;
                }
                
                // 2. Calculate Current Error e(k) in VOLTS
                control_error = TARGET_PRESSURE_VOLTS - filtered_voltage_V;
                
                
                // =================================================================
                // >> PD CONTROL CALCULATION <<
                // =================================================================
                // Calculates the control action u(k) based on the current error e(k) 
                // and the rate of change of the error.
                
                // 3. Calculate the Derivative Term (Using T_s_real!)
                // Derivative = (Current Error - Previous Error) / T_s_real
                float derivative_term = (control_error - previous_error) / T_s_real; 

                // 4. Calculate the Control Output u(k)
                float proportional_term = KP_GAIN * control_error;
                float derivative_output = KD_GAIN * derivative_term;
                
                control_output = proportional_term + derivative_output;

                // 5. UPDATE and Clamping
                previous_error = control_error;
                
            }
            // --------------------------------

            // 3. LOG CONSTRUCTION (Outputting key control variables)
            len += sprintf(log_buffer + len, 
                    "| D%d | RAW: %.2f | FILT: %.2f | ERR: %.3f | T_s: %.4f | OUT: %.1f |", 
                    i + 1, 
                    volt_V, 
                    filtered_voltage_V, 
                    control_error,
                    T_s_real, // Shows the actual sampling time
                    control_output); // Shows the controller output
        }
        
        // 4. Display the complete log line once
        ESP_LOGI(TAG, "%s", log_buffer);

        // Delay to maintain an approximate sampling rate (T_s ≈ 10ms)
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }


    ESP_ERROR_CHECK(adc_oneshot_del_unit(adc1_handle));
    for (int i = 0; i < NUM_FINGERS; i++) {
        if (do_calibration[i]) {
            example_adc_calibration_deinit(adc1_cali_handles[i]);
        }
    }
}

/*---------------------------------------------------------------
 * ADC Calibration Functions
 *---------------------------------------------------------------*/
static bool example_adc_calibration_init(adc_unit_t unit, adc_channel_t channel, adc_atten_t atten, adc_cali_handle_t *out_handle)
{
    // Function to initialize ADC calibration using eFuse data if available.
    adc_cali_handle_t handle = NULL;
    esp_err_t ret = ESP_FAIL;
    bool calibrated = false;

#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    if (!calibrated) {
        ESP_LOGI(TAG, "Calibration Scheme: %s (Ch %d)", "Curve Fitting", channel);
        adc_cali_curve_fitting_config_t cali_config = {
            .unit_id = unit,
            .chan = channel,
            .atten = atten,
            .bitwidth = ADC_BITWIDTH_DEFAULT,
        };
        ret = adc_cali_create_scheme_curve_fitting(&cali_config, &handle);
        if (ret == ESP_OK) {
            calibrated = true;
        }
    }
#endif

    *out_handle = handle;

    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Calibration Success for Ch %d", channel);
    } else if (ret == ESP_ERR_NOT_SUPPORTED || !calibrated) {
        ESP_LOGW(TAG, "Ch %d: eFuse not burnt, skipping software calibration", channel);
    } else {
        ESP_LOGE(TAG, "Ch %d: Invalid arg or no memory", channel);
    }

    return calibrated;
}

static void example_adc_calibration_deinit(adc_cali_handle_t handle)
{
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    ESP_LOGI(TAG, "Deregistering calibration scheme");
    ESP_ERROR_CHECK(adc_cali_delete_scheme_curve_fitting(handle));
#endif
}