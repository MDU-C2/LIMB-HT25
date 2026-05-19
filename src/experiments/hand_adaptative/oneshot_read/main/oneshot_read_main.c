/*
 * SPDX-FileCopyrightText: 2022-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "can_driver.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"
#define ID_SENSOR_DATA 0x020

const static char* TAG = "ADC_HAND";

/*---------------------------------------------------------------
 * ADC General Macros & Definitions
 *---------------------------------------------------------------*/
// Defines the number of FSR sensors used (corrected to 4 based on array
// initialization).
#define NUM_FINGERS 5
#define EXAMPLE_ADC_ATTEN \
  ADC_ATTEN_DB_12  // ADC attenuation (12dB = ~0V to 3.3V range)

// >>> CONTROL VARIABLES BASED ON VOLTS <<<
// >>> CONTROL VARIABLES BASED ON MILLIVOLTS (mV) <<<
// Diferentes setpoint por dedo y que sean adaptativas, diferentes formas de
// vaso, diferentes materiales
#define TARGET_PRESSURE_MV 3  // Setpoint en Milivoltios (1.5V)
#define ALPHA_LPF 0.1         // Coeficiente del Filtro Pasa-Bajas

// Control Gains (Initial test values)
// The controller output (angle change) is calculated based on the error in
// VOLTS.
#define KP_GAIN 0.01   // Proportional Gain (Output_Angle / mV)
#define KD_GAIN 0.001  // Derivative Gain (Output_Angle / (mV/second))

// Storage structures for ADC reading and calibration data.
static int adc_raw[NUM_FINGERS];
static int voltage_mV[NUM_FINGERS];  // Lecturas calibradas en mV
static adc_cali_handle_t adc1_cali_handles[NUM_FINGERS];
static bool do_calibration[NUM_FINGERS];

// Array of ADC channels to read.
const adc_channel_t adc_channels[NUM_FINGERS] = {
    ADC_CHANNEL_0, ADC_CHANNEL_1, ADC_CHANNEL_2, ADC_CHANNEL_3, ADC_CHANNEL_4};

// 1. Almacena el ángulo actual que se envía al servo (0-180 grados).
static int current_angle[NUM_FINGERS] = {90, 90, 90, 90, 90};

// 2. Estado de Control (todos en mV)
static float previous_error[NUM_FINGERS] = {0.0};
static float filtered_voltage_mV[NUM_FINGERS] = {0.0};

// Function prototypes for ADC calibration.
static bool example_adc_calibration_init(adc_unit_t unit, adc_channel_t channel,
                                         adc_atten_t atten,
                                         adc_cali_handle_t* out_handle);
static void example_adc_calibration_deinit(adc_cali_handle_t handle);

void app_main(void) {
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
    ESP_ERROR_CHECK(
        adc_oneshot_config_channel(adc1_handle, adc_channels[i], &config));

    // Initialize Calibration for the channel
    do_calibration[i] = example_adc_calibration_init(
        ADC_UNIT_1, adc_channels[i], EXAMPLE_ADC_ATTEN, &adc1_cali_handles[i]);
  }

  //--------CAN init
  // Inicialización del CAN: TX=5, RX=4, 125 kbps
  can_init(8, 9, 125000);
  uint8_t tx_data[5];
  //----------------

  // char log_buffer[512]; // Buffer for the complete log line

  static int64_t last_time_us = 0;
  float control_error = 0.0;
  float control_output = 0.0;
  float T_s_real = 0.01;

  //------------------------- Fast Reading Loop -------------------------//
  while (1) {
    // int len = 0; // Reset log buffer index at the start of each cycle
    // Cálculo de Tiempo de Muestreo (T_s_real)
    int64_t current_time_us = esp_timer_get_time();
    if (last_time_us != 0) {
      T_s_real = (float)(current_time_us - last_time_us) / 1000000.0;
      if (T_s_real <= 0 || T_s_real > 0.1) T_s_real = 0.01;
    }
    last_time_us = current_time_us;

    for (int i = 0; i < NUM_FINGERS; i++) {
      // 1. Read and Convert to Calibrated Voltage (mV)
      ESP_ERROR_CHECK(
          adc_oneshot_read(adc1_handle, adc_channels[i], &adc_raw[i]));
      if (do_calibration[i]) {
        ESP_ERROR_CHECK(adc_cali_raw_to_voltage(adc1_cali_handles[i],
                                                adc_raw[i], &voltage_mV[i]));
      }

      // --- LÓGICA DE CONTROL PD (Solo para un Dedo ) ---
      if (i == 4) {
        // 1. Aplicar Filtro Pasa-Bajas (LPF)
        if (filtered_voltage_mV[i] == 0.0) {
          filtered_voltage_mV[i] = (float)voltage_mV[i];
        } else {
          filtered_voltage_mV[i] = ALPHA_LPF * (float)voltage_mV[i] +
                                   (1.0 - ALPHA_LPF) * filtered_voltage_mV[i];
        }

        // 2. Calcular Error Actual e(k) en MILLIVOLTS (mV)
        control_error = TARGET_PRESSURE_MV - filtered_voltage_mV[i];

        // 3. Calcular Término Derivativo (Rate of change of error)
        float derivative_term = (control_error - previous_error[i]) / T_s_real;

        // 4. Calcular Salida de Control (Delta de Ángulo)
        float proportional_term = KP_GAIN * control_error;
        float derivative_output = KD_GAIN * derivative_term;

        control_output = proportional_term + derivative_output;

        // 5. Actualizar Ángulo
        current_angle[i] += (int)control_output;

        // 6. Clamping (Restricción del Ángulo entre 0° y 180°)
        if (current_angle[i] < 0) current_angle[i] = 0;
        if (current_angle[i] > 180) current_angle[i] = 180;

        // 7. Actualizar el estado para el próximo ciclo
        previous_error[i] = control_error;

        // LOG: Mostrar el control para el dedo 1
        // ESP_LOGI(TAG, "D1 Control | V_mV:%d | FILT_mV:%.0f | ERR_mV:%.0f |
        // Ts:%.4fs | Delta_A:%.1f° | New_A:%d°",
        //     voltage_mV[i],
        //     filtered_voltage_mV[i],
        //     control_error,
        //     T_s_real,
        //     control_output,
        //     current_angle[i]);
      } else {
        // Para los otros dedos, mantenemos el ángulo inicial (90°)
        current_angle[i] = 90;
      }

      // --------------------------------

      // 3. Empaquetamiento de ÁNGULOS (uint8_t: 1 byte)
      tx_data[i] = (uint8_t)current_angle[i];
    }

    ESP_LOGI(TAG, "D1: %d, D2: %d, D3: %d, D4: %d, D5: %d", voltage_mV[0],
             voltage_mV[1], voltage_mV[2], voltage_mV[3], voltage_mV[4]);
    // esp_err_t err1 = can_send(ID_SENSOR_DATA, tx_data, 5);
    // if (err1 == ESP_OK) {
    //     ESP_LOGI(TAG, "CAN TX OK [0x%03X] (DLC=5): A1:%d A2:%d A3:%d A4:%d
    //     A5:%d",
    //         ID_SENSOR_DATA,
    //         tx_data[0], tx_data[1], tx_data[2], tx_data[3], tx_data[4]);
    // } else {
    //     ESP_LOGE(TAG, "Error CAN TX: 0x%X", err1);
    // }

    // Delay to maintain an approximate sampling rate (T_s ≈ 10ms)
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

/*---------------------------------------------------------------
 * ADC Calibration Functions
 *---------------------------------------------------------------*/
static bool example_adc_calibration_init(adc_unit_t unit, adc_channel_t channel,
                                         adc_atten_t atten,
                                         adc_cali_handle_t* out_handle) {
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
    ESP_LOGW(TAG, "Ch %d: eFuse not burnt, skipping software calibration",
             channel);
  } else {
    ESP_LOGE(TAG, "Ch %d: Invalid arg or no memory", channel);
  }

  return calibrated;
}

static void example_adc_calibration_deinit(adc_cali_handle_t handle) {
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
  ESP_LOGI(TAG, "Deregistering calibration scheme");
  ESP_ERROR_CHECK(adc_cali_delete_scheme_curve_fitting(handle));
#endif
}