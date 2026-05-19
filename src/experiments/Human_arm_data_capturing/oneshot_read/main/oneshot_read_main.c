/*
 * SPDX-FileCopyrightText: 2022-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"

const static char* TAG = "EXAMPLE";

/*---------------------------------------------------------------
        ADC General Macros
---------------------------------------------------------------*/

#define EXAMPLE_ADC1_CHAN0 ADC_CHANNEL_2

#define EXAMPLE_ADC_ATTEN ADC_ATTEN_DB_2_5

static int adc_raw[2][10];
static int voltage[2][10];

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
  ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config1, &adc1_handle));

  //-------------ADC1 Config---------------//
  adc_oneshot_chan_cfg_t config = {
      .atten = EXAMPLE_ADC_ATTEN,
      .bitwidth = ADC_BITWIDTH_DEFAULT,
  };
  ESP_ERROR_CHECK(
      adc_oneshot_config_channel(adc1_handle, EXAMPLE_ADC1_CHAN0, &config));
  // ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, EXAMPLE_ADC1_CHAN1,
  // &config));

  //-------------ADC1 Calibration Init---------------//
  adc_cali_handle_t adc1_cali_chan0_handle = NULL;
  bool do_calibration1_chan0 =
      example_adc_calibration_init(ADC_UNIT_1, EXAMPLE_ADC1_CHAN0,
                                   EXAMPLE_ADC_ATTEN, &adc1_cali_chan0_handle);

  while (1) {
    ESP_ERROR_CHECK(
        adc_oneshot_read(adc1_handle, EXAMPLE_ADC1_CHAN0, &adc_raw[0][0]));
    ESP_LOGI(TAG, "ADC%d Channel[%d] Raw Data: %d", ADC_UNIT_1 + 1,
             EXAMPLE_ADC1_CHAN0, adc_raw[0][0]);

    float volt_V = 0;

    if (do_calibration1_chan0) {
      ESP_ERROR_CHECK(adc_cali_raw_to_voltage(adc1_cali_chan0_handle,
                                              adc_raw[0][0], &voltage[0][0]));

      volt_V = (float)voltage[0][0] / 1000.0;

      ESP_LOGI(TAG, "ADC%d Channel[%d] Cali Voltage: %f V", ADC_UNIT_1 + 1,
               EXAMPLE_ADC1_CHAN0, volt_V);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }

  // Tear Down
  ESP_ERROR_CHECK(adc_oneshot_del_unit(adc1_handle));
  if (do_calibration1_chan0) {
    example_adc_calibration_deinit(adc1_cali_chan0_handle);
  }
}

/*---------------------------------------------------------------
        ADC Calibration
---------------------------------------------------------------*/
static bool example_adc_calibration_init(adc_unit_t unit, adc_channel_t channel,
                                         adc_atten_t atten,
                                         adc_cali_handle_t* out_handle) {
  adc_cali_handle_t handle = NULL;
  esp_err_t ret = ESP_FAIL;
  bool calibrated = false;

#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
  if (!calibrated) {
    ESP_LOGI(TAG, "calibration scheme version is %s", "Curve Fitting");
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
    ESP_LOGI(TAG, "Calibration Success");
  } else if (ret == ESP_ERR_NOT_SUPPORTED || !calibrated) {
    ESP_LOGW(TAG, "eFuse not burnt, skip software calibration");
  } else {
    ESP_LOGE(TAG, "Invalid arg or no memory");
  }

  return calibrated;
}

static void example_adc_calibration_deinit(adc_cali_handle_t handle) {
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
  ESP_LOGI(TAG, "deregister %s calibration scheme", "Curve Fitting");
  ESP_ERROR_CHECK(adc_cali_delete_scheme_curve_fitting(handle));
#endif
}
