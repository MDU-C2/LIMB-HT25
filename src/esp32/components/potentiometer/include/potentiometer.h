#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "esp_adc/adc_oneshot.h"
#include "hal/adc_types.h"

/**
 * @brief Potentiometer configuration structure
 */
typedef struct {
    uint8_t gpio_pin;           // GPIO pin number (e.g., GPIO0)
    adc_channel_t adc_channel;  // ADC channel (e.g., ADC_CHANNEL_0)
    adc_atten_t adc_atten;      // ADC attenuation (ADC_ATTEN_DB_0 to ADC_ATTEN_DB_12)
    adc_bitwidth_t adc_bitwidth; // ADC bit width (ADC_BITWIDTH_DEFAULT, etc.)
} potentiometer_config_t;

/**
 * @brief Default potentiometer configuration for GPIO0
 */
#define POTENTIOMETER_CONFIG_DEFAULT() (potentiometer_config_t) { \
    .gpio_pin = 0, \
    .adc_channel = ADC_CHANNEL_0, \
    .adc_atten = ADC_ATTEN_DB_12, \
    .adc_bitwidth = ADC_BITWIDTH_DEFAULT, \
}

/**
 * @brief Initialize the potentiometer ADC
 *
 * @param config Configuration structure. If NULL, uses default configuration.
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t potentiometer_init(const potentiometer_config_t *config);

/**
 * @brief Read raw ADC value from potentiometer
 *
 * @param raw_value Pointer to store the raw ADC value
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t potentiometer_read_raw(int *raw_value);

/**
 * @brief Read voltage in millivolts from potentiometer
 *
 * @param voltage_mv Pointer to store the voltage in millivolts
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t potentiometer_read_voltage(int *voltage_mv);

/**
 * @brief Read normalized value (0-1000) from potentiometer
 *
 * @param normalized_value Pointer to store the normalized value (0-1000)
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t potentiometer_read_normalized(uint16_t *normalized_value);

/**
 * @brief Deinitialize the potentiometer ADC
 *
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t potentiometer_deinit(void);

