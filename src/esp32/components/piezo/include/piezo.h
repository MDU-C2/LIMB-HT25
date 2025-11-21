#pragma once

#include <stdint.h>
#include "esp_err.h"
#include "hal/adc_types.h"

/**
 * @brief Piezo sensor configuration structure
 */
typedef struct {
    uint8_t gpio_pin;           // GPIO pin number (e.g., GPIO1)
    adc_channel_t adc_channel;  // ADC channel (e.g., ADC_CHANNEL_1)
    adc_atten_t adc_atten;      // ADC attenuation (ADC_ATTEN_DB_12 recommended for 0-3.3V)
    adc_bitwidth_t adc_bitwidth; // ADC bit width (ADC_BITWIDTH_DEFAULT)
} piezo_config_t;

/**
 * @brief Default piezo configuration for GPIO1 (ADC_CHANNEL_1)
 */
#define PIEZO_CONFIG_DEFAULT() (piezo_config_t) { \
    .gpio_pin = 1, \
    .adc_channel = ADC_CHANNEL_1, \
    .adc_atten = ADC_ATTEN_DB_12, \
    .adc_bitwidth = ADC_BITWIDTH_DEFAULT, \
}

/**
 * @brief Initialize the piezo sensor ADC
 *
 * @param config Configuration structure. If NULL, uses default configuration.
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t piezo_init(const piezo_config_t *config);

/**
 * @brief Read raw ADC value from piezo sensor
 *
 * @param raw_value Pointer to store the raw ADC value
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t piezo_read_raw(int *raw_value);

/**
 * @brief Deinitialize the piezo sensor ADC
 *
 * @return esp_err_t ESP_OK on success, error code otherwise
 */
esp_err_t piezo_deinit(void);

