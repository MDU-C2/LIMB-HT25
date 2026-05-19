#ifndef ADC_SERVICE_H
#define ADC_SERVICE_H

#include "esp_err.h"
#include "hal/adc_types.h"

/* --- Configuration Settings --- */
#define NUM_FINGERS 5        // Number of FSR sensors used (one per finger)
#define SAMPLES_PER_WIND 10  // Number of samples per window for statistics
#define SAMPLE_PERIOD 50  // Sample interval (Total window: 50ms * 10 = 500ms)

/**
 * @brief Structure to store statistical analysis of pressure data.
 * Used to trigger transitions between Stability (E2) and Reaction (E3) states.
 */
typedef struct {
  float mean;      // Mean value (average pressure) of adc values in mV
  float variance;  // Variance (detects slips, noise, or sudden movements)
} wstats_t;

/**
 * @brief Initializes the ADC engine using the DMA-based adc_manager component.
 * Configures 5 physical channels for the robotic hand's FSR sensors.
 * * @return ESP_OK if initialized successfully, ESP_FAIL otherwise.
 */
esp_err_t init_adc_service(void);

/**
 * @brief Executes a 0.5-second sampling cycle.
 * Blocks the task during sampling to calculate mean and variance.
 * * @return estadisticas_t Calculated statistical data for the window.
 */
wstats_t get_window_stats(void);

/**
 * @brief Gets a single, instantaneous average reading from all 5 sensors.
 * Primarily used for high-speed PD control loops in state E3.
 * * @return float Current average pressure in mV.
 */
float get_instant_pressure(void);

#endif  // ADC_SERVICE_H