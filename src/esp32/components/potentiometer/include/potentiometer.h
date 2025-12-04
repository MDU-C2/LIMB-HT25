#pragma once

#include <stdint.h>

typedef struct {
  float min_degree;
  float max_degree;
  uint16_t adc_bitwidth;
} Potentiometer;

// Converts the provided ADC value of the provided bitwidth to the
// potentiometer's corresponding degree.
float potentiometer_adc_to_degrees(Potentiometer potentiometer,
                                   uint16_t adc_value);
