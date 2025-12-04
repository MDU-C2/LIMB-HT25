#include "potentiometer.h"

#include <stdint.h>
#include <sys/param.h>

// Performs a linear interpolation of x from the range [x0, x1] onto [y0, y1].
static float lerp_from_range(float x, float x0, float x1, float y0, float y1) {
  float x_range = x1 - x0;
  float y_range = y1 - y0;
  return y0 + ((x - x0) * y_range / x_range);
}

float potentiometer_adc_to_degrees(Potentiometer potentiometer,
                                   uint16_t adc_value) {
  const uint16_t max_adc_value = (1U << 12U) - 1U;

  // Don't allow values larger than the max possible ADC value.
  adc_value = MIN(adc_value, max_adc_value);

  return lerp_from_range((float)adc_value, 0, max_adc_value,
                         potentiometer.min_degree, potentiometer.max_degree);
}
