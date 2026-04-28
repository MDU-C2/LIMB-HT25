#include "limb_utils.h"

#include <stddef.h>
#include <stdint.h>

uint16_t limb_average16(const uint16_t *values, size_t n) {
  uint32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += values[i];
  }

  return sum / n;
}

uint16_t moving_average16(uint16_t start_value, const uint16_t *values,
                          uint16_t values_len) {
  uint16_t value = start_value;
  for (uint16_t i = 0; i < values_len; ++i) {
    value = (value + values[i]) / 2;
  }
  return value;
}
