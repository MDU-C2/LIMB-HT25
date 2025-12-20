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
