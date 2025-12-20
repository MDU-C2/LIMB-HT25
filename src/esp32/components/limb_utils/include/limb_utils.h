#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/param.h>

// Clamps a value to a range.
#define LIMB_CLAMP(x, x_min, x_max) (MIN(MAX((x), (x_min)), (x_max)))

// Performs a linear interpolation of x from the range [x_min, x_max] onto
// [y_min, y_max].
#define LIMB_LERP_FROM_RANGE(x, x_min, x_max, y_min, y_max) \
  ((y_min) + (((x) - (x_min)) * ((y_max) - (y_min)) / ((x_max) - (x_min))))

// Returns the length of an array.
#define LIMB_ARR_LEN(arr) (sizeof(arr) / sizeof(*(arr)))

// Calculates the average value from an array of uint16_ts.
uint16_t limb_average16(const uint16_t *values, size_t n);
