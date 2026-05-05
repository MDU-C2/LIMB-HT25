#pragma once

#ifndef htobe16
#include <endian.h>
#endif
#include <stddef.h>
#include <stdint.h>
#include <sys/param.h>

typedef enum {
  kFromLittleEndian,
  kFromBigEndian,
} FromEndian;

typedef struct {
  float dps;
} AngularVelocity;

typedef struct {
  float dps2;
} AngularAcceleration;

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

// Calculates the moving average starting with `start_value` and then following
// the sequence `values`.
uint16_t moving_average16(uint16_t start_value, const uint16_t *values,
                          uint16_t values_len);

// Returns a float deserialized from the buffer pointed to by float_buf.
// The serialized float in float_buf is expected to have the endianness
// declared by the provided from_endian argument.
float deserialize_float(uint8_t* float_buf, FromEndian from_endian);

#if !defined _BYTE_ORDER
#error "_BYTE_ORDER isn't available, can't determine endianness"
#elif !defined _LITTLE_ENDIAN
#error "_LITTLE_ENDIAN isn't available, can't determine endianness"
#endif

// Returns float_val with its byte order swapped.
float bswapf(float float_val);

#if _BYTE_ORDER == _LITTLE_ENDIAN
#define htolef(x) ((float)(x))
#define leftoh(x) ((float)(x))
#define htobef(x) bswapf(x)
#define beftoh(x) bswapf(x)
#else
#define htolef(x) bswapf((x))
#define leftoh(x) bswapf((x))
#define htobef(x) ((float)(x))
#define beftoh(x) ((float)(x))
#endif
