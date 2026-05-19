#include "limb_utils.h"

#include <endian.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "include/limb_utils.h"

float deserialize_float(uint8_t* float_buf, FromEndian from_endian) {
  // To extract the float from the byte buffer while avoiding strict aliasing
  // issues, we use memcpy. This assumes that float_buf is at least 4 bytes
  // long.
  float ret = 0;
  memcpy(&ret, float_buf, sizeof(ret));
  if (from_endian == kFromBigEndian) {
    return beftoh(ret);
  }
  return leftoh(ret);
}

float bswapf(float float_val) {
  // To perform the byte swap in a standard-compliant way, we memcpy the bytes
  // to an integer before calling bswap32, then memcpy the bytes back to the
  // float variable.
  uint32_t int_representation = 0;
  static_assert(sizeof(float_val) == sizeof(int_representation),
                "float is not 32 bits");
  memcpy(&int_representation, &float_val, sizeof(int_representation));
  int_representation = bswap32(int_representation);
  memcpy(&float_val, &int_representation, sizeof(float_val));
  return float_val;
}

uint16_t limb_average16(const uint16_t* values, size_t n) {
  uint32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += values[i];
  }

  return sum / n;
}

uint16_t moving_average16(uint16_t start_value, const uint16_t* values,
                          uint16_t values_len) {
  uint16_t value = start_value;
  for (uint16_t i = 0; i < values_len; ++i) {
    value = (value + values[i]) / 2;
  }
  return value;
}
