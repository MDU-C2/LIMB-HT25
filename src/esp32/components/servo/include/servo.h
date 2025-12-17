#pragma once
#include "esp_err.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "potentiometer.h"

// Determines if low angles should generate low or high pulse widths, and vice
// versa for high angles.
typedef enum {
  SERVO_DIR_NORMAL,
  SERVO_DIR_REVERSE,
} ServoDirection;

// Configuration options for initializing servos. Also serves as the handle for
// the servo once configured.
typedef struct {
  // GPIO pin used for the PWM signal for the servo.
  int gpio_pin;
  // LEDC channel used by the servo.
  ledc_channel_t ledc_channel;
  // Minimum angle in degrees.
  PotentiometerAngle min_angle;
  // Maximum angle in degrees.
  PotentiometerAngle max_angle;
  // The pulse width that corresponds to the minimum angle.
  uint32_t min_pulse_us;
  // The pulse width that corresponds to the maximum angle.
  uint32_t max_pulse_us;
  // The angle the servo should be set to right after initialization.
  PotentiometerAngle initial_angle;
  // The maximum allowed velocity of the servo.
  float max_velocity_dps;
  // The maximum allowed acceleration of the servo.
  float max_accel_dps2;

  // The ADC channel used for the servo's potentiometer.
  adc_channel_t pot_adc_channel;
  // The configuration/calibration of the servo's potentiometer.
  Potentiometer potentiometer;

  // If the angles should be reversed.
  ServoDirection direction;
  // Human-readable name for debugging
  const char *name;
} ServoConfig;

typedef ledc_channel_t ServoHandle;

// Initialize servos using provided configurations.
esp_err_t servo_init(const ServoConfig *servo_config,
                     const uint16_t *latest_potentiometer_values,
                     uint16_t latest_potentiometer_values_len,
                     ServoHandle *out_handle);

// This function is meant to be called periodically. It determines how far it
// should move the servo based on the distance to the target angle and the time
// taken since the last call to servo_update, determined by dt_seconds passed by
// the caller.
void servo_update(ServoHandle handle, float dt_seconds,
                  const uint16_t *potentiometer_values,
                  uint16_t potentiometer_values_len);

// Actuate servo to the specified degree.
void servo_move_to_degree(ServoHandle handle, PotentiometerAngle deg);

// Move the servo to the specified pulse width. The value is clamped to the
// servo's min and max pulse widths before being written.
void servo_move_to_pulse_width(ServoHandle handle, uint16_t pulse_width);
