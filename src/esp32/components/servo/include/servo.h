#pragma once
#include "esp_err.h"
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
  // If the angles should be reversed.
  ServoDirection direction;
  // Human-readable name for debugging
  const char *name;
} ServoConfig;

// Initialize servos using provided configurations.
esp_err_t servo_init(const ServoConfig *config);

// Actuate servo to the specified degree.
void servo_move_to_degree(const ServoConfig *servo, PotentiometerAngle deg);
