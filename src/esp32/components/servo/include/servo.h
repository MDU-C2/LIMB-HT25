#pragma once
#include "esp_err.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
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
  ledc_channel_t pwm_channel;
  // LEDC timer used by the servo.
  ledc_timer_t pwm_timer;

  // The pulse width that stops the servo.
  uint16_t motionless_pw;

  // The min and max speeds that the servo can actuate with.

  // The maximum capable angular velocity.
  AngularVelocity max_capable_angular_velocity;
  // The pulse width offset from motionless_pw that corresponds to the
  // max_capable_angular_velocity.
  uint16_t max_capable_angular_velocity_pw_offset;

  // The maximum allowed velocity of the servo.
  AngularVelocity max_velocity_positive;
  // The maximum allowed velocity of the servo.
  AngularVelocity max_velocity_negative;
  // The maximum allowed acceleration of the servo.
  AngularAcceleration max_accel;

  // The ratio between the servo's rotation and the joint's rotation
  // (i.e. 15 means the servo has to complete 15 turns for the joint to complete
  // 1 turn).
  float gear_ratio;

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
                     uint16_t latest_potentiometer_adc_value,
                     ServoHandle *out_handle);

// This function is meant to be called periodically. It determines how far it
// should move the servo based on the distance to the target angle and the time
// remaining until the next call to servo_update.
bool servo_update(ServoHandle handle, uint16_t ms_until_next_period,
                  uint16_t potentiometer_value);

// Set target angular velocity (degrees per second)
void servo_set_target_velocity(ServoHandle handle, AngularVelocity target_velocity);

// Sets the target angle that `servo_update` aims for.
void servo_set_target_angle(ServoHandle handle, JointAngle target_angle);

void servo_set_estop(ServoHandle handle, bool active);

PotentiometerAngle servo_get_current_angle(ServoHandle handle);

PotentiometerAngle servo_get_target_angle(ServoHandle handle);

AngularVelocity servo_get_current_velocity(ServoHandle handle);

// Apply the provided angular velocity.
void servo_apply_velocity(ServoHandle handle, AngularVelocity velocity);

// Apply the velocity represented by the provided pulse width.
void servo_apply_pulse_width_as_velocity(ServoHandle handle,
                                         uint16_t pulse_width);
