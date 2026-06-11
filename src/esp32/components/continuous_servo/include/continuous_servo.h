#pragma once
#include "esp_err.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"

// Determines if low angles should generate low or high pulse widths, and vice
// versa for high angles.
typedef enum {
  CONTINUOUS_SERVO_DIR_NORMAL,
  CONTINUOUS_SERVO_DIR_REVERSE,
} ContinuousServoDirection;

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

  // The maximum allowed speed of the servo in the direction that increases the
  // potentiometer angle.
  AngularVelocity max_speed_increasing_angle;
  // The maximum allowed speed of the servo in the direction that increases the
  // potentiometer angle.
  AngularVelocity max_speed_decreasing_angle;
  // The maximum allowed acceleration of the servo.
  AngularAcceleration max_accel;

  // The ratio between the servo's rotation and the joint's rotation
  // (i.e. 15 means the servo has to complete 15 turns for the joint to complete
  // 1 turn).
  float gear_ratio;

  // The configuration/calibration of the servo's potentiometer.
  Potentiometer potentiometer;

  // If the angles should be reversed.
  ContinuousServoDirection direction;
  // Human-readable name for debugging
  const char* name;
} ContinuousServoConfig;

typedef ledc_channel_t ContinuousServoHandle;

// Initialize servos using provided configurations.
esp_err_t continuous_servo_init(const ContinuousServoConfig* servo_config,
                                uint16_t latest_potentiometer_adc_value,
                                ContinuousServoHandle* out_handle);

// This function is meant to be called periodically. It determines how far it
// should move the servo based on the distance to the target angle and the time
// remaining until the next call to continuous_servo_update.
bool continuous_servo_update(ContinuousServoHandle handle,
                             uint16_t ms_until_next_period,
                             uint16_t potentiometer_value);

// Set target angular velocity (degrees per second)
void continuous_servo_set_target_velocity(ContinuousServoHandle handle,
                                          AngularVelocity target_velocity);

// Sets the target angle that `continuous_servo_update` aims for.
void continuous_servo_set_target_angle(ContinuousServoHandle handle,
                                       JointAngle target_angle);

void continuous_servo_set_estop(ContinuousServoHandle handle, bool active);

PotentiometerAngle continuous_servo_get_current_angle(
    ContinuousServoHandle handle);

PotentiometerAngle continuous_servo_get_target_angle(
    ContinuousServoHandle handle);

AngularVelocity continuous_servo_get_current_velocity(
    ContinuousServoHandle handle);

// Apply the provided angular velocity.
void continuous_servo_apply_velocity(ContinuousServoHandle handle,
                                     AngularVelocity velocity);

// Apply the velocity represented by the provided pulse width.
void continuous_servo_apply_pulse_width_as_velocity(
    ContinuousServoHandle handle, uint16_t pulse_width);
