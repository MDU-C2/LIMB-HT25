#pragma once

#include <stdint.h>

#include "limb_utils.h"
#include "potentiometer.h"

typedef struct {
  PotentiometerAngle current_angle;
  PotentiometerAngle target_angle;
  // The deadband extends +-(deadband / 2) degrees from target_angle.
  PotentiometerAngle deadband;
  AngularVelocity current_velocity;
  AngularAcceleration max_acceleration;
  // The max speed that the motor should be allowed to move at in the direction
  // that decreases the potentiometer angle.
  AngularVelocity max_speed_decreasing_angle;
  // The max speed that the motor should be allowed to move at in the direction
  // that increases the potentiometer angle.
  AngularVelocity max_speed_increasing_angle;
  // The amount of time between updates to the motor state.
  uint32_t timestep_ms;
} MotorRampingArgs;

// Calculates the velocity the motor should use to achieve a trapezoidal ramping
// effect.
AngularVelocity motor_ramping_trapezoidal(const MotorRampingArgs* args);
