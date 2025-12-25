#pragma once

#include <stdint.h>

#include "limb_utils.h"
#include "potentiometer.h"

typedef struct {
  PotentiometerAngle current_angle;
  PotentiometerAngle target_angle;
  PotentiometerAngle deadband;
  AngularVelocity current_velocity;
  AngularAcceleration max_acceleration;
  AngularVelocity max_velocity;
  // The amount of time between updates to the motor state.
  uint32_t timestep_ms;
} MotorRampingArgs;

// Calculates the velocity the motor should use to achieve a trapezoidal ramping
// effect.
AngularVelocity motor_ramping_trapezoidal(const MotorRampingArgs *args);
