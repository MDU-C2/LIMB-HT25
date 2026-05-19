#include "motor_ramping.h"

#include <math.h>
#include <stdbool.h>

#include "limb_utils.h"
#include "potentiometer.h"

AngularVelocity motor_ramping_trapezoidal(const MotorRampingArgs* args) {
  const PotentiometerAngle distance_to_target = {args->target_angle.degree -
                                                 args->current_angle.degree};
  const PotentiometerAngle abs_distance_to_target = {
      fabsf(distance_to_target.degree)};

  // The maximum velocity allowed during a timestep.
  const AngularVelocity abs_max_velocity_delta = {
      args->max_acceleration.dps2 * (float)args->timestep_ms / 1000.F};

  // If we're within the deadband and the current velocity is within the allowed
  // acceleration limit, we want to stop entirely.
  const bool within_deadband =
      abs_distance_to_target.degree < args->deadband.degree;
  const bool slow_enough =
      fabsf(args->current_velocity.dps) <= abs_max_velocity_delta.dps;

  if (within_deadband && slow_enough) {
    return (AngularVelocity){0};
  }

  // Braking: max velocity from remaining distance (trapezoidal profile)
  // v_max^2 = 2 * a * d  =>  v_max = sqrt(2 * a * d)
  const AngularVelocity vmax_from_distance = {sqrtf(
      2.0F * args->max_acceleration.dps2 * abs_distance_to_target.degree)};
  const AngularVelocity abs_target_velocity_negative = {
      fminf(args->max_velocity_negative.dps, vmax_from_distance.dps)};
  const AngularVelocity abs_target_velocity_positive = {
      fminf(args->max_velocity_positive.dps, vmax_from_distance.dps)};
  const AngularVelocity target_velocity = {
      distance_to_target.degree < 0.F ? -abs_target_velocity_negative.dps
                                      : abs_target_velocity_positive.dps};

  // Velocity ramping
  const AngularVelocity velocity_delta = {target_velocity.dps -
                                          args->current_velocity.dps};
  const AngularVelocity new_velocity = {args->current_velocity.dps +
                                        LIMB_CLAMP(velocity_delta.dps,
                                                   -abs_max_velocity_delta.dps,
                                                   abs_max_velocity_delta.dps)};
  return new_velocity;
}
