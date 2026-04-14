# Motor Ramping Component

This component provides functionality to generate smooth, trapezoidal
motion profiles (ramping) for motors. This results in an acceleration
towards a max velocity and deceleration towards a standstill rather than
instantly moving at the max velocity and then instantly stopping.

## Usage

Call `motor_ramping_trapezoidal` periodically with the current state
and target to get the updated required velocity to achieve a trapezoidal
motion profile.

```c
#include "motor_ramping.h"

// Assume you have some function to set your motor's velocity.
void motor_set_velocity(AngularVelocity vel);

// Suppose you have a motor update loop that gets called every N ms.
void update_motor(PotentiometerAngle current_pos, PotentiometerAngle target_pos,
                  AngularVelocity current_vel, uint32_t delta_time_ms) {
  MotorRampingArgs args = {
      .current_angle = current_pos,
      .target_angle = target_pos,
      .deadband = (PotentiometerAngle){2.0F},
      .current_velocity = current_vel,
      .max_acceleration = (AngularAcceleration){8.0F},
      // The max speed that the angle can decrease at.
      .max_speed_decreasing_angle = (AngularVelocity){2.0F},
      // The max speed that the angle can increase at.
      .max_speed_increasing_angle = (AngularVelocity){6.0F},
      .timestep_ms = delta_time_ms,
  };

  // Calculate the velocity at this point in time that overall results in a
  // trapezoidal motion profile.
  AngularVelocity new_velocity = motor_ramping_trapezoidal(&args);

  // Apply your new velocity until the next periodic call to this update
  // function.
  motor_set_velocity(new_velocity);
}
```
