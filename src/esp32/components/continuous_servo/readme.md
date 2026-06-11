# Continuous servo component targeting a modded JX Servo PDI-HV2060MG

This component allows for controlling a [JX Servo
PDI-HV2060MG](http://www.jx-servo.com/en/Product/STANDARD/SD/544.html)
that has been modded to be continuous.

## Servo requirements

The servo needs to have been modified to be continuous for this component to work.
The process of modifying it is described in [the documentation](../../../../docs/servo_modification.md)
along with the effect the modification has on its behavior and the consequences that arise as a result.
Once the servo is continuous, this component provides the ability to use velocities to control the servo,
allowing for it to also be controlled using a motor update function that gets called periodically.

> [!caution]
> After modding the motor, if a pulse width has been applied, the servo will keep on rotating until the pulse width has been reset.
> This means that if a pulse width has been set, and the `continuous_servo_update` isn't called properly to update the
> pulse width based on the target angle and potentiometer values, the motor *will not* stop! In worst case this
> could lead to parts of the robot arm breaking or human injury. As such it is *very important* that any program
> using the servo component actually manages to call `continuous_servo_update` periodically.

> [!note]
> The relation between the pulse width and the motor's velocity is approximate and does not take load into account.
> For example, the same pulse width will provide a slower velocity if the motor has to fight against gravity by
> raising the arm up from the ground compared to if it is assisted by gravity when moving the arm down. This means
> that the actual velocity of the arm might be different from what this component expects the velocity to be,
> resulting in the arm potentially overshooting its target.

## Usage

The basic usage is as follows:
```c
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"
#include "continuous_servo.h"
#include "soc/gpio_num.h"

// Create a config for the servo. Check servo.h for more details regarding the
// what the members represent.
static const ContinuousServoConfig kServoConfig = {
    .gpio_pin = GPIO_NUM_0,
    .pwm_timer = LEDC_TIMER_0,
    .pwm_channel = LEDC_CHANNEL_0,
    .name = "Servo 1",
    .direction = CONTINUOUS_SERVO_DIR_NORMAL,
    .motionless_pw = 1500,
    .max_capable_angular_velocity = {400},
    .max_capable_angular_velocity_pw_offset = 150,
    .gear_ratio = 15.F,
    .max_speed_decreasing_angle = {8.F},
    .max_speed_increasing_angle = {16.F},
    .max_accel = {8.F},
    .potentiometer =
        (Potentiometer){
            .range_of_motion = {285.F},
            .min_adc_value = 20,
            .max_adc_value = 3087,
            .min_potentiometer_angle = {170},
            .max_potentiometer_angle = {200},
            .min_potentiometer_angle_as_joint_angle = {0.F},
            .joint_angle_to_potentiometer_angle_ratio = 18.F / 15.F,
            .is_reversed = false,
        },
};

void app_main(void) {
  // How you get the ADC value is up to you.
  uint16_t adc_value = get_adc_value_for_potentiometer();
  ContinuousServoHandle servo_handle = {0};
  esp_err_t err = continuous_servo_init(&kServoConfig, adc_value, &servo_handle);
  ESP_ERROR_CHECK(err);

  // Option 1. Apply a velocity, either as an actual velocity or as a pulse
  // width value. Only recommended for testing, and make sure to be careful
  // since it won't stop moving until you set a new velocity.

  // These are equivalent, they all stop the servo.
  continuous_servo_apply_velocity(servo_handle, (AngularVelocity){0.F});
  continuous_servo_apply_pulse_width_as_velocity(servo_handle, 1500);

  // Option 2. Set a target angle and velocity followed by calling the update
  // function periodically. This will move the servo towards the target angle
  // using the provided velocity (clamped to the max velocities specified in
  // the configuration) and the acceleration you specified in the
  // configuration, with trapezoidal ramping. The velocity will be clamped to
  // the max velocity specified in the configuration. If the update function
  // isn't called periodically for some reason, the motor will keep moving at
  // its latest set velocity, so make sure your task doesn't get too delayed
  // or prevented from running at all.
  continuous_servo_set_target_velocity(servo_handle, (AngularVelocity){5.f});
  continuous_servo_set_target_angle(servo_handle, (JointAngle){15.F});

  enum {
    kPeriodMs = 10,
  };

  TickType_t current_tick = xTaskGetTickCount();

  while (true) {
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(kPeriodMs));
    // Again, how you get the ADC value is up to you.
    adc_value = get_adc_value_for_potentiometer();
    // Update servo velocity based on the angle read from the potentiometer.
    continuous_servo_update(servo_handle, kPeriodMs, adc_value);
  }
}
```
