# Servo component targeting a modded JX Servo PDI-HV2060MG

This component allows for controlling a JX Servo PDI-HV2060MG that has been modded to be continuous.

## Servo requirements

The servo used needs to have been modified to be continuous.
The basic process used in this project follows [this guide](https://learn.adafruit.com/modifying-servos-for-continuous-rotation/overview).
After removing the internal potentiometer and adding the resistors, the servo can be controlled by providing a velocity instead of an angle.

## Usage

The basic usage is as follows:
```c
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"
#include "servo.h"
#include "soc/gpio_num.h"

static const ServoConfig kServoConfig = {
    .gpio_pin = GPIO_NUM_0,
    .pwm_timer = LEDC_TIMER_0,
    .pwm_channel = LEDC_CHANNEL_0,
    .name = "Servo 1",
    .direction = SERVO_DIR_NORMAL,
    .motionless_pw = 1500,
    .max_capable_angular_velocity = {400},
    .max_capable_angular_velocity_pw_offset = 150,
    .gear_ratio = 15.F,
    .max_velocity = {8.F},
    .max_accel = {8.F},
    .potentiometer =
        (Potentiometer){
            .degrees_of_motion = {285.F},
            .min_adc_value = 20,
            .max_adc_value = 3087,
            .min_potentiometer_angle = {170},
            .max_potentiometer_angle = {200},
            .min_potentiometer_angle_as_joint_angle = {0.F},
            .joint_angle_to_potentiometer_angle_ratio = 18.F / 15.F,
        },
};

void app_main(void) {
  // How you get the ADC value is up to you.
  uint16_t adc_value = get_adc_value_for_potentiometer();
  ServoHandle servo_handle = {0};
  esp_err_t err = servo_init(&kServoConfig, adc_value, &servo_handle);
  ESP_ERROR_CHECK(err);

  // Option 1. Apply a velocity, either as an actual velocity or as a pulse
  // width value. Only recommended for testing, and make sure to be careful
  // since it won't stop moving until you set a new velocity.

  // These are equivalent, they all stop the servo.
  servo_apply_velocity(servo_handle, (AngularVelocity){0.F});
  servo_apply_pulse_width_as_velocity(servo_handle, 1500);
  servo_stop_

  // Option 2. Set a target velocity followed by calling the update function
  // periodically. This will move the servo towards the target angle using the
  // velocity and acceleration you specified in the configuration, with
  // trapezoidal ramping. If the update function isn't called periodically for
  // some reason, the motor will keep moving at its latest set velocity, so
  // make sure your task doesn't get too delayed or prevented from running at
  // all.
  servo_set_target_angle(servo_handle, (JointAngle){15.F});

  enum {
    kPeriodMs = 10,
  };

  TickType_t current_tick = xTaskGetTickCount();

  while (true) {
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(kPeriodMs));
    // Again, how you get the ADC value is up to you.
    adc_value = get_adc_value_for_potentiometer();
    // Update servo velocity based on the angle read from the potentiometer.
    servo_update(servo_handle, kPeriodMs, adc_value);
  }
}
```
