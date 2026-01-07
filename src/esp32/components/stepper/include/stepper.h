#pragma once

#include "driver/gpio.h"
#include "esp_err.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"

typedef enum {
  STEPPER_DIR_NORMAL,
  STEPPER_DIR_REVERSE,
} StepperDirection;

typedef enum {
  MICROSTEP_NONE = 0,
  MICROSTEP_1_1 = 1,
  MICROSTEP_1_2 = 2,
  MICROSTEP_1_4 = 4,
  MICROSTEP_1_8 = 8,
  MICROSTEP_1_16 = 16,
  MICROSTEP_1_32 = 32,
} MicrosteppingMode;

// Stepper motor control configuration
typedef struct {
  // GPIO pins

  // STEP pin (required)
  gpio_num_t step_gpio;
  // DIR pin (GPIO_NUM_NC if not used)
  gpio_num_t dir_gpio;
  // ENABLE pin (GPIO_NUM_NC if not used)
  gpio_num_t enable_gpio;
  gpio_num_t microstep_m0_gpio;
  gpio_num_t microstep_m1_gpio;
  gpio_num_t microstep_m2_gpio;

  MicrosteppingMode microstepping_mode;

  // Motor parameters

  // Steps per motor revolution (e.g., 200 for 1.8° stepper)
  uint16_t steps_per_rev;
  // Gear reduction ratio (e.g., 10.0 for 10:1 reduction)
  float gear_ratio;
  // If you want to reverse the direction or not.
  StepperDirection direction;

  // Motion limits (in degrees per second)

  // Maximum velocity (degrees/sec)
  AngularVelocity max_velocity_negative;
  AngularVelocity max_velocity_positive;
  // Maximum acceleration (degrees/sec²)
  AngularAcceleration max_accel;

  // Position feedback

  // ADC channel for potentiometer (use -1 or value >= SOC_ADC_MAX_CHANNEL_NUM
  // if not used)
  adc_channel_t pot_adc_channel;
  // The potentiometer configuration/calibration used with the stepper.
  Potentiometer potentiometer;

  // The caller has to provide the PWM channel and timer to use (since other
  // components might also want to use PWM).
  ledc_channel_t pwm_channel;
  ledc_timer_t pwm_timer;
} stepper_control_config_t;

typedef ledc_channel_t stepper_control_handle_t;

// Initialize stepper motor controller
esp_err_t stepper_init(const stepper_control_config_t *cfg,
                       uint16_t latest_potentiometer_adc_value,
                       stepper_control_handle_t *out_handle);

// Deinitialize stepper motor controller
esp_err_t stepper_deinit(stepper_control_handle_t handle);

// Get the config for the provided handle.
const stepper_control_config_t *stepper_get_cfg(
    stepper_control_handle_t handle);

// Update stepper control loop (call periodically, e.g., every 10ms)
// dt_ms: time delta between calls to the function.
void stepper_update(stepper_control_handle_t handle, uint16_t dt_ms,
                    uint16_t latest_potentiometer_adc_value);

// Set target angle (degrees)
void stepper_set_target_angle(stepper_control_handle_t handle,
                              JointAngle target_angle);

// Set emergency stop state
void stepper_set_estop(stepper_control_handle_t handle, bool active);

// Get current angle from feedback (degrees)
PotentiometerAngle stepper_get_current_angle(stepper_control_handle_t handle);

// Get target angle (degrees)
PotentiometerAngle stepper_get_target_angle(stepper_control_handle_t handle);

// Get current velocity (degrees per second)
AngularVelocity stepper_get_current_velocity(stepper_control_handle_t handle);

// Check if motor is moving
bool stepper_is_moving(stepper_control_handle_t handle);

// Check if position feedback is enabled
bool stepper_has_position_feedback(stepper_control_handle_t handle);

