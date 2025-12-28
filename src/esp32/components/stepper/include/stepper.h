#pragma once

#include "esp_err.h"
#include "driver/gpio.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"

typedef enum {
    STEPPER_DIR_NORMAL,
    STEPPER_DIR_REVERSE,
} StepperDirection;

// Stepper motor control configuration
typedef struct {
    // GPIO pins
    gpio_num_t step_gpio;        // STEP pin (required)
    gpio_num_t dir_gpio;         // DIR pin (GPIO_NUM_NC if not used)
    gpio_num_t enable_gpio;      // ENABLE pin (GPIO_NUM_NC if not used)
    
    // Motor parameters
    uint16_t steps_per_rev;       // Steps per motor revolution (e.g., 200 for 1.8° stepper)
    float gear_ratio;             // Gear reduction ratio (e.g., 10.0 for 10:1 reduction)
    StepperDirection direction;   // If you want to reverse the direction or not.
    
    // Motion limits (in degrees per second)
    AngularVelocity max_velocity;       // Maximum velocity (degrees/sec)
    AngularVelocity min_velocity;       // Minimum velocity (degrees/sec)
    AngularAcceleration max_accel;         // Maximum acceleration (degrees/sec²)
    
    // Position feedback
    adc_channel_t pot_adc_channel; // ADC channel for potentiometer (use -1 or value >= SOC_ADC_MAX_CHANNEL_NUM if not used)
    Potentiometer potentiometer; // The potentiometer configuration/calibration used with the stepper.

    // The caller has to provide the PWM channel to use (since other
    // components might use some of the channels).
    ledc_channel_t pwm_channel;
} stepper_control_config_t;

typedef ledc_channel_t stepper_control_handle_t;

// Initialize stepper motor controller
esp_err_t stepper_init(const stepper_control_config_t *cfg, const uint16_t *latest_potentiometer_values, uint16_t latest_potentiometer_values_len, stepper_control_handle_t *out_handle);

// Deinitialize stepper motor controller
esp_err_t stepper_deinit(stepper_control_handle_t handle);

// Get the config for the provided handle.
const stepper_control_config_t *stepper_get_cfg(stepper_control_handle_t handle);

// Update stepper control loop (call periodically, e.g., every 10ms)
// dt_seconds: time delta since last update
void stepper_update(stepper_control_handle_t handle, uint16_t dt_ms, const uint16_t *latest_potentiometer_values, uint16_t latest_potentiometer_values_len);

// Set target angle (degrees)
void stepper_set_target_angle(stepper_control_handle_t handle, PotentiometerAngle angle_deg);

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

