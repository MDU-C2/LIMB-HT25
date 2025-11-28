#pragma once

#include "esp_err.h"
#include "driver/gpio.h"
#include "hal/adc_types.h"

// Calibration constants for potentiometer mapping
// These should be calibrated for each joint
#define RAW_MIN_CAL 0        // Minimum raw ADC value at minimum angle
#define RAW_MAX_CAL 4095     // Maximum raw ADC value at maximum angle (12-bit ADC)
#define DEG_MIN_CAL -90.0f   // Minimum angle in degrees
#define DEG_MAX_CAL 90.0f    // Maximum angle in degrees
#define MAX_JOINT_ANGLE_DEG 90.0f  // Maximum joint angle limit

// Stepper motor control configuration
typedef struct {
    // GPIO pins
    gpio_num_t step_gpio;        // STEP pin (required)
    gpio_num_t dir_gpio;         // DIR pin (GPIO_NUM_NC if not used)
    gpio_num_t enable_gpio;      // ENABLE pin (GPIO_NUM_NC if not used)
    
    // Motor parameters
    uint16_t steps_per_rev;       // Steps per motor revolution (e.g., 200 for 1.8° stepper)
    float gear_ratio;             // Gear reduction ratio (e.g., 10.0 for 10:1 reduction)
    
    // Motion limits (in degrees per second)
    float max_velocity_dps;       // Maximum velocity (degrees/sec)
    float min_velocity_dps;       // Minimum velocity (degrees/sec)
    float max_accel_dps2;         // Maximum acceleration (degrees/sec²)
    
    // Position feedback
    adc_channel_t pot_adc_channel; // ADC channel for potentiometer (use -1 or value >= SOC_ADC_MAX_CHANNEL_NUM if not used)
} stepper_control_config_t;

// Initialize stepper motor controller
esp_err_t stepper_init(const stepper_control_config_t *cfg);

// Deinitialize stepper motor controller
esp_err_t stepper_deinit(void);

// Update stepper control loop (call periodically, e.g., every 10ms)
// dt_seconds: time delta since last update
void stepper_update(float dt_seconds);

// Set target angle (degrees)
void stepper_set_target_angle_deg(float angle_deg);

// Set emergency stop state
void stepper_set_estop(bool active);

// Get current angle from feedback (degrees)
float stepper_get_current_angle_deg(void);

// Get target angle (degrees)
float stepper_get_target_angle_deg(void);

// Get current velocity (degrees per second)
float stepper_get_current_velocity_dps(void);

// Check if motor is moving
bool stepper_is_moving(void);

// Check if position feedback is enabled
bool stepper_has_position_feedback(void);
