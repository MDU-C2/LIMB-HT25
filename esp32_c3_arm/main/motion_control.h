#pragma once

/**
 * @file motion_control.h
 * @brief Stepper motion control for robotic arm joint using LEDC.
 */

#include "freertos/FreeRTOS.h"
#include "esp_err.h"
#include "driver/gpio.h"
#include "driver/ledc.h"

#include "app_types.h"

typedef struct {
    gpio_num_t step_gpio;
    gpio_num_t dir_gpio;
    gpio_num_t enable_gpio;
    int32_t steps_per_revolution;
    int32_t microstepping;
    float gear_ratio;
    float max_velocity_dps;      // Maximum velocity in degrees per second
    float min_velocity_dps;      // Minimum reliable velocity while moving
    float max_accel_dps2;        // Maximum acceleration in degrees per second squared
    float deadband_deg;          // Stop window in degrees
    uint32_t control_period_ms;  // Control loop period in milliseconds
} motion_control_config_t;

esp_err_t motion_control_init(const motion_control_config_t *config);

esp_err_t motion_control_apply_command(const arm_motion_command_t *command);

void motion_control_handle_estop(estop_state_t state);

void motion_control_update(float dt_seconds);

float motion_control_get_current_angle_deg(void);

float motion_control_get_target_angle_deg(void);

float motion_control_get_error_deg(void);

void motion_control_get_status(arm_status_t *status);

static float map_pot_to_deg(int raw);

static int read_adc_avg(int n);

static inline float clampf(float x, float lo, float hi);

static float clamp_angle(float angle_deg);

/**
 * @brief Set position feedback from external source (e.g., ADC, encoder)
 * @param angle_deg Current angle in degrees from feedback sensor
 */
void motion_control_set_position_feedback(float angle_deg);

