#pragma once

/**
 * @file motion_control.h
 * @brief Stepper motion control for robotic arm joint.
 */

#include "freertos/FreeRTOS.h"
#include "esp_err.h"
#include "driver/gpio.h"
#include "driver/gptimer.h"

#include "app_types.h"

typedef struct {
    gpio_num_t step_gpio;
    gpio_num_t dir_gpio;
    gpio_num_t enable_gpio;
    uint32_t timer_resolution_hz;
    uint32_t timer_base_period_us;
    uint32_t pulse_width_us;
    int32_t steps_per_revolution;
    int32_t microstepping;
    float gear_ratio;
} motion_control_config_t;

esp_err_t motion_control_init(const motion_control_config_t *config);

esp_err_t motion_control_apply_command(const arm_motion_command_t *command);

void motion_control_handle_estop(estop_state_t state);

void motion_control_update(float dt_seconds);

float motion_control_get_current_angle_deg(void);

float motion_control_get_target_angle_deg(void);

float motion_control_get_error_deg(void);

void motion_control_get_status(arm_status_t *status);

