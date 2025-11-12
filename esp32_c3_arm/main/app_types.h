#pragma once

#include <stdbool.h>

typedef struct {
    float pitch_deg;
    float roll_deg;
    float yaw_deg;
} imu_orientation_t;

typedef struct {
    float target_angle_deg;
    float max_velocity_dps;
    float max_accel_dps2;
    bool has_command;
} arm_motion_command_t;

typedef struct {
    float angle_deg;
    float position_error_deg;
    bool estop_active;
} arm_status_t;

typedef enum {
    ESTOP_STATE_CLEAR = 0,
    ESTOP_STATE_ACTIVE = 1,
} estop_state_t;

