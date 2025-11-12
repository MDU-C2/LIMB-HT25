#pragma once

/**
 * @file can_interface.h
 * @brief CAN (TWAI) interface for robotic arm node.
 */

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_err.h"

#include "app_types.h"

/**
 * @brief Configuration structure for the CAN interface.
 */
typedef struct {
    QueueHandle_t motion_cmd_queue; // Queue for the motion commands
    QueueHandle_t estop_queue; // Queue for the estop state messages
    QueueHandle_t imu_queue; // Queue for the IMU orientation messages  
    SemaphoreHandle_t status_mutex; // Mutex for the status messages
} can_interface_config_t;

esp_err_t can_interface_init(const can_interface_config_t *config);

esp_err_t can_interface_start(uint32_t baud_rate);

void can_interface_rx_task(void *arg);

esp_err_t can_interface_send_status(const arm_status_t *status);

esp_err_t can_interface_send_orientation(const imu_orientation_t *orientation);

esp_err_t can_interface_send_estop_state(estop_state_t estop_state);

void can_interface_shutdown(void);

