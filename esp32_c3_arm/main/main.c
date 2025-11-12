#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_err.h"
#include "nvs_flash.h"

#include "app_config.h"
#include "app_types.h"
#include "can_interface.h"
#include "imu_driver.h"
#include "motion_control.h"

static const char *TAG = "app";

static QueueHandle_t s_motion_cmd_queue = NULL;
static QueueHandle_t s_estop_queue = NULL;
static QueueHandle_t s_can_imu_queue = NULL;
static QueueHandle_t s_local_imu_queue = NULL;
static SemaphoreHandle_t s_can_tx_mutex = NULL;

/*
 * @brief The main receive task for the CAN interface.
 * @param arg - The argument to the task.
 */
static void can_rx_task_entry(void *arg)
{
    (void)arg;
    can_interface_rx_task(NULL); // Receive tasks for the CAN interface
}

/*
 * @brief The IMU task for the application.
 * @param arg - The argument to the task.
 */
static void imu_task_entry(void *arg)
{
    (void)arg;
    const TickType_t period_ticks = pdMS_TO_TICKS(1000 / IMU_UPDATE_HZ); // Period for the IMU task
    TickType_t last_wake = xTaskGetTickCount(); // Last wake time for the IMU task
    imu_orientation_t orientation = {0};

    while (true) {
        vTaskDelayUntil(&last_wake, period_ticks);
        float dt = (float)period_ticks / (float)configTICK_RATE_HZ;
        if (imu_driver_read_orientation(dt, &orientation) == ESP_OK) { // Read the orientation from the IMU
            if (xQueueOverwrite(s_local_imu_queue, &orientation) != pdTRUE) { // Overwrite the last orientation
                // queue length 1, overwrite ensures latest orientation always available
            }
        }
    }
}

/*
 * @brief The motion task for the application.
 * @param arg - The argument to the task.
 */
static void motion_task_entry(void *arg)
{
    (void)arg;
    const TickType_t period_ticks = pdMS_TO_TICKS(MOTION_CONTROL_PERIOD_MS); // Period for the motion task
    TickType_t last_wake = xTaskGetTickCount(); // Last wake time for the motion task

    arm_motion_command_t motion_cmd = {0};
    estop_state_t estop_state = ESTOP_STATE_CLEAR;

    while (true) {
        if (xQueueReceive(s_motion_cmd_queue, &motion_cmd, 0) == pdTRUE) { // Receive the motion command
            ESP_ERROR_CHECK(motion_control_apply_command(&motion_cmd)); // Apply the motion command
        }

        if (xQueueReceive(s_estop_queue, &estop_state, 0) == pdTRUE) { // Receive the estop state
            motion_control_handle_estop(estop_state); // Handle the estop state
        }

        float dt = (float)period_ticks / (float)configTICK_RATE_HZ;
        motion_control_update(dt); // Update the motion control

        vTaskDelayUntil(&last_wake, period_ticks);
    }
}

/*
 * @brief The CAN TX task for the application.
 * @param arg - The argument to the task.
 */
static void can_tx_task_entry(void *arg)
{
    (void)arg;
    const TickType_t period_ticks = pdMS_TO_TICKS(1000 / CAN_STATUS_HZ); // Period for the CAN TX task
    TickType_t last_wake = xTaskGetTickCount(); // Last wake time for the CAN TX task
    imu_orientation_t orientation = {0};
    arm_status_t status = {0};

    while (true) {
        vTaskDelayUntil(&last_wake, period_ticks);

        if (xQueuePeek(s_local_imu_queue, &orientation, 0) != pdTRUE) {
            // keep using last value
        }

        motion_control_get_status(&status); // Get the status from the motion control

        esp_err_t err = can_interface_send_status(&status); // Send the status to the CAN interface
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Failed to send status frame: %s", esp_err_to_name(err));
        }

        err = can_interface_send_orientation(&orientation); // Send the orientation to the CAN interface
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Failed to send orientation frame: %s", esp_err_to_name(err));
        }
    }
}

/*
 * @brief Processes the incoming CAN orientation.
 */
static void process_incoming_can_orientation(void)
{
    imu_orientation_t orientation;
    while (xQueueReceive(s_can_imu_queue, &orientation, 0) == pdTRUE) { // Receive the incoming orientation
        // For now, log the incoming IMU data. Could be fused later.
        ESP_LOGI(TAG, "Received CAN IMU: pitch=%.2f roll=%.2f yaw=%.2f", orientation.pitch_deg, orientation.roll_deg, orientation.yaw_deg); // Log the incoming orientation
    }
}

/*
 * @brief The main function for the application.
 */
void app_main(void)
{
    // Initialize Non-Volatile Storage (NVS)
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase()); // Erase the NVS flash
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // Create queues and mutex
    s_motion_cmd_queue = xQueueCreate(1, sizeof(arm_motion_command_t));
    s_estop_queue = xQueueCreate(1, sizeof(estop_state_t));
    s_can_imu_queue = xQueueCreate(1, sizeof(imu_orientation_t));
    s_local_imu_queue = xQueueCreate(1, sizeof(imu_orientation_t));
    s_can_tx_mutex = xSemaphoreCreateMutex();

    if (!s_motion_cmd_queue || !s_estop_queue || !s_can_imu_queue || !s_local_imu_queue || !s_can_tx_mutex) {
        ESP_LOGE(TAG, "Failed to create queues or mutexes");
        abort();
    }

    // Initialize motion control
    motion_control_config_t motion_cfg = {
        .step_gpio = STEPPER_STEP_GPIO, // Step GPIO for the motion control
        .dir_gpio = STEPPER_DIR_GPIO, // Direction GPIO for the motion control
        .enable_gpio = STEPPER_ENABLE_GPIO, // Enable GPIO for the motion control
        .timer_resolution_hz = MOTION_TIMER_RESOLUTION_HZ, // Resolution for the motion control
        .timer_base_period_us = 50, // Base period for the motion control
        .pulse_width_us = 10, // Pulse width for the motion control
        .steps_per_revolution = DEFAULT_STEPS_PER_REV, // Steps per revolution for the motion control
        .microstepping = DEFAULT_MICROSTEPPING, // Microstepping for the motion control
        .gear_ratio = DEFAULT_GEAR_RATIO, // Gear ratio for the motion control
    };
    ESP_ERROR_CHECK(motion_control_init(&motion_cfg));

    // Initialize IMU driver
    imu_driver_config_t imu_cfg = {
        .i2c_port = IMU_I2C_PORT, // I2C port for the IMU
        .sda_gpio = IMU_I2C_SDA_GPIO, // SDA GPIO for the IMU
        .scl_gpio = IMU_I2C_SCL_GPIO, // SCL GPIO for the IMU
        .clk_speed_hz = IMU_I2C_FREQ_HZ, // Clock speed for the IMU
        .i2c_addr = LSM6DSO32_I2C_ADDR, // I2C address for the IMU
    };
    ESP_ERROR_CHECK(imu_driver_init(&imu_cfg));

    // Initialize CAN interface
    can_interface_config_t can_cfg = {
        .motion_cmd_queue = s_motion_cmd_queue, // Motion command queue for the CAN interface
        .estop_queue = s_estop_queue, // E-stop queue for the CAN interface
        .imu_queue = s_can_imu_queue, // IMU queue for the CAN interface
        .status_mutex = s_can_tx_mutex, // Status mutex for the CAN interface
    };
    ESP_ERROR_CHECK(can_interface_init(&can_cfg)); // Initialize the CAN interface
    ESP_ERROR_CHECK(can_interface_start(CAN_BAUD_RATE_DEFAULT)); // Start the CAN interface

    ESP_LOGI(TAG, "System initialized");

    // Create tasks
    // CAN RX task
    BaseType_t task_ok = xTaskCreate(can_rx_task_entry, "can_rx", CAN_RX_TASK_STACK, NULL, CAN_RX_TASK_PRIO, NULL);
    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create CAN RX task");
        abort();
    }

    // IMU task
    task_ok = xTaskCreate(imu_task_entry, "imu", IMU_TASK_STACK, NULL, IMU_TASK_PRIO, NULL);
    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create IMU task");
        abort();
    }

    // Motion task
    task_ok = xTaskCreate(motion_task_entry, "motion", MOTION_TASK_STACK, NULL, MOTION_TASK_PRIO, NULL);
    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create motion task");
        abort();
    }

    // CAN TX task
    task_ok = xTaskCreate(can_tx_task_entry, "can_tx", CAN_TX_TASK_STACK, NULL, CAN_TX_TASK_PRIO, NULL);
    if (task_ok != pdPASS) {
        ESP_LOGE(TAG, "Failed to create CAN TX task");
        abort();
    }

    // Process incoming CAN orientation
    while (true) {
        process_incoming_can_orientation(); // Process the incoming CAN orientation
        vTaskDelay(pdMS_TO_TICKS(100)); // 100ms delay
    }
}