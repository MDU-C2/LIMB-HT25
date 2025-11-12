#include "can_interface.h"

#include <string.h>
#include <math.h>
#include <limits.h>

#include "driver/twai.h"
#include "esp_log.h"

#include "app_config.h"

static const char *TAG = "can_if";

typedef struct {
    can_interface_config_t cfg; // Configuration for the CAN interface
    bool is_started; // Whether the CAN interface is started
    uint32_t current_baud; // Current baud rate of the CAN interface
} can_interface_ctx_t; // Context for the CAN interface

static can_interface_ctx_t s_ctx = {0};

/*
 * @brief Selects the timing configuration for the CAN interface based on the baud rate.
 * @param baud_rate - The baud rate to select the timing configuration for.
 * @return The timing configuration for the CAN interface.
 */
static twai_timing_config_t select_timing(uint32_t baud_rate)
{
    if (baud_rate >= 1000000) {
        return TWAI_TIMING_CONFIG_1MBITS();
    } else if (baud_rate >= 800000) {
        twai_timing_config_t cfg = TWAI_TIMING_CONFIG_1MBITS();
        cfg.brp = 4; // approx 800 kbps (depending on APB clk)
        return cfg;
    } else if (baud_rate >= 500000) {
        return TWAI_TIMING_CONFIG_500KBITS();
    } else if (baud_rate >= 250000) {
        return TWAI_TIMING_CONFIG_250KBITS();
    }
    return TWAI_TIMING_CONFIG_500KBITS();
}

/*
 * @brief Configures the CAN driver based on the baud rate.
 * @param baud_rate - The baud rate to configure the driver for.
 * @return ESP_OK if the driver was configured successfully, otherwise an error code.
 */
static esp_err_t configure_driver(uint32_t baud_rate)
{
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_GPIO, CAN_RX_GPIO, TWAI_MODE_NORMAL); // General configuration for the CAN driver
    g_config.tx_queue_len = 8; // Length of the transmit queue
    g_config.rx_queue_len = 32; // Length of the receive queue
    g_config.clkout_divider = 0; // Clock output divider
    g_config.alerts_enabled = TWAI_ALERT_RX_FIFO_OVERRUN | TWAI_ALERT_ERR_PASS | TWAI_ALERT_BUS_OFF | TWAI_ALERT_BUS_RECOVERED; // Alerts enabled

    twai_timing_config_t t_config = select_timing(baud_rate); // Timing configuration for the CAN driver
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL(); // Filter configuration for the CAN driver

    ESP_LOGI(TAG, "Installing TWAI driver at %lu bps", (unsigned long)baud_rate);
    esp_err_t err = twai_driver_install(&g_config, &t_config, &f_config); // Install the CAN driver
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install TWAI driver: %s", esp_err_to_name(err));
        return err;
    }

    err = twai_start(); // Start the CAN driver
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start TWAI driver: %s", esp_err_to_name(err));
        twai_driver_uninstall();
        return err;
    }

    return ESP_OK;
}

/*
 * @brief Initializes the CAN interface.
 * @param config - The configuration for the CAN interface.
 * @return ESP_OK if the interface was initialized successfully, otherwise an error code.
 */
esp_err_t can_interface_init(const can_interface_config_t *config)
{
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    s_ctx.cfg = *config;
    s_ctx.is_started = false;
    s_ctx.current_baud = CAN_BAUD_RATE_DEFAULT; // Default baud rate

    if (s_ctx.cfg.status_mutex == NULL) {
        s_ctx.cfg.status_mutex = xSemaphoreCreateMutex(); // Create a mutex for the status
        if (!s_ctx.cfg.status_mutex) {
            ESP_LOGE(TAG, "Failed to create status mutex");
            return ESP_ERR_NO_MEM;
        }
    }

    return ESP_OK;
}

/*
 * @brief Starts the CAN interface.
 * @param baud_rate - The baud rate to start the interface at.
 * @return ESP_OK if the interface was started successfully, otherwise an error code.
 */
esp_err_t can_interface_start(uint32_t baud_rate)
{
    if (baud_rate < CAN_BAUD_RATE_MIN || baud_rate > CAN_BAUD_RATE_MAX) {
        ESP_LOGW(TAG, "Requested baud %lu outside supported range, clamping", (unsigned long)baud_rate);
        baud_rate = CAN_BAUD_RATE_DEFAULT;
    }

    esp_err_t err = configure_driver(baud_rate); // Configure the driver
    if (err != ESP_OK) {
        return err;
    }

    s_ctx.is_started = true; // Set the interface to started
    s_ctx.current_baud = baud_rate; // Set the current baud rate
    return ESP_OK;
}

/*
 * @brief Handles the ARM command message.
 * @param msg - The message to handle.
 */
static void handle_arm_cmd(const twai_message_t *msg)
{
    if (!s_ctx.cfg.motion_cmd_queue || msg->data_length_code != 12) { // Check if the motion command queue is valid and the message length code is 12
        ESP_LOGW(TAG, "Ignoring ARM_CMD message with DLC=%d", msg->data_length_code);
        return;
    }

    arm_motion_command_t cmd = {0}; // Initialize the motion command
    memcpy(&cmd.target_angle_deg, &msg->data[0], sizeof(float)); // Copy the target angle from the message
    memcpy(&cmd.max_velocity_dps, &msg->data[4], sizeof(float)); // Copy the max velocity from the message
    memcpy(&cmd.max_accel_dps2, &msg->data[8], sizeof(float)); // Copy the max acceleration from the message
    cmd.has_command = true;

    if (xQueueSend(s_ctx.cfg.motion_cmd_queue, &cmd, 0) != pdTRUE) {
        ESP_LOGW(TAG, "Motion command queue full, overwriting last command");
        xQueueOverwrite(s_ctx.cfg.motion_cmd_queue, &cmd); // Overwrite the last command
    }
}

/*
 * @brief Handles the ARM estop message.
 * @param msg - The message to handle.
 */
static void handle_arm_estop(const twai_message_t *msg)
{
    if (!s_ctx.cfg.estop_queue || msg->data_length_code < 1) { // Check if the estop queue is valid and the message length code is greater than 0
        return;
    }

    estop_state_t estop_state = msg->data[0] ? ESTOP_STATE_ACTIVE : ESTOP_STATE_CLEAR; // Convert the message data to an estop state
    if (xQueueSend(s_ctx.cfg.estop_queue, &estop_state, 0) != pdTRUE) { // Send the estop state to the queue
        xQueueOverwrite(s_ctx.cfg.estop_queue, &estop_state); // Overwrite the last estop state
    }
}

/*
 * @brief Handles the IMU orientation message.
 * @param msg - The message to handle.
 */
static void handle_imu_orient(const twai_message_t *msg)
{
    if (!s_ctx.cfg.imu_queue || msg->data_length_code != 6) { // Check if the IMU queue is valid and the message length code is 6
        return;
    }

    imu_orientation_t orientation = {0}; // Initialize the IMU orientation
    int16_t pitch = (int16_t)((msg->data[1] << 8) | msg->data[0]); // Convert the pitch from the message
    int16_t roll = (int16_t)((msg->data[3] << 8) | msg->data[2]); // Convert the roll from the message
    int16_t yaw = (int16_t)((msg->data[5] << 8) | msg->data[4]); // Convert the yaw from the message

    orientation.pitch_deg = (float)pitch / 100.0f; // Convert the pitch to degrees
    orientation.roll_deg = (float)roll / 100.0f; // Convert the roll to degrees
    orientation.yaw_deg = (float)yaw / 100.0f; // Convert the yaw to degrees

    if (xQueueSend(s_ctx.cfg.imu_queue, &orientation, 0) != pdTRUE) { // Send the IMU orientation to the queue
        xQueueOverwrite(s_ctx.cfg.imu_queue, &orientation); // Overwrite the last IMU orientation
    }
}

/*
 * @brief The main receive task for the CAN interface.
 * @param arg - The argument to the task.
 */
void can_interface_rx_task(void *arg)
{
    const TickType_t wait_ticks = pdMS_TO_TICKS(50); // Wait time for the receive task

    while (true) {
        twai_message_t message = {0}; // Initialize the message
        esp_err_t err = twai_receive(&message, wait_ticks); // Receive the message
        if (err == ESP_OK) {
            switch (message.identifier) {
                case CAN_ID_ARM_CMD:
                    handle_arm_cmd(&message); // Handle the ARM command message
                    break;
                case CAN_ID_ARM_ESTOP:
                    handle_arm_estop(&message); // Handle the ARM estop message
                    break;
                case CAN_ID_IMU_ORIENT:
                    handle_imu_orient(&message); // Handle the IMU orientation message
                    break;
                default:
                    ESP_LOGD(TAG, "Unhandled CAN ID 0x%03X", message.identifier);
                    break;
            }
        } else if (err != ESP_ERR_TIMEOUT) {
            ESP_LOGE(TAG, "CAN receive error: %s", esp_err_to_name(err));
        }
    }
}

/*
 * @brief Converts a float value to a fixed 100 value.
 * @param value - The value to convert.
 * @return The fixed 100 value.
 */
static inline uint16_t float_to_fixed_100(float value)
{
    int32_t scaled = (int32_t)lroundf(value * 100.0f);
    if (scaled > INT16_MAX) {
        scaled = INT16_MAX;
    } else if (scaled < INT16_MIN) {
        scaled = INT16_MIN;
    }
    return (uint16_t)scaled;
}

/*
 * @brief Sends the IMU orientation message.
 * @param orientation - The IMU orientation to send.
 * @return ESP_OK if the message was sent successfully, otherwise an error code.
 */
esp_err_t can_interface_send_orientation(const imu_orientation_t *orientation)
{
    if (!orientation) {
        return ESP_ERR_INVALID_ARG;
    }

    twai_message_t message = {
        .identifier = CAN_ID_IMU_ORIENT,
        .data_length_code = 6, // Length of the data in the message
        .rtr = 0,
    }; // Non-remote transmission request

    uint16_t pitch = float_to_fixed_100(orientation->pitch_deg); // Convert the pitch to a fixed 100 value
    uint16_t roll = float_to_fixed_100(orientation->roll_deg); // Convert the roll to a fixed 100 value
    uint16_t yaw = float_to_fixed_100(orientation->yaw_deg); // Convert the yaw to a fixed 100 value

    message.data[0] = pitch & 0xFF; // Copy the pitch to the message data
    message.data[1] = (pitch >> 8) & 0xFF; // Copy the pitch to the message data
    message.data[2] = roll & 0xFF; // Copy the roll to the message data
    message.data[3] = (roll >> 8) & 0xFF; // Copy the roll to the message data
    message.data[4] = yaw & 0xFF; // Copy the yaw to the message data
    message.data[5] = (yaw >> 8) & 0xFF; // Copy the yaw to the message data

    if (s_ctx.cfg.status_mutex) {
        if (xSemaphoreTake(s_ctx.cfg.status_mutex, pdMS_TO_TICKS(10)) == pdTRUE) { // Take the status mutex
            esp_err_t err = twai_transmit(&message, pdMS_TO_TICKS(10)); // Send the message
            xSemaphoreGive(s_ctx.cfg.status_mutex); // Give the status mutex
            return err;
        }
        return ESP_ERR_TIMEOUT;
    }
    return twai_transmit(&message, pdMS_TO_TICKS(10)); // Send the message
}

/*
 * @brief Sends the ARM status message.
 * @param status - The ARM status to send.
 * @return ESP_OK if the message was sent successfully, otherwise an error code.
 */
esp_err_t can_interface_send_status(const arm_status_t *status)
{
    if (!status) {
        return ESP_ERR_INVALID_ARG;
    }

    twai_message_t message = {
        .identifier = CAN_ID_ARM_STATUS, // Identifier for the ARM status message
        .data_length_code = 8, // Length of the data in the message
        .rtr = 0,
    };

    memcpy(&message.data[0], &status->angle_deg, sizeof(float)); // Copy the angle to the message data
    memcpy(&message.data[4], &status->position_error_deg, sizeof(float)); // Copy the position error to the message data

    if (s_ctx.cfg.status_mutex) {
        if (xSemaphoreTake(s_ctx.cfg.status_mutex, pdMS_TO_TICKS(10)) == pdTRUE) { // Take the status mutex
            esp_err_t err = twai_transmit(&message, pdMS_TO_TICKS(10)); // Send the message
            xSemaphoreGive(s_ctx.cfg.status_mutex); // Give the status mutex
            return err;
        }
        return ESP_ERR_TIMEOUT;
    }

    return twai_transmit(&message, pdMS_TO_TICKS(10)); // Send the message
}

/*
 * @brief Sends the ARM estop state message.
 * @param estop_state - The ARM estop state to send.
 * @return ESP_OK if the message was sent successfully, otherwise an error code.
 */
esp_err_t can_interface_send_estop_state(estop_state_t estop_state)
{
    twai_message_t message = {
        .identifier = CAN_ID_ARM_ESTOP, // Identifier for the ARM estop state message
        .data_length_code = 1, // Length of the data in the message
        .rtr = 0,
    };
    message.data[0] = (uint8_t)estop_state; // Copy the estop state to the message data

    if (s_ctx.cfg.status_mutex) {
        if (xSemaphoreTake(s_ctx.cfg.status_mutex, pdMS_TO_TICKS(10)) == pdTRUE) { // Take the status mutex
            esp_err_t err = twai_transmit(&message, pdMS_TO_TICKS(10)); // Send the message
            xSemaphoreGive(s_ctx.cfg.status_mutex); // Give the status mutex
            return err;
        }
        return ESP_ERR_TIMEOUT;
    }
    return twai_transmit(&message, pdMS_TO_TICKS(10)); // Send the message
}

/*
 * @brief Shuts down the CAN interface.
 */
void can_interface_shutdown(void)
{
    if (!s_ctx.is_started) {
        return;
    }

    ESP_LOGI(TAG, "Stopping TWAI driver");
    twai_stop(); // Stop the CAN driver
    twai_driver_uninstall(); // Uninstall the CAN driver
    s_ctx.is_started = false; // Set the interface to not started
}

