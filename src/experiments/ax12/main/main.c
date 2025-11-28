#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/gpio.h"

static const char *TAG = "AX12_TEST";

// UART Configuration
#define UART_NUM            UART_NUM_0
#define UART_TX_PIN         GPIO_NUM_21  // Change to your TX pin
#define UART_RX_PIN         GPIO_NUM_20  // Change to your RX pin
#define UART_BUF_SIZE       1024
#define BAUDRATE            1000000     // 1Mbps for Dynamixel

// AX12+ Configuration
#define AX12_ID             1            // Change to your servo ID (default is 1)
#define AX12_GOAL_POSITION  30           // Address 30 for goal position
#define AX12_PRESENT_POSITION 36         // Address 36 for present position
#define AX12_MOVING_SPEED   32           // Address 32 for moving speed
#define AX12_TORQUE_ENABLE  24           // Address 24 for torque enable

// Dynamixel Protocol Constants
#define DXL_BROADCAST_ID    0xFE
#define DXL_INSTRUCTION_PING 0x01
#define DXL_INSTRUCTION_READ 0x02
#define DXL_INSTRUCTION_WRITE 0x03

/**
 * @brief Calculate checksum for Dynamixel packet
 */
static uint8_t calculate_checksum(uint8_t *packet, int length)
{
    uint8_t checksum = 0;
    // Sum all bytes from ID (index 2) to the last parameter byte (index length-2)
    // The checksum byte itself is at index length-1
    for (int i = 2; i < length; i++) {  // Changed from length-1 to length
        checksum += packet[i];
    }
    return ~checksum;
}

/**
 * @brief Build a Dynamixel write packet
 */
static int build_write_packet(uint8_t *buffer, uint8_t id, uint8_t address, uint16_t value)
{
    int idx = 0;
    buffer[idx++] = 0xFF;
    buffer[idx++] = 0xFF;
    buffer[idx++] = id;
    buffer[idx++] = 5;  // Length: ID(1) + Length(1) + Instruction(1) + Address(1) + Value(2) + Checksum(1)
    buffer[idx++] = DXL_INSTRUCTION_WRITE;
    buffer[idx++] = address;
    buffer[idx++] = value & 0xFF;        // Low byte
    buffer[idx++] = (value >> 8) & 0xFF; // High byte
    buffer[idx] = calculate_checksum(buffer, idx);
    return idx + 1;
}

/**
 * @brief Build a Dynamixel read packet
 */
static int build_read_packet(uint8_t *buffer, uint8_t id, uint8_t address, uint8_t length)
{
    int idx = 0;
    buffer[idx++] = 0xFF;
    buffer[idx++] = 0xFF;
    buffer[idx++] = id;
    buffer[idx++] = 4;  // Length: ID(1) + Length(1) + Instruction(1) + Address(1) + Length(1) + Checksum(1)
    buffer[idx++] = DXL_INSTRUCTION_READ;
    buffer[idx++] = address;
    buffer[idx++] = length;
    buffer[idx] = calculate_checksum(buffer, idx);
    return idx + 1;
}

/**
 * @brief Build a Dynamixel ping packet
 */
static int build_ping_packet(uint8_t *buffer, uint8_t id)
{
    int idx = 0;
    buffer[idx++] = 0xFF;
    buffer[idx++] = 0xFF;
    buffer[idx++] = id;
    buffer[idx++] = 2;  // Length: ID(1) + Length(1) + Instruction(1) + Checksum(1)
    buffer[idx++] = DXL_INSTRUCTION_PING;
    buffer[idx] = calculate_checksum(buffer, idx);
    return idx + 1;
}

/**
 * @brief Send packet and wait for response
 */
static esp_err_t send_packet(uint8_t *packet, int packet_len, uint8_t *response, int *response_len, int timeout_ms)
{
    // Clear UART buffer
    uart_flush(UART_NUM);
    
    // Send packet
    int bytes_written = uart_write_bytes(UART_NUM, packet, packet_len);
    if (bytes_written != packet_len) {
        ESP_LOGE(TAG, "Failed to write all bytes. Written: %d, Expected: %d", bytes_written, packet_len);
        return ESP_FAIL;
    }
    
    // Wait for response
    vTaskDelay(pdMS_TO_TICKS(timeout_ms));
    
    // Read response
    int len = uart_read_bytes(UART_NUM, response, UART_BUF_SIZE, pdMS_TO_TICKS(100));
    if (len > 0) {
        *response_len = len;
        return ESP_OK;
    }
    
    return ESP_ERR_TIMEOUT;
}

/**
 * @brief Send packet (Tx only, no response waiting)
 */
static esp_err_t send_packet_tx_only(uint8_t *packet, int packet_len)
{
    // Clear UART buffer
    uart_flush(UART_NUM);
    
    // Debug: Print packet bytes
    ESP_LOGI(TAG, "Packet bytes: ");
    for (int i = 0; i < packet_len; i++) {
        ESP_LOGI(TAG, "  [%d] = 0x%02X", i, packet[i]);
    }
    
    // Send packet
    int bytes_written = uart_write_bytes(UART_NUM, packet, packet_len);
    if (bytes_written != packet_len) {
        ESP_LOGE(TAG, "Failed to write all bytes. Written: %d, Expected: %d", bytes_written, packet_len);
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Sent %d bytes", bytes_written);
    return ESP_OK;
}

/**
 * @brief Initialize UART for Dynamixel communication
 */
static esp_err_t init_uart(void)
{
    uart_config_t uart_config = {
        .baud_rate = BAUDRATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, UART_BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, UART_TX_PIN, UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    
    ESP_LOGI(TAG, "UART initialized: TX=%d, RX=%d, Baudrate=%d", UART_TX_PIN, UART_RX_PIN, BAUDRATE);
    return ESP_OK;
}

/**
 * @brief Ping the servo to check if it's connected
 */
static esp_err_t ping_servo(uint8_t id)
{
    uint8_t packet[10];
    uint8_t response[UART_BUF_SIZE];
    int response_len = 0;
    
    int packet_len = build_ping_packet(packet, id);
    
    ESP_LOGI(TAG, "Pinging servo ID %d...", id);
    esp_err_t ret = send_packet(packet, packet_len, response, &response_len, 50);
    
    if (ret == ESP_OK && response_len > 0) {
        ESP_LOGI(TAG, "Servo ID %d responded! Response length: %d", id, response_len);
        return ESP_OK;
    } else {
        ESP_LOGW(TAG, "Servo ID %d did not respond", id);
        return ESP_FAIL;
    }
}

/**
 * @brief Read a value from the servo
 */
static esp_err_t read_servo(uint8_t id, uint8_t address, uint8_t length, uint16_t *value)
{
    uint8_t packet[10];
    uint8_t response[UART_BUF_SIZE];
    int response_len = 0;
    
    int packet_len = build_read_packet(packet, id, address, length);
    
    esp_err_t ret = send_packet(packet, packet_len, response, &response_len, 50);
    
    if (ret == ESP_OK && response_len >= 6) {
        // Response format: 0xFF 0xFF ID LENGTH ERROR DATA... CHECKSUM
        // For 2-byte read, data starts at index 6
        if (length == 2 && response_len >= 7) {
            *value = response[5] | (response[6] << 8);
            return ESP_OK;
        } else if (length == 1 && response_len >= 6) {
            *value = response[5];
            return ESP_OK;
        }
    }
    
    return ESP_FAIL;
}

/**
 * @brief Write a value to the servo
 */
static esp_err_t write_servo(uint8_t id, uint8_t address, uint16_t value)
{
    uint8_t packet[10];
    uint8_t response[UART_BUF_SIZE];
    int response_len = 0;
    
    int packet_len = build_write_packet(packet, id, address, value);
    
    esp_err_t ret = send_packet(packet, packet_len, response, &response_len, 50);
    
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Write successful to servo ID %d, address %d, value %d", id, address, value);
        return ESP_OK;
    }
    
    return ESP_FAIL;
}

/**
 * @brief Set servo position (0-1023, where 512 is center)
 */
static esp_err_t set_position(uint8_t id, uint16_t position)
{
    // Clamp position to valid range (0-1023)
    if (position > 1023) position = 1023;
    
    ESP_LOGI(TAG, "Setting servo ID %d to position %d", id, position);
    return write_servo(id, AX12_GOAL_POSITION, position);
}

/**
 * @brief Get current servo position
 */
static esp_err_t get_position(uint8_t id, uint16_t *position)
{
    return read_servo(id, AX12_PRESENT_POSITION, 2, position);
}

void app_main(void)
{
    ESP_LOGI(TAG, "AX12+ Tx Test Starting...");
    ESP_LOGI(TAG, "Testing transmission only (no Rx operations)");
    ESP_LOGI(TAG, "Current config: TX=%d, RX=%d, Servo ID=%d", UART_TX_PIN, UART_RX_PIN, AX12_ID);
    
    // Initialize UART
    ESP_ERROR_CHECK(init_uart());
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Test 1: Send ping packet (Tx only)
    ESP_LOGI(TAG, "\n=== Test 1: Send Ping Packet (Tx Only) ===");
    uint8_t ping_packet[10];
    int ping_len = build_ping_packet(ping_packet, AX12_ID);
    send_packet_tx_only(ping_packet, ping_len);
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Test 2: Enable torque (Tx only)
    ESP_LOGI(TAG, "\n=== Test 2: Enable Torque (Tx Only) ===");
    uint8_t torque_packet[10];
    int torque_len = build_write_packet(torque_packet, AX12_ID, AX12_TORQUE_ENABLE, 1);
    send_packet_tx_only(torque_packet, torque_len);
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Test 3: Send position commands (Tx only)
    ESP_LOGI(TAG, "\n=== Test 3: Send Position Commands (Tx Only) ===");
    uint16_t positions[] = {300, 512, 724, 512};  // Left, Center, Right, Center
    const char *position_names[] = {"Left (300)", "Center (512)", "Right (724)", "Center (512)"};
    
    for (int i = 0; i < 4; i++) {
        ESP_LOGI(TAG, "Sending position command: %s...", position_names[i]);
        uint8_t pos_packet[10];
        int pos_len = build_write_packet(pos_packet, AX12_ID, AX12_GOAL_POSITION, positions[i]);
        send_packet_tx_only(pos_packet, pos_len);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
    
    // Test 4: Continuous position commands (Tx only)
    ESP_LOGI(TAG, "\n=== Test 4: Continuous Position Commands (Tx Only) ===");
    ESP_LOGI(TAG, "Sending position commands from 300 to 724 and back...");
    
    for (int cycle = 0; cycle < 3; cycle++) {
        // Sweep forward
        for (uint16_t pos = 300; pos <= 724; pos += 20) {
            uint8_t pos_packet[10];
            int pos_len = build_write_packet(pos_packet, AX12_ID, AX12_GOAL_POSITION, pos);
            send_packet_tx_only(pos_packet, pos_len);
            vTaskDelay(pdMS_TO_TICKS(100));
        }
        
        // Sweep backward
        for (uint16_t pos = 724; pos >= 300; pos -= 20) {
            uint8_t pos_packet[10];
            int pos_len = build_write_packet(pos_packet, AX12_ID, AX12_GOAL_POSITION, pos);
            send_packet_tx_only(pos_packet, pos_len);
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
    
    ESP_LOGI(TAG, "\n=== Tx Test Complete ===");
    ESP_LOGI(TAG, "Sending final position command (center)...");
    uint8_t final_packet[10];
    int final_len = build_write_packet(final_packet, AX12_ID, AX12_GOAL_POSITION, 512);
    send_packet_tx_only(final_packet, final_len);
    vTaskDelay(pdMS_TO_TICKS(100));
    
    ESP_LOGI(TAG, "All Tx tests completed!");
}

