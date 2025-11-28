#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "can_driver.h"

static const char *TAG = "ELBOW_CMD";

// CAN configuration (must match elbow module)
#define CAN_TX_PIN 5
#define CAN_RX_PIN 4
#define CAN_BAUDRATE 125000

// CAN message IDs (must match elbow module)
#define CAN_ID_ELBOW_STATUS 0x030
#define CAN_ID_ELBOW_COMMAND 0x010

// Task to send periodic commands
void command_task(void *pvParameter) {
    // Wait for system to initialize
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    ESP_LOGI(TAG, "Starting command sequence...");
    
    // Test angles to send (in degrees, as signed 8-bit values)
    int8_t test_angles[] = {0, 30, -30, 45, -45, 60, -60, 0};
    int num_angles = sizeof(test_angles) / sizeof(test_angles[0]);
    int angle_index = 0;
    
    while (1) {
        // Send target angle command
        int8_t target_angle = test_angles[angle_index];
        uint8_t can_data[8] = {0};
        can_data[0] = (uint8_t)target_angle;  // First byte is the angle
        
        esp_err_t ret = can_send(CAN_ID_ELBOW_COMMAND, can_data, 8);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, ">>> Sent command: target angle = %d degrees", target_angle);
        } else {
            ESP_LOGE(TAG, "Failed to send CAN message: %s", esp_err_to_name(ret));
        }
        
        // Wait before sending next command
        vTaskDelay(pdMS_TO_TICKS(5000));  // 5 seconds between commands
        
        // Move to next angle
        angle_index = (angle_index + 1) % num_angles;
    }
}

// Task to receive status messages from elbow
void status_rx_task(void *pvParameter) {
    uint8_t msg_rx[8];
    uint32_t rx_id;
    uint8_t rx_len = sizeof(msg_rx);
    
    ESP_LOGI(TAG, "Status RX task started, listening for messages...");
    
    while (1) {
        if (can_receive(&rx_id, msg_rx, &rx_len, 1000) == ESP_OK) {
            if (rx_id == CAN_ID_ELBOW_STATUS) {
                // Parse status message (assuming first 2 bytes are angle as int16_t)
                int16_t angle = (int16_t)(msg_rx[0] | (msg_rx[1] << 8));
                float angle_deg = angle / 10.0f;  // Assuming 0.1 degree resolution
                ESP_LOGI(TAG, "<<< Received status: current angle = %.1f°", angle_deg);
            } else {
                ESP_LOGI(TAG, "<<< Received message from ID: 0x%03lX", rx_id);
            }
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "   Elbow Command Node Starting");
    ESP_LOGI(TAG, "========================================");
    
    // Initialize CAN
    esp_err_t ret = can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize CAN: %s", esp_err_to_name(ret));
        return;
    }
    ESP_LOGI(TAG, "CAN initialized (TX=%d, RX=%d, %d baud)", CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE);
    
    // Create tasks
    xTaskCreate(command_task, "command_task", 4096, NULL, 5, NULL);
    xTaskCreate(status_rx_task, "status_rx", 4096, NULL, 5, NULL);
    
    ESP_LOGI(TAG, "Tasks created, system running");
    ESP_LOGI(TAG, "Will send commands every 5 seconds");
}

