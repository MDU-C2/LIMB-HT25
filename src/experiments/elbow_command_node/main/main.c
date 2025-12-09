#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "can_driver.h"

static const char *TAG = "ELBOW_CMD";

// CAN configuration (must match elbow module)
#define CAN_TX_PIN 5
#define CAN_RX_PIN 4
#define CAN_BAUDRATE 1000000

// CAN message IDs (must match elbow module)
#define CAN_ID_ELBOW_STATUS 0x030
#define CAN_ID_ELBOW_COMMAND 0x010
#define CAN_ID_UPPER_ARM_ROTATION_STATUS 0x040
#define CAN_ID_UPPER_ARM_ROTATION_COMMAND 0x015

// Task to send periodic commands
void command_task(void *pvParameter) {
    // Wait for system to initialize
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    ESP_LOGI(TAG, "Starting command sequence...");
    
    // Test angles to send (in degrees, as signed 8-bit values)
    const float test_angles[] = {0, 30, -30, 45, -45, 60, -60, 0};
    int num_angles = sizeof(test_angles) / sizeof(test_angles[0]);
    int angle_index = 0;
    
    while (1) {
        // Send target angle command
        float target_angle = test_angles[angle_index];
        uint8_t can_data[8] = {0};
        *(float*)can_data = target_angle;
        
        esp_err_t ret = can_send(CAN_ID_ELBOW_COMMAND, can_data, 8);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, ">>> Sent command: target angle = %f degrees", target_angle);
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
                float angle = *(float*)msg_rx;
                ESP_LOGI(TAG, "<<< Received elbow status: current angle = %.1f°", angle);
            } else if (rx_id == CAN_ID_UPPER_ARM_ROTATION_STATUS) {
                float angle = *(float*)msg_rx;
                ESP_LOGI(TAG, "<<< Received upper arm rotation status: current angle = %.1f°", angle);
            } else {
                ESP_LOGI(TAG, "<<< Received message from ID: %x", rx_id);
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

