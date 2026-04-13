#include "esp_err.h"
#include "esp_log.h"
#include "HS422_led.h"
#include "can_driver.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "portmacro.h"

static const char *TAG = "SERVOS";

enum {
    CAN_TX_PIN = 3,
    CAN_RX_PIN = 4,
    CAN_BAUDRATE = 1000000,
};

void app_main() 
{
    ESP_LOGI(TAG, "Starting servo control application");
    // vTaskDelay(pdMS_TO_TICKS(2000));
    
    // Initialize all servos
    ESP_LOGI(TAG, "Initializing servos...");
    servo_led_init();
    // vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Initialize rotary encoder
    // ESP_LOGI(TAG, "Initializing rotary encoder...");
    // rotary_encoder_init();
    // vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Start calibration mode
    // Uncomment the line below to enter calibration mode
    // start_calibration_mode();

    //init CAN CX---------------
    {
        esp_err_t err = can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE, NULL);
        if (err) {
            ESP_LOGE(TAG, "Couldn't start can driver: %s", esp_err_to_name(err));
            return;
        }
    }
    uint8_t msg_rx[8];
    uint32_t rx_id;
    uint8_t rx_len = 1; 
    
    // uint32_t loop_counter = 0;
    uint8_t cmd_data;
    //--------------------------

    // Variables estáticas para almacenar los 5 valores de milivoltios recibidos
    
    ESP_LOGI(TAG, "Starting servo test loop...");
    // ESP_LOGI(TAG, "number %d", rx_len);
    TickType_t current_tick = xTaskGetTickCount();

    // Ready to pick cup up
    servo_write_deg_channel(WRIST_SERVO_CONFIG_INDEX, 35);

    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(15000));

    // Grip
    for (int i = 0; i < NUM_FINGER_SERVOS; i++) {
        servo_write_deg_channel(i, 10);  // Start at center position
    }

    // place cup down
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(22000));
    servo_write_deg_channel(WRIST_SERVO_CONFIG_INDEX, 52);

    // Release
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(32000));
    for (int i = 0; i < NUM_FINGER_SERVOS; i++) {
        servo_write_deg_channel(i, 180);  // Start at center position
    }
    return;
    
    while(1) {

        if (can_receive(&rx_id, msg_rx, &rx_len, 100) == ESP_OK) {
            // 1. Verificación del ID
            if (rx_id == CAN_ID_ROBOT_THUMB_ACTUATION && rx_len == 1) {
        
                int angle = (int)msg_rx[0];

                for (int i = 0; i < NUM_SERVOS; i++) {
                    servo_write_deg_channel(i, angle);
                    vTaskDelay(pdMS_TO_TICKS(50));
                }

                ESP_LOGI(TAG, "RX-angles %d", angle);
                
            } else if (rx_id == CAN_ID_ROBOT_LOWER_ARM_ROTATION_ACTUATION) {
                float angle = *(float*)msg_rx;
                servo_write_deg_channel(WRIST_SERVO_CONFIG_INDEX, angle);
                ESP_LOGI(TAG, "Actuation wrist to %.2f degrees", angle);
            } else {
                ESP_LOGI(TAG, "CAN RX: Mensaje con ID 0x%X ", rx_id);
            } 

        } 

        // // --- sending test ---
        // vTaskDelay(pdMS_TO_TICKS(100)); 
        // loop_counter++;

        // if (loop_counter == 50) {
        //     ESP_LOGW("TEST", ">>> sending start");
        //     cmd_data = 0x01;
        //     can_send(CAN_ID_ROBOT_HAND_SET_GRIP_STATE, &cmd_data, 1);
        // }

        // if (loop_counter == 150) {
        //     ESP_LOGW("TEST", ">>> sending stop");
        //     cmd_data = 0x02;
        //     can_send(CAN_ID_ROBOT_HAND_SET_GRIP_STATE, &cmd_data, 1);
        //     loop_counter = 0; 
        // }

    }
}
