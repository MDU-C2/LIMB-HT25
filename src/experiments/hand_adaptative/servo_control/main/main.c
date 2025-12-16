#include "esp_log.h"
#include "HS422_led.h"
#include "can_driver.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "SERVIS";
#define ID_SENSOR_DATA 0x020 
#define NUM_FINGERS 5 

void app_main() 
{
    ESP_LOGI(TAG, "Starting servo control application");
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // Initialize all servos
    ESP_LOGI(TAG, "Initializing servos...");
    servo_led_init();
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Initialize rotary encoder
    ESP_LOGI(TAG, "Initializing rotary encoder...");
    rotary_encoder_init();
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Start calibration mode
    // Uncomment the line below to enter calibration mode
    // start_calibration_mode();

    //init CAN CX---------------
    can_init(9, 10, 125000);
    uint8_t msg_rx[5];
    uint32_t rx_id;
    uint8_t rx_len = sizeof(msg_rx); 
    //--------------------------

    // Variables estáticas para almacenar los 5 valores de milivoltios recibidos
    
    ESP_LOGI(TAG, "Starting servo test loop...");
    ESP_LOGI(TAG, "number %d", rx_len);
    
    while(1) {

        // Intentamos recibir un mensaje CAN (bloquea hasta 100ms)
        if (can_receive(&rx_id, msg_rx, &rx_len, 100) == ESP_OK) {
            
            // 1. Verificación del ID
            if (rx_id == ID_SENSOR_DATA && rx_len == NUM_FINGERS) {
                
                // Mensaje válido: procesar los 5 ángulos
                for (int i = 0; i < NUM_FINGERS; i++) {
                    
                    // El byte i de la data es el ángulo (0-180) para el canal i
                    int angle = (int)msg_rx[i];
                    
                    // Aplicar el ángulo al canal de servo correspondiente (i)
                    // El canal del servo (0 a 4) coincide con el índice del dedo (i)
                    
                    // servo_write_deg_channel(i, angle); //
                    
                }

                ESP_LOGI(TAG, "Angles %d - %d - %d - %d - %d ", (int)msg_rx[0], (int)msg_rx[1], (int)msg_rx[2], (int)msg_rx[3], (int)msg_rx[4]);
                
            } else {
                 ESP_LOGI(TAG, "CAN RX: Mensaje con ID 0x%X y Longitud %d ignorado (Esperado: ID 0x020, Len 5).", rx_id, rx_len);
            } 

        } 

        // start_calibration_mode();
        
        // make_fist_gesture();
        // vTaskDelay(pdMS_TO_TICKS(2000));
        // open_hand_gesture();
        // vTaskDelay(pdMS_TO_TICKS(2000));
        // count_to_five_gesture();
        // vTaskDelay(pdMS_TO_TICKS(2000));
        // make_peace_gesture();
        // vTaskDelay(pdMS_TO_TICKS(2000));
        // rock_gesture();
        // vTaskDelay(pdMS_TO_TICKS(2000));
        servo_write_deg_channel(0, 0); // thumb
        servo_write_deg_channel(1, 0); // pinky
        servo_write_deg_channel(2, 0); // ring
        servo_write_deg_channel(3, 0); // mid
        servo_write_deg_channel(4, 0); //index

        vTaskDelay(pdMS_TO_TICKS(3000)); 

        // servo_write_deg_channel(0, 90); // thumb
        // servo_write_deg_channel(1, 90); // pinky
        // servo_write_deg_channel(2, 90); // ring
        // servo_write_deg_channel(3, 90); // mid
        // servo_write_deg_channel(4, 90); //index

        // vTaskDelay(pdMS_TO_TICKS(3000)); 

        servo_write_deg_channel(0, 160); // thumb
        servo_write_deg_channel(1, 160); // index
        servo_write_deg_channel(2, 160); // middle
        servo_write_deg_channel(3, 160); // ring
        servo_write_deg_channel(4, 160); // pinky

        vTaskDelay(pdMS_TO_TICKS(3000)); 

    }
}