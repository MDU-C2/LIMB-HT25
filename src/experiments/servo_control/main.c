#include "esp_log.h"
#include "HS422_led.h"

static const char *TAG = "MAIN";

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
    
    ESP_LOGI(TAG, "Starting servo test loop...");
    ESP_LOGI(TAG, "To enter calibration mode, uncomment start_calibration_mode() in main.c");
    
    // Normal operation - test servo movements
    while(1) {
        //start_calibration_mode();
        
        ESP_LOGI(TAG, "Moving servos to minimum positions");
        servo_write_deg_channel(0, 0);      // Thumb to min
        servo_write_deg_channel(1, 0);      // Index to min
        servo_write_deg_channel(2, 0);      // Middle to min
        servo_write_deg_channel(3, 0);    // Ring to max
        servo_write_deg_channel(4, 0);    // Pinky to max
        vTaskDelay(pdMS_TO_TICKS(2000));
        
        ESP_LOGI(TAG, "Moving servos to maximum positions");
        servo_write_deg_channel(0, 180);    // Thumb to max
        servo_write_deg_channel(1, 180);    // Index to max
        servo_write_deg_channel(2, 180);    // Middle to max
        servo_write_deg_channel(3, 180);      // Ring to min
        servo_write_deg_channel(4, 180);      // Pinky to min
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}