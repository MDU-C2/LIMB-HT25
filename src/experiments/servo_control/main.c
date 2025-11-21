#include "esp_log.h"
#include "HS422_led.h"

static const char *TAG = "MAIN";

void app_main() 
{
    ESP_LOGI(TAG, "Starting servo test");
    vTaskDelay(pdMS_TO_TICKS(5000));
    // Initialize all servos
    servo_led_init();
    
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    ESP_LOGI(TAG, "Testing individual servos one by one...");
    
   
    while(1) {
    
        servo_write_deg_channel(0, 0);      // Thumb to min
        servo_write_deg_channel(1, 0);      // Index to min
        servo_write_deg_channel(2, 0);      // Middle to min
        servo_write_deg_channel(3, 180);      // Ring to min
        servo_write_deg_channel(4, 180);      // Pinky to min
        vTaskDelay(pdMS_TO_TICKS(2000));
        
        servo_write_deg_channel(0, 180);      // Thumb to min
        servo_write_deg_channel(1, 180);      // Index to min
        servo_write_deg_channel(2, 180);      // Middle to min
        servo_write_deg_channel(3, 0);      // Ring to min
        servo_write_deg_channel(4, 0);      // Pinky to min
        vTaskDelay(pdMS_TO_TICKS(2000));
        
    
    }
}