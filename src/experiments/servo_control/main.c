
#include "esp_log.h"
//#include "freertos/FreeRTOS.h"
//#include "freertos/task.h"
//#include "lsm6dso32.h"
#include "mg90s.h"

//static const char *TAG = "MAIN";



void app_main() 
{
    uint8_t data;
    esp_err_t ret;
    float imu_data_x;
    int current_angle = SERVO_MIN_DEG;

    servo_init();
    /*
    // Disable logging on UART0 to keep JSON output clean
    esp_log_level_set("*", ESP_LOG_NONE);
    
    ESP_ERROR_CHECK(i2c_master_init());
    ESP_LOGI(TAG, "I2C initialized successfully");
 
    ESP_ERROR_CHECK(uart_init());
    ESP_LOGI(TAG, "UART initialized successfully");
 
    // Read the LSM6DSO32 WHO_AM_I register, on power up the register should have the value 0x6C 
    ESP_ERROR_CHECK(lsm6dso32_register_read(LSM6DSO32_WHO_AM_I_REG, &data, 1));
    //ESP_LOGI(TAG, "WHO_AM_I = 0x%02X (expected: 0x6C)", data);
 
    if (data != 0x6C) {
        //ESP_LOGE(TAG, "WHO_AM_I register value incorrect. Expected 0x6C, got 0x%02X", data);
        return;
    }
 
    ESP_ERROR_CHECK(lsm6dso32_init());
    
    // Read LSM6DSO32 sensor data in a loop 
    lsm6dso32_data_t imu_data; */
    while (1) {

        /*ret = lsm6dso32_read_data(&imu_data);
        if (ret == ESP_OK) {

            // Send data via UART
            send_imu_data_json(&imu_data);

        }
        imu_data_x = imu_data.accel.x;
        if(imu_data_x < 0.0f){
            imu_data_x = imu_data_x * -2.0f;
        }
        servo_write_deg(imu_data_x*10);  // Scale accel x to degrees
        prev_angle = get_user_input(prev_angle);
        vTaskDelay(pdMS_TO_TICKS(500));*/

        current_angle = servo_button_control(current_angle, 1);
        
    }
}