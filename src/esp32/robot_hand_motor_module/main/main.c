#include "endian.h"
#include "esp_err.h"
#include "esp_log.h"
#include "HS422_led.h"
#include "can_driver.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "portmacro.h"
#include "imu.h"

static const char *TAG = "SERVOS";

enum {
    CAN_TX_PIN = 3,
    CAN_RX_PIN = 4,
    CAN_BAUDRATE = 1000000,
};

static void imu_task([[maybe_unused]] void *pvParameter) {
    const uint16_t period_ms = 1000;
    TickType_t current_tick = xTaskGetTickCount();
    while (true) {
        xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(period_ms));
        ImuRawData raw_data = {0};
        esp_err_t err = imu_read_data(&raw_data);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "Error reading IMU data: %s", esp_err_to_name(err));
            continue;
        }

        ESP_LOGI(TAG, "Read IMU accel [%d, %d, %d], gyro [%d, %d, %d]",
                 raw_data.accel.x, raw_data.accel.y, raw_data.accel.z, raw_data.gyro.pitch, raw_data.gyro.roll, raw_data.gyro.yaw);

        // We store the data in 16-bit arrays to allow us to use `htole16` to ensure
        // the values are sent as little-endian while avoiding breaking strict aliasing
        // when casting to a different pointer type.
        {
            const uint16_t can_data[] = {htole16(raw_data.accel.x), htole16(raw_data.accel.y), htole16(raw_data.accel.z)};
            ESP_ERROR_CHECK_WITHOUT_ABORT(can_send(CAN_ID_ROBOT_HAND_IMU_ACCEL, (const uint8_t*)can_data, sizeof(can_data), 0));
        }
        {
            const uint16_t can_data[] = {htole16(raw_data.gyro.pitch), htole16(raw_data.gyro.roll), htole16(raw_data.gyro.yaw)};
            ESP_ERROR_CHECK_WITHOUT_ABORT(can_send(CAN_ID_ROBOT_HAND_IMU_GYRO, (const uint8_t*)can_data, sizeof(can_data), 0));
        }
    }
}

void app_main() 
{
    ESP_LOGI(TAG, "Starting servo control application");
    // vTaskDelay(pdMS_TO_TICKS(2000));
    
    // Initialize all servos
    ESP_LOGI(TAG, "Initializing servos...");
    servo_led_init();
    // vTaskDelay(pdMS_TO_TICKS(1000));

    {
        ESP_LOGI(TAG, "Initializing IMUs...");
        ImuConfig imu_config = IMU_CONFIG_DEFAULT();
        imu_config.sda_pin = GPIO_NUM_0;
        imu_config.scl_pin = GPIO_NUM_1;
        ESP_ERROR_CHECK_WITHOUT_ABORT(imu_init(&imu_config));
        if (!imu_is_present()) {
            ESP_LOGW(TAG, "IMU isn't present");
        }
    }
    
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

    {
        BaseType_t err =
            xTaskCreate(imu_task, "imu_task", 1024 * 2 * 2,
                        NULL, 6, NULL);
        if (err != pdPASS) {
            ESP_LOGE(TAG, "Failed to create imu task, err code: %d", err);
            can_deinit();
            imu_deinit();
            abort();
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
