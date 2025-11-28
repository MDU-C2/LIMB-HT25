#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
// #include "can_driver.h"  // Commented out - no CAN hardware
#include "imu.h"
#include "stepper.h"

static const char *TAG = "robot_elbow_module";

// CAN configuration
#define CAN_TX_PIN 5 // TODO: Check that this is the correct pin for the CAN TX
#define CAN_RX_PIN 4 // TODO: Check that this is the correct pin for the CAN RX
#define CAN_BAUDRATE 125000 // TODO: Check that this is the correct baudrate
#define CAN_MSG_SIZE 8 // TODO: Check that this is the correct size for the CAN message

// CAN message IDs
#define CAN_ID_ELBOW_STATUS 0x030 // TODO: Check that these are correct IDs
#define CAN_ID_ELBOW_COMMAND 0x010 // TODO: Check that these are correct IDs

// GPIO pin definitions
#define STEPPER_STEP_PIN GPIO_NUM_6
#define STEPPER_DIR_PIN GPIO_NUM_7
#define STEPPER_ENABLE_PIN GPIO_NUM_8

// CAN RX task commented out - no CAN hardware
/*
void can_rx_task(void *pvParameter) {
    uint8_t msg_rx[CAN_MSG_SIZE]; 
    uint8_t rx_len = CAN_MSG_SIZE;
    uint32_t rx_id;
    
    
    while (1) {
        if (can_receive(&rx_id, msg_rx, &rx_len, 100) == ESP_OK) {
            if (rx_id == CAN_ID_ELBOW_COMMAND) {
                // Simple command: first byte is angle in degrees (signed), just for testing.
                int8_t target_angle = (int8_t)msg_rx[0];
                stepper_set_target_angle_deg((float)target_angle);
                ESP_LOGI(TAG, "Received command: target angle = %d degrees", target_angle);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
*/

void imu_task(void *pvParameter) {
    imu_data_t imu_data; // (imu_vector_t) accel and (imu_vector_t) gyro      
    
    while (1) {
        if (imu_read_data(&imu_data) == ESP_OK) {
            // Log IMU data instead of sending over CAN (no CAN hardware)
            ESP_LOGI(TAG, "IMU - Accel: X=%d, Y=%d, Z=%d | Gyro: X=%d, Y=%d, Z=%d",
                     imu_data.accel.x, imu_data.accel.y, imu_data.accel.z,
                     imu_data.gyro.x, imu_data.gyro.y, imu_data.gyro.z);
        }
        vTaskDelay(pdMS_TO_TICKS(100)); // 10 Hz
    }
}

void stepper_task(void *pvParameter) {
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t period_ms = pdMS_TO_TICKS(10); // 10ms = 100Hz update rate
    
    while (1) {
        float dt = 0.01f; // 10ms in seconds
        stepper_update(dt);
        
        // Log status periodically instead of sending over CAN
        static uint32_t status_counter = 0;
        if (++status_counter >= 10) { // Every 100ms
            status_counter = 0;
            float current_angle = stepper_get_current_angle_deg();
            float target_angle = stepper_get_target_angle_deg();
            float velocity = stepper_get_current_velocity_dps();
            bool moving = stepper_is_moving();
            ESP_LOGI(TAG, "Stepper - Current: %.2f°, Target: %.2f°, Velocity: %.2f°/s, Moving: %s",
                     current_angle, target_angle, velocity, moving ? "Yes" : "No");
        }
        
        vTaskDelayUntil(&last_wake_time, period_ms);
    }
}

// Test task to cycle through different target angles
void stepper_test_task(void *pvParameter) {
    // Wait a bit for system to initialize
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // Test angles to cycle through (in degrees)
    float test_angles[] = {0.0f, 30.0f, -30.0f, 45.0f, -45.0f, 0.0f};
    int num_angles = sizeof(test_angles) / sizeof(test_angles[0]);
    int angle_index = 0;
    
    ESP_LOGI(TAG, "Stepper test task started - will cycle through test angles");
    
    while (1) {
        // Set new target angle
        float target = test_angles[angle_index];
        stepper_set_target_angle_deg(target);
        ESP_LOGI(TAG, ">>> Setting target angle to %.1f°", target);
        
        // Wait for stepper to reach target (or timeout after 5 seconds)
        TickType_t start_time = xTaskGetTickCount();
        while (stepper_is_moving() && (xTaskGetTickCount() - start_time < pdMS_TO_TICKS(5000))) {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
        
        // Hold at this position for 2 seconds
        vTaskDelay(pdMS_TO_TICKS(2000));
        
        // Move to next angle
        angle_index = (angle_index + 1) % num_angles;
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "Robot elbow module starting...");
    
    // Initialize hardware
    // can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE);  // Commented out - no CAN hardware
    
    imu_config_t imu_cfg = IMU_CONFIG_DEFAULT();
    imu_cfg.sda_pin = 10;  // Custom SDA pin
    imu_cfg.scl_pin = 9;   // Custom SCL pin
    imu_init(&imu_cfg);
    ESP_LOGI(TAG, "IMU initialized (SDA=10, SCL=9)");
    
    // Initialize stepper motor
    stepper_control_config_t stepper_cfg = {
        .step_gpio = STEPPER_STEP_PIN,
        .dir_gpio = STEPPER_DIR_PIN,
        .enable_gpio = STEPPER_ENABLE_PIN,
        .steps_per_rev = 200,
        .gear_ratio = 1.0f,
        .max_velocity_dps = 90.0f,
        .min_velocity_dps = 1.0f,
        .max_accel_dps2 = 100.0f,
        .pot_adc_channel = ADC_CHANNEL_2
    };
    if (stepper_init(&stepper_cfg) == ESP_OK) {
        ESP_LOGI(TAG, "Stepper initialized");
    } else {
        ESP_LOGE(TAG, "Failed to initialize stepper");
    }
    
    // Create FreeRTOS tasks
    // xTaskCreate(can_rx_task, "can_rx", 4096, NULL, 5, NULL);  // Commented out - no CAN hardware
    xTaskCreate(imu_task, "imu_task", 4096, NULL, 5, NULL);
    xTaskCreate(stepper_task, "stepper_task", 4096, NULL, 5, NULL);
    xTaskCreate(stepper_test_task, "stepper_test", 4096, NULL, 4, NULL);  // Lower priority than stepper_task
    
    ESP_LOGI(TAG, "Tasks created, system running");
}
