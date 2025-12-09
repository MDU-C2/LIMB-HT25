#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "can_driver.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "imu.h"
#include "portmacro.h"
#include "stepper.h"
#include "potentiometer.h"
#include "adc_manager.h"

static const char * const TAG = "robot_elbow_module";

#define ARR_LEN(x) (sizeof(x) / sizeof(*(x)))

enum {
    // CAN configuration
    CAN_TX_PIN = 5,
    CAN_RX_PIN = 4,
    CAN_BAUDRATE = 1000000,
    CAN_MSG_SIZE = 8,

    // CAN message IDs
    CAN_ID_ELBOW_STATUS = 0x030,
    CAN_ID_ELBOW_COMMAND = 0x010,

    // GPIO pin definitions
    STEPPER_STEP_PIN = GPIO_NUM_6,
    STEPPER_DIR_PIN = GPIO_NUM_7,
    STEPPER_ENABLE_PIN = GPIO_NUM_8,

    // ADC channels
    ADC_ELBOW_CHANNEL = ADC_CHANNEL_2,

    // PWM channels
    PWM_ELBOW_CHANNEL = LEDC_CHANNEL_0,
};

// Stepper configurations
const stepper_control_config_t s_elbow_stepper_cfg = {
    .step_gpio = STEPPER_STEP_PIN,
    .dir_gpio = STEPPER_DIR_PIN,
    .enable_gpio = STEPPER_ENABLE_PIN,
    .steps_per_rev = 200,
    .gear_ratio = 1.0F,
    .max_velocity_dps = 90.0F,
    .min_velocity_dps = 1.0F,
    .max_accel_dps2 = 100.0F,
    .pot_adc_channel = ADC_ELBOW_CHANNEL,
    .pwm_channel = PWM_ELBOW_CHANNEL,
    .potentiometer = (Potentiometer) {
        .degrees_of_motion = {285.F},
        .min_adc_value = 0,
        .max_adc_value = 4095,
        .min_joint_angle = {-90.F},
        .max_joint_angle = {90.F},
        .min_joint_angle_as_potentiometer_angle = {(285.F / 2.F) - 90.F},
        .max_joint_angle_as_potentiometer_angle = {(285.F / 2.F) + 90.F}
    },
};

// ADC manager configuration.
const AdcMgrChannelConfig s_adc_mgr_channel_configs[] = {
    {
        .channel = ADC_ELBOW_CHANNEL,
        .sample_rate = 1000,
    }
};

const AdcMgrConfig s_adc_mgr_config = {
    .channel_configs = s_adc_mgr_channel_configs,
    .channel_configs_len = ARR_LEN(s_adc_mgr_channel_configs),
    .ms_worth_of_buffer_size = 100,
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
// Assuming a sample rate of 1 kHz, we can comfortably keep a buffer of 1 seconds worth of data.
enum {
    ADC_ELBOW_UNDERLYING_BUF_SIZE = 1000,
};

uint16_t s_adc_elbow_channel_underlying_buffer[ADC_ELBOW_UNDERLYING_BUF_SIZE] = {0};

AdcMgrReadResults s_adc_mgr_read_results = {
    .channel_buffers = {
        [ADC_ELBOW_CHANNEL] = {
            .data = s_adc_elbow_channel_underlying_buffer,
            .capacity = ARR_LEN(s_adc_elbow_channel_underlying_buffer)
        }
    }
};

AdcMgrChannelBuffer *s_adc_mgr_elbow_buffer = &s_adc_mgr_read_results.channel_buffers[ADC_ELBOW_CHANNEL];

stepper_control_handle_t s_elbow_stepper_handle = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// CAN RX task commented out - no CAN hardware
void can_rx_task([[maybe_unused]] void *pvParameter) {
    const stepper_control_config_t* elbow_stepper_config = stepper_get_cfg(s_elbow_stepper_handle);

    uint8_t msg_rx[CAN_MSG_SIZE]; 
    uint8_t rx_len = CAN_MSG_SIZE;
    uint32_t rx_id = 0;
    
    
    while (1) {
        if (can_receive(&rx_id, msg_rx, &rx_len, portMAX_DELAY) == ESP_OK) {
            if (rx_id == CAN_ID_ELBOW_COMMAND) {
                // Simple command: first byte is angle in degrees (signed), just for testing.
                JointAngle target_angle = {*(float*)msg_rx};
                stepper_set_target_angle_deg(s_elbow_stepper_handle, to_potentiometer_angle(&elbow_stepper_config->potentiometer, target_angle));
                ESP_LOGI(TAG, "Received command: target angle = %d degrees", target_angle.degree);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}


void imu_task([[maybe_unused]] void *pvParameter) {
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

void stepper_task([[maybe_unused]] void *pvParameter) {
    const stepper_control_config_t* elbow_stepper_cfg = stepper_get_cfg(s_elbow_stepper_handle);

    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t period_ms = pdMS_TO_TICKS(10); // 10ms = 100Hz update rate
    
    while (1) {
        // FIXME: This should probably be checked instead of just assumed.
        const float dt_seconds = 0.01F; // 10ms in seconds

        adc_mgr_read(&s_adc_mgr_read_results, 0);
        stepper_update(s_elbow_stepper_handle, dt_seconds, s_adc_mgr_elbow_buffer->data, s_adc_mgr_elbow_buffer->length);
        s_adc_mgr_elbow_buffer->length = 0;
        
        // Send status over CAN and log
        static uint32_t status_counter = 0;
        enum {
            ITERATIONS_PER_LOGGING = 10,
        };
        if (++status_counter >= ITERATIONS_PER_LOGGING) { // Every 100ms
            status_counter = 0;
            // FIXME: The actual angle that gets returned is extremely delayed. When changing the potentiometer, it takes a long time for the current angle to update to the actual proper value. The filtering is probably the culprit.
            PotentiometerAngle current_pot_angle = stepper_get_current_angle_deg(s_elbow_stepper_handle);
            PotentiometerAngle target_pot_angle = stepper_get_target_angle_deg(s_elbow_stepper_handle);
            JointAngle current_angle = to_joint_angle(&elbow_stepper_cfg->potentiometer, current_pot_angle);
            JointAngle target_angle = to_joint_angle(&elbow_stepper_cfg->potentiometer, target_pot_angle);
            float velocity = stepper_get_current_velocity_dps(s_elbow_stepper_handle);
            bool moving = stepper_is_moving(s_elbow_stepper_handle);
            
            // Send status over CAN
            uint8_t can_data[CAN_MSG_SIZE] = {0};
            *(float*)can_data = current_angle.degree;
            can_send(CAN_ID_ELBOW_STATUS, can_data, CAN_MSG_SIZE);
            
            // Also log locally
            ESP_LOGI(TAG, "Stepper - Current(pot): %.2f°, Target(pot): %.2f°, Current(Joint): %.2f, Target(Joint): %.2f, Velocity: %.2f°/s, Moving: %s",
                     current_pot_angle.degree, target_pot_angle.degree, current_angle.degree, target_angle.degree, velocity, moving ? "Yes" : "No");
        }
        
        vTaskDelayUntil(&last_wake_time, period_ms);
    }
}

// Test task to cycle through different target angles
void stepper_test_task([[maybe_unused]] void *pvParameter) {
    const stepper_control_config_t *elbow_cfg = stepper_get_cfg(s_elbow_stepper_handle);

    // Wait a bit for system to initialize
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    // Test angles to cycle through (in degrees)
    const float test_angles[] = {0.0F, 30.0F, -30.0F, 45.0F, -45.0F, 0.0F};
    int num_angles = sizeof(test_angles) / sizeof(test_angles[0]);
    int angle_index = 0;
    
    ESP_LOGI(TAG, "Stepper test task started - will cycle through test angles");
    
    while (1) {
        // Set new target angle
        float target = test_angles[angle_index];
        stepper_set_target_angle_deg(s_elbow_stepper_handle, to_potentiometer_angle(&elbow_cfg->potentiometer, (JointAngle){target}));
        ESP_LOGI(TAG, ">>> Setting target angle to %.1f°", target);
        
        // Wait for stepper to reach target (or timeout after 5 seconds)
        TickType_t start_time = xTaskGetTickCount();
        while (stepper_is_moving(s_elbow_stepper_handle) && (xTaskGetTickCount() - start_time < pdMS_TO_TICKS(5000))) {
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
    can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE);
    ESP_LOGI(TAG, "CAN initialized (TX=%d, RX=%d, %d baud)", CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE);
    
    imu_config_t imu_cfg = IMU_CONFIG_DEFAULT();
    imu_cfg.sda_pin = GPIO_NUM_10;  // Custom SDA pin
    imu_cfg.scl_pin = GPIO_NUM_9;   // Custom SCL pin
    imu_init(&imu_cfg);
    ESP_LOGI(TAG, "IMU initialized (SDA=10, SCL=9)");

    if (adc_mgr_init(s_adc_mgr_config) == ESP_OK) {
        ESP_LOGI(TAG, "ADC manager initialized");
    } else {
        ESP_LOGE(TAG, "Failed to initialize ADC manager");
    }

    // Wait for a bit to get initial results.
    vTaskDelay(pdMS_TO_TICKS(10));
    adc_mgr_read(&s_adc_mgr_read_results, 0);
    
    if (stepper_init(&s_elbow_stepper_cfg, s_adc_mgr_elbow_buffer->data, s_adc_mgr_elbow_buffer->length, &s_elbow_stepper_handle) == ESP_OK) {
        ESP_LOGI(TAG, "Stepper initialized");
    } else {
        ESP_LOGE(TAG, "Failed to initialize stepper");
    }
    s_adc_mgr_elbow_buffer->length = 0;
    
    // Create FreeRTOS tasks
    enum {
        TASK_STACK_DEPTH = 4096,
        TASK_HIGH_PRIORITY = 5,
        TASK_LOW_PRIORITY = 4,
    };
    xTaskCreate(can_rx_task, "can_rx", TASK_STACK_DEPTH, NULL, TASK_HIGH_PRIORITY, NULL);
    xTaskCreate(imu_task, "imu_task", TASK_STACK_DEPTH, NULL, TASK_HIGH_PRIORITY, NULL);
    xTaskCreate(stepper_task, "stepper_task", TASK_STACK_DEPTH, NULL, TASK_HIGH_PRIORITY, NULL);
    // xTaskCreate(stepper_test_task, "stepper_test", TASK_STACK_DEPTH, NULL, TASK_LOW_PRIORITY, NULL);  // Lower priority than stepper_task

    ESP_LOGI(TAG, "Tasks created, system running");
}
