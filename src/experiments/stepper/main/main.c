#include <stdio.h>
#include <math.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/task.h"
#include "portmacro.h"
#include "stepper.h"
#include "driver/gpio.h"
#include "hal/adc_types.h"
#include "adc_manager.h"

static const char *TAG = "STEPPER_TEST";

// Configuration - adjust these GPIO pins and parameters for your hardware
#define STEP_GPIO          GPIO_NUM_6   // STEP pin
#define DIR_GPIO           GPIO_NUM_7   // DIR pin (or GPIO_NUM_NC if not used)
#define ENABLE_GPIO        GPIO_NUM_8   // ENABLE pin (or GPIO_NUM_NC if not used)
#define POT_ADC_CHANNEL    ADC_CHANNEL_2  // Potentiometer ADC channel (or ADC_CHANNEL_MAX if not used)

// Motor parameters - adjust for your stepper motor
#define STEPS_PER_REV      200          // 1.8° stepper (200 steps/rev)
#define GEAR_RATIO         10.0f       // 10:1 gear reduction
#define MAX_VELOCITY_DPS   90.0f       // Maximum velocity (degrees/second)
#define MIN_VELOCITY_DPS   5.0f        // Minimum velocity (degrees/second)
#define MAX_ACCEL_DPS2     180.0f      // Maximum acceleration (degrees/second²)

// Test parameters
#define UPDATE_PERIOD_MS   10          // Control loop update period (10ms = 100Hz)
#define STATUS_PERIOD_MS   500          // Status print period (500ms)

#define AVG_SAMPLES 5           // Number of ADC samples to average

stepper_control_handle_t s_stepper_handle;

adc_mgr_handle_t adc_handle;

SemaphoreHandle_t s_latest_potentiometer_mutex;
uint16_t s_latest_potentiometer_values[1024];
uint16_t s_latest_potentiometer_values_len;

static void adc_reading_task(void *pvParameters)
{
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(UPDATE_PERIOD_MS);
    while (1) {
        // Get potentiometer samples.
        xSemaphoreTake(s_latest_potentiometer_mutex, portMAX_DELAY);
        s_latest_potentiometer_values_len = 0;
        for (int i = 0; i < AVG_SAMPLES; ++i) {
            int raw = 0;
            if (adc_mgr_read(adc_handle, &raw) == ESP_OK) {
                s_latest_potentiometer_values[s_latest_potentiometer_values_len++] = raw;
            }
        }
        xSemaphoreGive(s_latest_potentiometer_mutex);

        vTaskDelayUntil(&last_wake_time, period);
    }
    
}

/**
 * @brief Control loop task - updates stepper control periodically
 */
static void stepper_control_task(void *pvParameters)
{
    stepper_control_handle_t *stepper_handle = pvParameters;
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(UPDATE_PERIOD_MS);
    float dt = UPDATE_PERIOD_MS / 1000.0f; // Convert to seconds

    ESP_LOGI(TAG, "Control task started (update period: %d ms)", UPDATE_PERIOD_MS);

    while (1) {
        xSemaphoreTake(s_latest_potentiometer_mutex, portMAX_DELAY);
        // Update stepper control loop
        stepper_update(*stepper_handle, dt, s_latest_potentiometer_values, s_latest_potentiometer_values_len);
        xSemaphoreGive(s_latest_potentiometer_mutex);

        // Wait for next period
        vTaskDelayUntil(&last_wake_time, period);
    }
}

/**
 * @brief Status monitoring task - prints stepper status periodically
 */
static void status_task(void *pvParameters)
{
    stepper_control_handle_t *stepper_handle = pvParameters;
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(STATUS_PERIOD_MS);

    ESP_LOGI(TAG, "Status task started (print period: %d ms)", STATUS_PERIOD_MS);

    while (1) {
        // Get current status
        float current_angle = stepper_get_current_angle_deg(*stepper_handle);
        float target_angle = stepper_get_target_angle_deg(*stepper_handle);
        float velocity = stepper_get_current_velocity_dps(*stepper_handle);
        bool is_moving = stepper_is_moving(*stepper_handle);
        bool has_feedback = stepper_has_position_feedback(*stepper_handle);

        // Print status
        ESP_LOGI(TAG, "Status: current=%.2f°, target=%.2f°, vel=%.2f°/s, moving=%d, feedback=%d",
                 current_angle, target_angle, velocity, is_moving, has_feedback);

        // Wait for next period
        vTaskDelayUntil(&last_wake_time, period);
    }
}

/**
 * @brief Test sequence task - performs various test movements
 */
static void test_sequence_task(void *pvParameters)
{
    stepper_control_handle_t *stepper_handle = pvParameters;

    // Wait a bit for initialization
    vTaskDelay(pdMS_TO_TICKS(1000));

    ESP_LOGI(TAG, "=== Starting Stepper Test Sequence ===");

    // Test 1: Check if position feedback is available
    bool has_feedback = stepper_has_position_feedback(*stepper_handle);
    ESP_LOGI(TAG, "Test 1: Position feedback available: %s", has_feedback ? "YES" : "NO");
    if (has_feedback) {
        float initial_angle = stepper_get_current_angle_deg(*stepper_handle);
        ESP_LOGI(TAG, "  Initial angle: %.2f°", initial_angle);
    }
    vTaskDelay(pdMS_TO_TICKS(2000));

    // Test 2: Move to positive angle
    ESP_LOGI(TAG, "Test 2: Moving to +45°");
    stepper_set_target_angle_deg(*stepper_handle, 45.0f);
    vTaskDelay(pdMS_TO_TICKS(5000)); // Wait for movement

    // Test 3: Move to negative angle
    ESP_LOGI(TAG, "Test 3: Moving to -45°");
    stepper_set_target_angle_deg(*stepper_handle, -45.0f);
    vTaskDelay(pdMS_TO_TICKS(5000)); // Wait for movement

    // Test 4: Move to zero
    ESP_LOGI(TAG, "Test 4: Moving to 0°");
    stepper_set_target_angle_deg(*stepper_handle, 0.0f);
    vTaskDelay(pdMS_TO_TICKS(5000)); // Wait for movement

    // Test 5: Test emergency stop
    ESP_LOGI(TAG, "Test 5: Testing emergency stop");
    stepper_set_target_angle_deg(*stepper_handle, 30.0f);
    vTaskDelay(pdMS_TO_TICKS(1000)); // Start moving
    ESP_LOGI(TAG, "  Activating E-stop...");
    stepper_set_estop(*stepper_handle, true);
    vTaskDelay(pdMS_TO_TICKS(2000));
    ESP_LOGI(TAG, "  Releasing E-stop...");
    stepper_set_estop(*stepper_handle, false);
    vTaskDelay(pdMS_TO_TICKS(2000));

    // Test 6: Continuous movement test
    ESP_LOGI(TAG, "Test 6: Continuous movement test (sine wave pattern)");
    for (int i = 0; i < 20; i++) {
        float angle = 60.0f * sinf(i * 0.314f); // ±60° sine wave
        stepper_set_target_angle_deg(*stepper_handle, angle);
        ESP_LOGI(TAG, "  Target: %.2f°", angle);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // Test 7: Return to center
    ESP_LOGI(TAG, "Test 7: Returning to center (0°)");
    stepper_set_target_angle_deg(*stepper_handle, 0.0f);
    vTaskDelay(pdMS_TO_TICKS(5000));

    ESP_LOGI(TAG, "=== Test Sequence Complete ===");
    ESP_LOGI(TAG, "Stepper will continue running. Check status logs for current state.");

    // Keep running - just monitor status
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "Stepper Motor Test Application");
    ESP_LOGI(TAG, "==============================");

    // Register channel with ADC manager
    adc_oneshot_chan_cfg_t chan_config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = ADC_ATTEN_DB_12,
    };
    
    adc_handle = adc_mgr_register_channel(POT_ADC_CHANNEL, &chan_config);
    if (adc_handle < 0) {
        ESP_LOGE(TAG, "Failed to register ADC channel with ADC manager");
        return;
    }

    s_latest_potentiometer_mutex = xSemaphoreCreateMutex();
    if (s_latest_potentiometer_mutex == NULL) {
        ESP_LOGE(TAG, "Couldn't allocate mutex");
        return;
    }

    // Collect initial ADC readings for potentiometer.
    xSemaphoreTake(s_latest_potentiometer_mutex, portMAX_DELAY);
    for (int i = 0; i < 10; ++i) {
        int raw = 0;
        if (adc_mgr_read(adc_handle, &raw) == ESP_OK) {
            s_latest_potentiometer_values[s_latest_potentiometer_values_len++] = raw;
        }
    }
    xSemaphoreGive(s_latest_potentiometer_mutex);
    
    // Configure stepper motor
    stepper_control_config_t config = {
        .step_gpio = STEP_GPIO,
        .dir_gpio = DIR_GPIO,
        .enable_gpio = ENABLE_GPIO,
        .steps_per_rev = STEPS_PER_REV,
        .gear_ratio = GEAR_RATIO,
        .max_velocity_dps = MAX_VELOCITY_DPS,
        .min_velocity_dps = MIN_VELOCITY_DPS,
        .max_accel_dps2 = MAX_ACCEL_DPS2,
        .pot_adc_channel = POT_ADC_CHANNEL,
    };

    // Initialize stepper
    esp_err_t ret = stepper_init(&config, s_latest_potentiometer_values, s_latest_potentiometer_values_len, &s_stepper_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize stepper: %s", esp_err_to_name(ret));
        return;
    }

    ESP_LOGI(TAG, "Stepper initialized successfully");
    ESP_LOGI(TAG, "Configuration:");
    ESP_LOGI(TAG, "  STEP GPIO: %d", STEP_GPIO);
    ESP_LOGI(TAG, "  DIR GPIO: %d", DIR_GPIO);
    ESP_LOGI(TAG, "  ENABLE GPIO: %d", ENABLE_GPIO);
    ESP_LOGI(TAG, "  Steps/rev: %d", STEPS_PER_REV);
    ESP_LOGI(TAG, "  Gear ratio: %.1f:1", GEAR_RATIO);
    ESP_LOGI(TAG, "  Max velocity: %.1f°/s", MAX_VELOCITY_DPS);
    ESP_LOGI(TAG, "  Min velocity: %.1f°/s", MIN_VELOCITY_DPS);
    ESP_LOGI(TAG, "  Max acceleration: %.1f°/s²", MAX_ACCEL_DPS2);
    ESP_LOGI(TAG, "  ADC channel: %d", POT_ADC_CHANNEL);

    // Create control task (high priority, runs at 100Hz)
    xTaskCreate(stepper_control_task, "stepper_ctrl", 4096, &s_stepper_handle, 5, NULL);

    // Create status monitoring task (low priority)
    xTaskCreate(status_task, "status", 2048, &s_stepper_handle, 1, NULL);

    // Create test sequence task (medium priority)
    xTaskCreate(test_sequence_task, "test_seq", 4096, &s_stepper_handle, 3, NULL);

    xTaskCreate(adc_reading_task, "adc_read", 4096, NULL, 4, NULL);

    ESP_LOGI(TAG, "All tasks created. Test sequence will start in 1 second...");
}

