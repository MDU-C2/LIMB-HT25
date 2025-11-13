/*
 * imu_service.c
 *
 * Implements the IMU processing task, filter, and buffering.
 * This is the logic that was formerly in main.c.
 */
#include "imu_service.h" 
#include "imu_driver.h"

#include "esp_log.h"
#include <math.h>        // For the filter (atan2f, sqrtf, powf)
#include "esp_timer.h"   // For the filter (esp_timer_get_time)

// --- Filter Definitions ---
#define FILTER_ALPHA 0.98f
#define RAD_TO_DEG 57.295779513f

// --- Private Component State Variables ---
static const char *TAG = "IMU_SERVICE";

// Synchronization Primitives 
static EventGroupHandle_t s_sync_event_group;
static EventBits_t s_imu_data_ready_bit;
static SemaphoreHandle_t g_imu_buffer_mutex;

// Data Buffers 
static lsm6dso32_data_t g_imu_buffer_window[IMU_WINDOW_SIZE]; //## -- add array to read second sensor
static int g_imu_write_idx = 0;

// --- Task Prototype ---
static void imu_task(void *pvParameters);


// --- Public Function Implementation ---

SemaphoreHandle_t imu_service_get_buffer_mutex(void) {
    return g_imu_buffer_mutex;
}

void imu_service_get_window_packet(imu_data_window_t *destination_buffer) {  //## -- add the new data from sensor 2
    // This reconstruction logic was in sync_send_task,
    // now the component is responsible for it.
    int start_idx = (g_imu_write_idx - IMU_WINDOW_SIZE + IMU_WINDOW_SIZE) % IMU_WINDOW_SIZE;
    
    if (start_idx + IMU_WINDOW_SIZE <= IMU_WINDOW_SIZE) { // No wrap
        memcpy(destination_buffer->samples, &g_imu_buffer_window[start_idx], IMU_WINDOW_SIZE * sizeof(lsm6dso32_data_t));
    } else { // Wraps around
        int first_chunk_size = IMU_WINDOW_SIZE - start_idx;
        memcpy(destination_buffer->samples, &g_imu_buffer_window[start_idx], first_chunk_size * sizeof(lsm6dso32_data_t));
        memcpy(&destination_buffer->samples[first_chunk_size], g_imu_buffer_window, (IMU_WINDOW_SIZE - first_chunk_size) * sizeof(lsm6dso32_data_t));
    }
}

esp_err_t imu_service_start_task(EventGroupHandle_t event_group, EventBits_t imu_ready_bit) {
    ESP_LOGI(TAG, "Initializing IMU Module...");

    // Initialize the Hardware Driver
    if (imu_init() != ESP_OK) { 
        ESP_LOGE(TAG, "IMU driver init failed!"); 
        return ESP_FAIL; 
    }
    
    // Store synchronization primitives
    s_sync_event_group = event_group;
    s_imu_data_ready_bit = imu_ready_bit;
    
    // Create the Mutex (now owned by this component)
    g_imu_buffer_mutex = xSemaphoreCreateMutex();
    if (!g_imu_buffer_mutex) {
        ESP_LOGE(TAG, "Failed to create IMU mutex!"); 
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "IMU Mutex created.");

    // Create IMU Task
    xTaskCreate(imu_task, "imu_task", 4096, NULL, 3, NULL);
    ESP_LOGI(TAG, "IMU task created.");
    return ESP_OK;
}


// =========================================================================
// == IMU TASK 
// =========================================================================
static void imu_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Task started.");

    // --- Filter state & Calibration variables --- //## -- remember add temporal vars for callib sensor 2
    static float angle_pitch = 0.0f, angle_roll = 0.0f;
    static int64_t last_update_time_us = 0;
    const int calibration_samples = 600;
    static float gyro_bias_x = 0.0f, gyro_bias_y = 0.0f, gyro_bias_z = 0.0f;
    static float initial_pitch = 0.0f, initial_roll = 0.0f;
    float sum_gyro_x = 0.0f, sum_gyro_y = 0.0f, sum_gyro_z = 0.0f;
    float sum_pitch_accel = 0.0f, sum_roll_accel = 0.0f;
    lsm6dso32_data_t current_imu_data;

    int samples_collected = 0;

    // --- Initial Calibration --- //## -- add calibration to the sensor 2
    vTaskDelay(pdMS_TO_TICKS(500));
    for (int i = 0; i < calibration_samples; i++) {
        if (imu_read_data(&current_imu_data) == ESP_OK) { //## -- remember param to read second sensor
            sum_gyro_x += current_imu_data.gyro.x; sum_gyro_y += current_imu_data.gyro.y; sum_gyro_z += current_imu_data.gyro.z;
            float pitch_accel_calib = atan2f(-current_imu_data.accel.x, sqrtf(powf(current_imu_data.accel.y, 2) + powf(current_imu_data.accel.z, 2)));
            float roll_accel_calib = atan2f(current_imu_data.accel.y, current_imu_data.accel.z);
            sum_pitch_accel += pitch_accel_calib; sum_roll_accel += roll_accel_calib;
        } else {
            ESP_LOGE(TAG, "IMU read failed during calibration!");
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
    gyro_bias_x = sum_gyro_x / calibration_samples; gyro_bias_y = sum_gyro_y / calibration_samples; gyro_bias_z = sum_gyro_z / calibration_samples;
    initial_pitch = sum_pitch_accel / calibration_samples; 
    initial_roll = sum_roll_accel / calibration_samples;
    angle_pitch = initial_pitch; 
    angle_roll = initial_roll;
    ESP_LOGI(TAG, " Gyro Bias(r/s): X%.4f Y%.4f Z%.4f | Initial Orient(deg): P%.2f R%.2f",
             gyro_bias_x, gyro_bias_y, gyro_bias_z, initial_pitch * RAD_TO_DEG, initial_roll * RAD_TO_DEG);
    vTaskDelay(pdMS_TO_TICKS(500));
    last_update_time_us = esp_timer_get_time();
    // --- End Calibration ---

    // --- Main Task Loop ---
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10)); // ~100Hz sampling

        if (imu_read_data(&current_imu_data) == ESP_OK) { //## -- remember param to read second sensor and apply callib and CF
            // Apply Gyro Bias Correction
            current_imu_data.gyro.x -= gyro_bias_x;
            current_imu_data.gyro.y -= gyro_bias_y;
            current_imu_data.gyro.z -= gyro_bias_z;

            // Calculate dt
            int64_t now_us = esp_timer_get_time();
            float dt = (float)(now_us - last_update_time_us) / 1000000.0f;
            last_update_time_us = now_us;
            if (dt <= 0 || dt > 0.1f) dt = 0.005f; 

            // Complementary Filter Calculation
            float pitch_accel = atan2f(-current_imu_data.accel.x, sqrtf(powf(current_imu_data.accel.y, 2) + powf(current_imu_data.accel.z, 2)));
            float roll_accel = atan2f(current_imu_data.accel.y, current_imu_data.accel.z);
            float delta_pitch_gyro = current_imu_data.gyro.y * dt;
            float delta_roll_gyro = current_imu_data.gyro.x * dt; 
            angle_pitch = FILTER_ALPHA * (angle_pitch + delta_pitch_gyro) + (1.0f - FILTER_ALPHA) * pitch_accel;
            angle_roll  = FILTER_ALPHA * (angle_roll + delta_roll_gyro)  + (1.0f - FILTER_ALPHA) * roll_accel;

            // Calculate and store Relative Angles
            current_imu_data.pitch = angle_pitch - initial_pitch;
            current_imu_data.roll = angle_roll - initial_roll;

            // Store sample in global buffer
            if (xSemaphoreTake(g_imu_buffer_mutex, portMAX_DELAY) == pdTRUE) {

                g_imu_buffer_window[g_imu_write_idx] = current_imu_data; //## -- save value for sensor 2
                g_imu_write_idx = (g_imu_write_idx + 1) % IMU_WINDOW_SIZE;

                xSemaphoreGive(g_imu_buffer_mutex);

                // Check if window is full
                samples_collected++;
                if (samples_collected >= IMU_WINDOW_SIZE) {
                    samples_collected = 0;
                    // Signal the synchronization task
                    xEventGroupSetBits(s_sync_event_group, s_imu_data_ready_bit);

                    // --- DEBUG LOG (AS REQUESTED) ---
                    // Print the size of the window we just finished collecting.
                    // ESP_LOGI(TAG, "Window complete! Samples: %d (Total Size: %d bytes)",
                    //          IMU_WINDOW_SIZE, sizeof(g_imu_buffer_window));
                    // ESP_LOGI(TAG, "  Accel (X,Y,Z): %.3f, %.3f, %.3f Gyro  (X,Y,Z): %.3f, %.3f, %.3f Temp:   %.3f Angle (P,R):   %.3f, %.3f", 
                    //          current_imu_data.accel.x, 
                    //          current_imu_data.accel.y, 
                    //          current_imu_data.accel.z, 
                    //          current_imu_data.gyro.x, 
                    //          current_imu_data.gyro.y, 
                    //          current_imu_data.gyro.z,
                    //          current_imu_data.temperature, 
                    //          current_imu_data.pitch * RAD_TO_DEG, 
                    //          current_imu_data.roll * RAD_TO_DEG);
                    // --- Fin del DEBUG LOG ---
                    ESP_LOGI(TAG, " Accel Z: %.3f", current_imu_data.accel.z);
                }
            } else {
                ESP_LOGE(TAG, "CRITICAL: Mutex unavailable for IMU buffer write!");
            }
        } else {
            ESP_LOGW(TAG, "Failed to read IMU data");
        }
    } 
}