/**
 * @file main.c
 * @brief Main application for Human Arm project, implementing a producer-consumer architecture
 * with EMG and IMU data acquisition, synchronization.
 *
 * Tasks:
 * 1. emg_task (Producer): Acquires EMG data windows.
 * 2. imu_task (Producer): Acquires IMU data, applies filter, buffers window.
 * 3. sync_send_task (Consumer): Synchronizes EMG/IMU data .
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h" // Still needed for initial logs and critical error logs if uncommented later

// --- Project Includes ---
#include "adc_emg_driver.h"
#include "imu_driver.h"

#include "limb_ble_periph.h" 
#include "nvs_flash.h"
#include "sensors_service.h"

// --- RTOS Synchronization ---
#include "freertos/semphr.h"      // For Mutexes
#include "freertos/event_groups.h" // For Event Groups

// --- Sensor Fusion Includes & Defines ---
#include <math.h>         // For atan2f, sqrtf, powf
#include "esp_timer.h"   // For high-resolution timer (esp_timer_get_time)

#define FILTER_ALPHA 0.98f        // Complementary filter coefficient (Gyro trust)
#define RAD_TO_DEG   57.295779513f // Conversion factor: Radians to Degrees

// --- Data Structure Definitions ---
#define IMU_WINDOW_SIZE 20 // Number of IMU samples per data window (matches 100ms at 200Hz)

/**
 * @brief Structure to hold a complete data packet (EMG + IMU).
 * Ensure this structure size is consistent with the Python receiver script.
 */
typedef struct {
    emg_data_packet_t emg_data;                   // EMG data window
    lsm6dso32_data_t  imu_window[IMU_WINDOW_SIZE]; // IMU data window (raw + relative angles)
} combined_packet_t;

typedef struct {
    float pitch;
    float roll;
} imu_angles_t;


// --- Global Variables ---
// Logging Tags (still defined, used in initial logs)
static const char *TAG_EMG  = "EMG_TASK";
static const char *TAG_IMU  = "IMU_TASK";
static const char *TAG_SYNC = "SYNC_TASK";
static const char *TAG_MAIN = "APP_MAIN";

// Synchronization Primitives
static EventGroupHandle_t s_sync_event_group;   // Event group for task coordination
const int EMG_DATA_READY_BIT = BIT0;            // Bit flag for EMG window ready
const int IMU_DATA_READY_BIT = BIT1;            // Bit flag for IMU window ready
static SemaphoreHandle_t g_emg_buffer_mutex;    // Mutex for EMG buffer access
static SemaphoreHandle_t g_imu_buffer_mutex;    // Mutex for IMU buffer access

// Data Buffers
static emg_data_packet_t g_emg_buffer;                       // Global buffer for one EMG window
static lsm6dso32_data_t  g_imu_buffer_window[IMU_WINDOW_SIZE]; // Global circular buffer for IMU window
static int g_imu_write_idx = 0;                                // Write index for IMU circular buffer



// =========================================================================
// == TASK 1: EMG PRODUCER
// Acquires EMG data windows and signals readiness via Event Group.
// =========================================================================
void emg_task(void *pvParameters)
{
    ESP_LOGI(TAG_EMG, "Task started."); // Initial log OK
    while (1) {
        // Wait for notification from ADC driver (ISR)
        emg_driver_process_data();

        if (emg_driver_is_window_ready()) {
            if (xSemaphoreTake(g_emg_buffer_mutex, portMAX_DELAY) == pdTRUE) {
                emg_driver_get_packet(&g_emg_buffer);
                xSemaphoreGive(g_emg_buffer_mutex);
                // Signal synchronization task
                xEventGroupSetBits(s_sync_event_group, EMG_DATA_READY_BIT);
            }
            else { // Removed empty else block
                //  Critical error handling for mutex failure could go here if needed
                 ESP_LOGE(TAG_EMG, "Failed to take EMG mutex!"); // SILENCED
            }
        }
    }
}


// =========================================================================
// == TASK 2: IMU PRODUCER
// Reads IMU, calibrates gyro, calculates orientation via complementary filter,
// buffers the window, and signals readiness via Event Group.
// =========================================================================
void imu_task(void *pvParameters)
{
    ESP_LOGI(TAG_IMU, "Task started."); // Initial log OK

    // Filter state & Calibration variables
    static float angle_pitch = 0.0f, angle_roll = 0.0f;
    static int64_t last_update_time_us = 0;
    const int calibration_samples = 600;
    static float gyro_bias_x = 0.0f, gyro_bias_y = 0.0f, gyro_bias_z = 0.0f;
    static float initial_pitch = 0.0f, initial_roll = 0.0f;
    float sum_gyro_x = 0.0f, sum_gyro_y = 0.0f, sum_gyro_z = 0.0f;
    float sum_pitch_accel = 0.0f, sum_roll_accel = 0.0f;
    lsm6dso32_data_t current_imu_data;
    int samples_collected = 0;

    // --- Initial Calibration ---
    ESP_LOGW(TAG_IMU, "Starting IMU calibration: Keep sensor STILL..."); // Initial log OK
    vTaskDelay(pdMS_TO_TICKS(500));
    for (int i = 0; i < calibration_samples; i++) { // Calibration loop
        if (imu_read_data(&current_imu_data) == ESP_OK) {
            sum_gyro_x += current_imu_data.gyro.x; sum_gyro_y += current_imu_data.gyro.y; sum_gyro_z += current_imu_data.gyro.z;
            float pitch_accel_calib = atan2f(-current_imu_data.accel.x, sqrtf(powf(current_imu_data.accel.y, 2) + powf(current_imu_data.accel.z, 2)));
            float roll_accel_calib = atan2f(current_imu_data.accel.y, current_imu_data.accel.z);
            sum_pitch_accel += pitch_accel_calib; sum_roll_accel += roll_accel_calib;
        } else {
            ESP_LOGE(TAG_IMU, "IMU read failed during calibration!"); // Keep error log during init
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
    // Calculate and store results
    gyro_bias_x = sum_gyro_x / calibration_samples; gyro_bias_y = sum_gyro_y / calibration_samples; gyro_bias_z = sum_gyro_z / calibration_samples;
    initial_pitch = sum_pitch_accel / calibration_samples; initial_roll = sum_roll_accel / calibration_samples;
    angle_pitch = initial_pitch; angle_roll = initial_roll;
    ESP_LOGW(TAG_IMU, "Calibration complete."); // Initial log OK
    ESP_LOGI(TAG_IMU, " Gyro Bias(r/s): X%.4f Y%.4f Z%.4f | Initial Orient(deg): P%.2f R%.2f",
             gyro_bias_x, gyro_bias_y, gyro_bias_z, initial_pitch * RAD_TO_DEG, initial_roll * RAD_TO_DEG); // Initial log OK
    vTaskDelay(pdMS_TO_TICKS(500));
    last_update_time_us = esp_timer_get_time();
    // --- End Calibration ---

    // --- Main Task Loop ---
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5)); // ~200Hz sampling

        if (imu_read_data(&current_imu_data) == ESP_OK) {
            // Apply Gyro Bias Correction
            current_imu_data.gyro.x -= gyro_bias_x;
            current_imu_data.gyro.y -= gyro_bias_y;
            current_imu_data.gyro.z -= gyro_bias_z;

            // Calculate dt
            int64_t now_us = esp_timer_get_time();
            float dt = (float)(now_us - last_update_time_us) / 1000000.0f;
            last_update_time_us = now_us;
            if (dt <= 0 || dt > 0.1f) dt = 0.005f;

            // Complementary Filter Calculation (absolute angles)
            float pitch_accel = atan2f(-current_imu_data.accel.x, sqrtf(powf(current_imu_data.accel.y, 2) + powf(current_imu_data.accel.z, 2)));
            float roll_accel = atan2f(current_imu_data.accel.y, current_imu_data.accel.z);
            float delta_pitch_gyro = current_imu_data.gyro.y * dt; // Verify Axis Mapping!
            float delta_roll_gyro = current_imu_data.gyro.x * dt;  // Verify Axis Mapping!
            angle_pitch = FILTER_ALPHA * (angle_pitch + delta_pitch_gyro) + (1.0f - FILTER_ALPHA) * pitch_accel;
            angle_roll  = FILTER_ALPHA * (angle_roll + delta_roll_gyro)  + (1.0f - FILTER_ALPHA) * roll_accel;

            // Calculate and store Relative Angles
            current_imu_data.pitch = angle_pitch - initial_pitch;
            current_imu_data.roll = angle_roll - initial_roll;

            // Store sample in global buffer
            if (xSemaphoreTake(g_imu_buffer_mutex, portMAX_DELAY) == pdTRUE) {
                g_imu_buffer_window[g_imu_write_idx] = current_imu_data;
                g_imu_write_idx = (g_imu_write_idx + 1) % IMU_WINDOW_SIZE;
                xSemaphoreGive(g_imu_buffer_mutex);

                // Check if window is full
                samples_collected++;
                if (samples_collected >= IMU_WINDOW_SIZE) {
                    samples_collected = 0;
                    // Signal the synchronization task
                    xEventGroupSetBits(s_sync_event_group, IMU_DATA_READY_BIT);
                } // End if window full
            }
            else { // Removed empty else block
                 ESP_LOGE(TAG_IMU, "CRITICAL: Mutex unavailable for IMU buffer write!"); // SILENCED
            }
        }
        else { // Removed empty else block
             ESP_LOGW(TAG_IMU, "Failed to read IMU data"); // SILENCED
        }
    } // End main loop (while(1))
} // End imu_task


// =========================================================================
// == TASK 3: CONSUMER (
// Waits for both EMG and IMU windows, combines them.
// =========================================================================
void sync_send_task(void *pvParameters)
{
    ESP_LOGI(TAG_SYNC, "Task started."); // Initial log OK

    static combined_packet_t local_packet;
    int imu_start_idx;
    const EventBits_t bits_to_wait_for = (EMG_DATA_READY_BIT | IMU_DATA_READY_BIT);

    CharacteristicBuffer emg_buffer   = get_emg_buf(); 
    CharacteristicBuffer imu_buffer   = get_imu_buf(); 
    // CharacteristicBuffer piezo_buffer = get_piezo_buf(); 

    static imu_angles_t imu_data_to_send;

    while (1) {
        // 1. Wait until both producers signal data is ready
        EventBits_t uxBits = xEventGroupWaitBits(s_sync_event_group, bits_to_wait_for, pdTRUE, pdTRUE, portMAX_DELAY);

        // Check if both bits were set before proceeding
        if((uxBits & bits_to_wait_for) == bits_to_wait_for) {

            // 2. Copy data from global buffers
            // Copy EMG data
            if (xSemaphoreTake(g_emg_buffer_mutex, portMAX_DELAY) == pdTRUE) {
                memcpy(&local_packet.emg_data, &g_emg_buffer, sizeof(emg_data_packet_t));
                xSemaphoreGive(g_emg_buffer_mutex);
            } else {
                 ESP_LOGE(TAG_SYNC, "Failed to take EMG mutex! Skipping packet."); // SILENCED
                 continue; // Skip packet if failed to get data
            }

            // Copy IMU data
            if (xSemaphoreTake(g_imu_buffer_mutex, portMAX_DELAY) == pdTRUE) {
                imu_start_idx = (g_imu_write_idx - IMU_WINDOW_SIZE + IMU_WINDOW_SIZE) % IMU_WINDOW_SIZE;
                 if (imu_start_idx + IMU_WINDOW_SIZE <= IMU_WINDOW_SIZE) { // No wrap
                     memcpy(local_packet.imu_window, &g_imu_buffer_window[imu_start_idx], IMU_WINDOW_SIZE * sizeof(lsm6dso32_data_t));
                 } else { // Wraps around
                     int first_chunk_size = IMU_WINDOW_SIZE - imu_start_idx;
                     memcpy(local_packet.imu_window, &g_imu_buffer_window[imu_start_idx], first_chunk_size * sizeof(lsm6dso32_data_t));
                     memcpy(&local_packet.imu_window[first_chunk_size], g_imu_buffer_window, (IMU_WINDOW_SIZE - first_chunk_size) * sizeof(lsm6dso32_data_t));
                 }
                xSemaphoreGive(g_imu_buffer_mutex);
            } else {
                 ESP_LOGE(TAG_SYNC, "Failed to take IMU mutex! Skipping packet."); // SILENCED
                 continue; // Skip packet if failed to get data
            }

            memcpy(emg_buffer.data, &local_packet.emg_data.emg_ch0_window[0], sizeof(uint16_t));
            imu_data_to_send.pitch = local_packet.imu_window[0].pitch;
            imu_data_to_send.roll = local_packet.imu_window[0].roll;
            memcpy(imu_buffer.data, &imu_data_to_send, sizeof(imu_angles_t));

            TryNotifyEmgSubscribers();
            TryNotifyImuSubscribers();

            // 3. Send the combined packet


            ESP_LOGI(TAG_SYNC, "Packet Ready. IMU: P=%.2f R=%.2f | EMG: CH0=%d",
                     imu_data_to_send.pitch * RAD_TO_DEG, 
                     imu_data_to_send.roll * RAD_TO_DEG,     
                     local_packet.emg_data.emg_ch0_window[0]);
        }
        else { // Removed empty else block
             ESP_LOGE(TAG_SYNC, "Event group wait returned unexpected bits: %x", uxBits); // SILENCED
        }
    } // End while(1)
} // End sync_send_task


// =========================================================================
// == APP_MAIN: Entry Point, Initialization, and Task Creation
// =========================================================================
void app_main(void)
{
    ESP_LOGI(TAG_MAIN, "Starting APP_MAIN..."); // Initial log OK
    // Initialize Non-Volatile Storage (required by BLE)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    xTaskCreate(BleTask, "ble_task", 4096, NULL, 5, NULL);

    // Initialize Sensor Drivers
    ESP_LOGI(TAG_MAIN, "Initializing IMU Module..."); // Initial log OK
    if (imu_init() != ESP_OK) { ESP_LOGE(TAG_MAIN, "IMU Init Failed!"); return; } // Keep error log

    ESP_LOGI(TAG_MAIN, "Initializing EMG Module..."); // Initial log OK
    if (emg_driver_init() != ESP_OK) { ESP_LOGE(TAG_MAIN, "EMG Init Failed!"); return; } // Keep error log

    // Initialize RTOS Synchronization Primitives
    s_sync_event_group = xEventGroupCreate();
    g_emg_buffer_mutex = xSemaphoreCreateMutex();
    g_imu_buffer_mutex = xSemaphoreCreateMutex();
    if (!s_sync_event_group || !g_emg_buffer_mutex || !g_imu_buffer_mutex) {
        ESP_LOGE(TAG_MAIN, "Failed to create sync primitives!"); return; // Keep error log
    }
    ESP_LOGI(TAG_MAIN, "Mutexes and Event Group created."); // Initial log OK

    // Create Application Tasks
    TaskHandle_t emg_task_handle = NULL;

    xTaskCreate(sync_send_task, "sync_task", 4096, NULL, 4, NULL);
    xTaskCreate(imu_task, "imu_task", 4096, NULL, 3, NULL);
    xTaskCreate(emg_task, "emg_task", 4096, NULL, 6, &emg_task_handle);

    if (!emg_task_handle) { ESP_LOGE(TAG_MAIN, "Failed to create EMG task!"); return; } // Keep error log
    ESP_LOGI(TAG_MAIN, "All tasks created."); // Initial log OK

    // Configure and Start ADC/EMG Acquisition
    emg_driver_set_notify_task(emg_task_handle);
    ESP_LOGI(TAG_MAIN, "EMG Task Handle configured in driver."); // Initial log OK
    if (emg_driver_start() != ESP_OK) { ESP_LOGE(TAG_MAIN, "Failed to start EMG driver!"); return; } // Keep error log
    ESP_LOGI(TAG_MAIN, "EMG Driver started."); // Initial log OK

    ESP_LOGI(TAG_MAIN, "app_main finished setup. All tasks running."); // Initial log OK
    // FreeRTOS scheduler now manages the tasks. app_main exits.
}