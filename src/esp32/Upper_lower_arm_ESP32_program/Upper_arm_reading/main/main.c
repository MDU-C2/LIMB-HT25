#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/twai.h" // Official CAN driver (Two-Wire Automotive Interface)

// --- Sensor drivers ---
#include "adc_emg_driver.h"
#include "imu_driver.h"

// --- RTOS Synchronization ---
#include "freertos/semphr.h"
#include "freertos/event_groups.h"
#include <math.h>
#include "esp_timer.h"

// --- CONFIGURATION ---
#define CAN_TX_PIN GPIO_NUM_7
#define CAN_RX_PIN GPIO_NUM_6

// CAN IDs: Upper arm TRANSMITS ONLY
#define CAN_ID_UPPER_EMG  0x100  // Upper EMG
#define CAN_ID_UPPER_IMU  0x101  // Upper IMU
// main.c - New definitions just in case
#define CAN_ID_UPPER_IMU_AXY  0x105 
#define CAN_ID_UPPER_IMU_AZGX 0x106 
#define CAN_ID_UPPER_IMU_GYGZ 0x107

#define FILTER_ALPHA 0.98f
#define IMU_WINDOW_SIZE 20

// --- Global Variables ---
static const char *TAG_MAIN = "MAIN";
static const char *TAG_SYNC = "SYNC";

static EventGroupHandle_t s_sync_event_group;
const int EMG_DATA_READY_BIT = BIT0;
const int IMU_DATA_READY_BIT = BIT1;
static SemaphoreHandle_t g_emg_buffer_mutex;
static SemaphoreHandle_t g_imu_buffer_mutex;

static emg_data_packet_t g_emg_buffer;
static lsm6dso32_data_t  g_imu_buffer_window[IMU_WINDOW_SIZE];
static int g_imu_write_idx = 0;


// =========================================================================
// == TASK 1: EMG (Muscle)
// =========================================================================
void emg_task(void *pvParameters) {
    while (1) {
        emg_driver_process_data(); // Wait for DMA to complete
        if (emg_driver_is_window_ready()) {
            if (xSemaphoreTake(g_emg_buffer_mutex, portMAX_DELAY) == pdTRUE) {
                emg_driver_get_packet(&g_emg_buffer);
                xSemaphoreGive(g_emg_buffer_mutex);
                xEventGroupSetBits(s_sync_event_group, EMG_DATA_READY_BIT);
            }
        }
    }
}

// =========================================================================
// == TASK 2: IMU (Mouvement)
// =========================================================================
// For now we are only sending accel x and y via CAN due to CAN bus limitations 
void imu_task(void *pvParameters) {
    // Filter variables
    float angle_pitch = 0.0f, angle_roll = 0.0f;
    static int64_t last_update_time_us = 0;
    lsm6dso32_data_t current_imu_data;
    int samples_collected = 0;

    // Warm-up calibration at startup
    vTaskDelay(pdMS_TO_TICKS(500));
    last_update_time_us = esp_timer_get_time();

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(20)); // ~200Hz sampling

        if (imu_read_data(&current_imu_data) == ESP_OK) {
            // Calculate elapsed time (dt)
            int64_t now_us = esp_timer_get_time();
            float dt = (float)(now_us - last_update_time_us) / 1000000.0f;
            last_update_time_us = now_us;
            if (dt <= 0) dt = 0.005f;

            // Complementary filter (Magic Formula)
            float pitch_accel = atan2f(-current_imu_data.accel.x, sqrtf(powf(current_imu_data.accel.y, 2) + powf(current_imu_data.accel.z, 2)));
            float roll_accel = atan2f(current_imu_data.accel.y, current_imu_data.accel.z);
            
            angle_pitch = FILTER_ALPHA * (angle_pitch + current_imu_data.gyro.y * dt) + (1.0f - FILTER_ALPHA) * pitch_accel;
            angle_roll  = FILTER_ALPHA * (angle_roll + current_imu_data.gyro.x * dt)  + (1.0f - FILTER_ALPHA) * roll_accel;

            current_imu_data.pitch = angle_pitch;
            current_imu_data.roll = angle_roll;

            // Thread-safe storage
            if (xSemaphoreTake(g_imu_buffer_mutex, portMAX_DELAY) == pdTRUE) {
                g_imu_buffer_window[g_imu_write_idx] = current_imu_data;
                g_imu_write_idx = (g_imu_write_idx + 1) % IMU_WINDOW_SIZE;
                xSemaphoreGive(g_imu_buffer_mutex);

                samples_collected++;
                if (samples_collected >= IMU_WINDOW_SIZE) {
                    samples_collected = 0;
                    xEventGroupSetBits(s_sync_event_group, IMU_DATA_READY_BIT);
                }
            }
        }
    }
}

// =========================================================================
// == TASK 3: SYNCHRO & SEND OF CAN (MODIFIED)
// =========================================================================
void sync_send_task(void *pvParameters) {
    const EventBits_t bits_to_wait_for = (EMG_DATA_READY_BIT | IMU_DATA_READY_BIT);
    
    while (1) {
        // Wait for both sensor data to be ready
        EventBits_t uxBits = xEventGroupWaitBits(s_sync_event_group, bits_to_wait_for, pdTRUE, pdTRUE, portMAX_DELAY);

        if((uxBits & bits_to_wait_for) == bits_to_wait_for) {
            
            // --- 1. Retrieve sensor data ---
            xSemaphoreTake(g_emg_buffer_mutex, portMAX_DELAY);
            float emg_val = (float)g_emg_buffer.emg_ch0_window[0];
            xSemaphoreGive(g_emg_buffer_mutex);

            xSemaphoreTake(g_imu_buffer_mutex, portMAX_DELAY);
            int last_idx = (g_imu_write_idx - 1 + IMU_WINDOW_SIZE) % IMU_WINDOW_SIZE;
            
            // Extraction des 6 valeurs float (pour les envois complets 0x105-0x107)
            float ax = g_imu_buffer_window[last_idx].accel.x;
            float ay = g_imu_buffer_window[last_idx].accel.y;
            float az = g_imu_buffer_window[last_idx].accel.z;
            float gx = g_imu_buffer_window[last_idx].gyro.x;
            float gy = g_imu_buffer_window[last_idx].gyro.y;
            float gz = g_imu_buffer_window[last_idx].gyro.z;
            
            xSemaphoreGive(g_imu_buffer_mutex);
            // Les variables accel_x et accel_y utilisées ci-dessous sont maintenant ax et ay pour plus de clarté.


            // --- 2. Transmit EMG on 0x100 (UNCHANGED) ---
            twai_message_t msg_emg;
            msg_emg.identifier = CAN_ID_UPPER_EMG;
            msg_emg.extd = 0;
            msg_emg.data_length_code = 4;
            memcpy(&msg_emg.data[0], &emg_val, 4);
            if (twai_transmit(&msg_emg, pdMS_TO_TICKS(10)) != ESP_OK) {
                ESP_LOGE(TAG_MAIN, "TX EMG failed");
            }

            // --- 3. Transmit IMU on 0x101 (KEEPING THE ORIGINAL INCOMPLETE MESSAGE) ---
            // C'est l'ancien message qui n'envoyait que 2 floats et provoquait une erreur côté parseur Python
            twai_message_t msg_imu_legacy;
            msg_imu_legacy.identifier = CAN_ID_UPPER_IMU; // 0x101
            msg_imu_legacy.extd = 0;
            msg_imu_legacy.data_length_code = 8;
            memcpy(&msg_imu_legacy.data[0], &ax, 4); // Ancien accel_x
            memcpy(&msg_imu_legacy.data[4], &ay, 4); // Ancien accel_y
            if (twai_transmit(&msg_imu_legacy, pdMS_TO_TICKS(10)) != ESP_OK) {
                ESP_LOGE(TAG_MAIN, "TX IMU 0x101 failed");
            }
            
            // --- 4. Transmit COMPLETE IMU data via 0x105, 0x106, 0x107 ---
            
            // 4.1 Transmit IMU AXY on 0x105 (Accel X, Accel Y)
            twai_message_t msg_imu_0x105;
            msg_imu_0x105.identifier = CAN_ID_UPPER_IMU_AXY;
            msg_imu_0x105.extd = 0;
            msg_imu_0x105.data_length_code = 8;
            memcpy(&msg_imu_0x105.data[0], &ax, 4);
            memcpy(&msg_imu_0x105.data[4], &ay, 4);
            if (twai_transmit(&msg_imu_0x105, pdMS_TO_TICKS(10)) != ESP_OK) {
                ESP_LOGE(TAG_MAIN, "TX IMU 0x105 failed");
            }

            // 4.2 Transmit IMU AZGX on 0x106 (Accel Z, Gyro X)
            twai_message_t msg_imu_0x106;
            msg_imu_0x106.identifier = CAN_ID_UPPER_IMU_AZGX;
            msg_imu_0x106.extd = 0;
            msg_imu_0x106.data_length_code = 8;
            memcpy(&msg_imu_0x106.data[0], &az, 4);
            memcpy(&msg_imu_0x106.data[4], &gx, 4);
            if (twai_transmit(&msg_imu_0x106, pdMS_TO_TICKS(10)) != ESP_OK) {
                ESP_LOGE(TAG_MAIN, "TX IMU 0x106 failed");
            }

            // 4.3 Transmit IMU GYGZ on 0x107 (Gyro Y, Gyro Z)
            twai_message_t msg_imu_0x107;
            msg_imu_0x107.identifier = CAN_ID_UPPER_IMU_GYGZ;
            msg_imu_0x107.extd = 0;
            msg_imu_0x107.data_length_code = 8;
            memcpy(&msg_imu_0x107.data[0], &gy, 4);
            memcpy(&msg_imu_0x107.data[4], &gz, 4);
            if (twai_transmit(&msg_imu_0x107, pdMS_TO_TICKS(10)) != ESP_OK) {
                ESP_LOGE(TAG_MAIN, "TX IMU 0x107 failed");
            }
        }
    }
}


// =========================================================================
// == APP_MAIN
// =========================================================================
void app_main(void) {
    // 1. Initialize CAN Bus
    ESP_LOGI(TAG_MAIN, "Init CAN Bus...");
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
    twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

    if (twai_driver_install(&g_config, &t_config, &f_config) == ESP_OK) {
        ESP_LOGI(TAG_MAIN, "CAN driver installed");
    } else {
        ESP_LOGE(TAG_MAIN, "CAN driver installation failed");
        return;
    }
    if (twai_start() == ESP_OK) {
        ESP_LOGI(TAG_MAIN, "CAN driver started");
    } else {
        ESP_LOGE(TAG_MAIN, "CAN driver startup failed");
        return;
    }

    // 2. Initialize Sensor Drivers
    if (imu_init() != ESP_OK) { ESP_LOGE(TAG_MAIN, "IMU initialization failed"); return; }
    if (emg_driver_init() != ESP_OK) { ESP_LOGE(TAG_MAIN, "EMG driver initialization failed"); return; }

    // 3. Initialize RTOS Synchronization
    s_sync_event_group = xEventGroupCreate();
    g_emg_buffer_mutex = xSemaphoreCreateMutex();
    g_imu_buffer_mutex = xSemaphoreCreateMutex();

    // 4. Create and start tasks
    TaskHandle_t emg_task_handle = NULL;
    xTaskCreate(sync_send_task, "sync", 4096, NULL, 4, NULL);
    xTaskCreate(imu_task, "imu", 4096, NULL, 3, NULL);
    xTaskCreate(emg_task, "emg", 4096, NULL, 6, &emg_task_handle);

    // 5. Start EMG driver (task must exist first)
    emg_driver_set_notify_task(emg_task_handle);
    emg_driver_start();

    ESP_LOGI(TAG_MAIN, "Upper ARM Ready - Transmitting EMG & IMU via CAN");
}