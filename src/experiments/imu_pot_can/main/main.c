#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "imu.h"
#include "potentiometer.h"
#include "driver/twai.h"

static const char *TAG = "IMU_POT_CAN";

// IMU configuration
#define IMU_I2C_PORT       I2C_NUM_0
#define IMU_SDA_PIN        10
#define IMU_SCL_PIN        9
#define IMU_I2C_FREQ_HZ    400000
#define IMU_SENSOR_ADDR    0x6A

// Potentiometer configuration
#define POT_GPIO_PIN       2
#define POT_ADC_CHANNEL    ADC_CHANNEL_2

// CAN/TWAI configuration
#define CAN_TX_PIN         8
#define CAN_RX_PIN         9
#define CAN_BITRATE        TWAI_TIMING_CONFIG_500KBITS()

// CAN message IDs
#define CAN_ID_IMU         0x100
#define CAN_ID_POT         0x101

// Update period
#define SEND_PERIOD_MS     100  // Send data every 100ms (10Hz)

/**
 * @brief Initialize CAN/TWAI interface
 */
static esp_err_t can_init(void)
{
    // Configure TWAI
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX_PIN, CAN_RX_PIN, TWAI_MODE_NORMAL);
    twai_timing_config_t t_config = CAN_BITRATE;
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

    esp_err_t ret = twai_driver_install(&g_config, &t_config, &f_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install TWAI driver: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = twai_start();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start TWAI driver: %s", esp_err_to_name(ret));
        twai_driver_uninstall();
        return ret;
    }

    ESP_LOGI(TAG, "CAN/TWAI initialized successfully");
    return ESP_OK;
}

/**
 * @brief Send IMU data over CAN
 */
static void send_imu_data(const imu_data_t *imu_data)
{
    twai_message_t message;
    message.identifier = CAN_ID_IMU;
    message.flags = TWAI_MSG_FLAG_NONE;
    message.data_length_code = 8; // 8 bytes

    // Pack accelerometer data (3x int16 = 6 bytes)
    message.data[0] = (uint8_t)(imu_data->accel.x & 0xFF);
    message.data[1] = (uint8_t)((imu_data->accel.x >> 8) & 0xFF);
    message.data[2] = (uint8_t)(imu_data->accel.y & 0xFF);
    message.data[3] = (uint8_t)((imu_data->accel.y >> 8) & 0xFF);
    message.data[4] = (uint8_t)(imu_data->accel.z & 0xFF);
    message.data[5] = (uint8_t)((imu_data->accel.z >> 8) & 0xFF);
    
    // Pack gyroscope X (2 bytes) - just send X axis for simplicity
    message.data[6] = (uint8_t)(imu_data->gyro.x & 0xFF);
    message.data[7] = (uint8_t)((imu_data->gyro.x >> 8) & 0xFF);

    esp_err_t ret = twai_transmit(&message, pdMS_TO_TICKS(100));
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Failed to send IMU data: %s", esp_err_to_name(ret));
    }
}

/**
 * @brief Send potentiometer data over CAN
 */
static void send_pot_data(int raw_value)
{
    twai_message_t message;
    message.identifier = CAN_ID_POT;
    message.flags = TWAI_MSG_FLAG_NONE;
    message.data_length_code = 2; // 2 bytes for raw ADC value

    // Pack raw ADC value (12-bit, but send as 16-bit)
    message.data[0] = (uint8_t)(raw_value & 0xFF);
    message.data[1] = (uint8_t)((raw_value >> 8) & 0xFF);

    esp_err_t ret = twai_transmit(&message, pdMS_TO_TICKS(100));
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Failed to send potentiometer data: %s", esp_err_to_name(ret));
    }
}

/**
 * @brief Main sensor reading and CAN transmission task
 */
static void sensor_task(void *pvParameters)
{
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(SEND_PERIOD_MS);

    imu_data_t imu_data;
    int pot_raw = 0;

    ESP_LOGI(TAG, "Sensor task started (period: %d ms)", SEND_PERIOD_MS);

    while (1) {
        // Read IMU data
        esp_err_t ret = imu_read_data(&imu_data);
        if (ret == ESP_OK) {
            send_imu_data(&imu_data);
            ESP_LOGD(TAG, "IMU: accel=(%d, %d, %d), gyro=(%d, %d, %d)",
                     imu_data.accel.x, imu_data.accel.y, imu_data.accel.z,
                     imu_data.gyro.x, imu_data.gyro.y, imu_data.gyro.z);
        } else {
            ESP_LOGW(TAG, "Failed to read IMU: %s", esp_err_to_name(ret));
        }

        // Read potentiometer data
        ret = potentiometer_read_raw(&pot_raw);
        if (ret == ESP_OK) {
            send_pot_data(pot_raw);
            ESP_LOGD(TAG, "Pot: raw=%d", pot_raw);
        } else {
            ESP_LOGW(TAG, "Failed to read potentiometer: %s", esp_err_to_name(ret));
        }

        // Wait for next period
        vTaskDelayUntil(&last_wake_time, period);
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "IMU + Potentiometer + CAN Application");
    ESP_LOGI(TAG, "=====================================");

    // Initialize IMU
    imu_config_t imu_config = IMU_CONFIG_DEFAULT();
    imu_config.i2c_port = IMU_I2C_PORT;
    imu_config.sda_pin = IMU_SDA_PIN;
    imu_config.scl_pin = IMU_SCL_PIN;
    imu_config.i2c_freq_hz = IMU_I2C_FREQ_HZ;
    imu_config.sensor_addr = IMU_SENSOR_ADDR;

    esp_err_t ret = imu_init(&imu_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize IMU: %s", esp_err_to_name(ret));
        return;
    }

    if (!imu_is_present()) {
        ESP_LOGE(TAG, "IMU sensor not detected");
        return;
    }

    ESP_LOGI(TAG, "IMU initialized successfully");

    // Initialize potentiometer
    potentiometer_config_t pot_config = POTENTIOMETER_CONFIG_DEFAULT();
    pot_config.gpio_pin = POT_GPIO_PIN;
    pot_config.adc_channel = POT_ADC_CHANNEL;

    ret = potentiometer_init(&pot_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize potentiometer: %s", esp_err_to_name(ret));
        imu_deinit();
        return;
    }

    ESP_LOGI(TAG, "Potentiometer initialized successfully");

    // Initialize CAN
    ret = can_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize CAN");
        imu_deinit();
        potentiometer_deinit();
        return;
    }

    ESP_LOGI(TAG, "CAN initialized successfully");
    ESP_LOGI(TAG, "CAN IDs: IMU=0x%03X, POT=0x%03X", CAN_ID_IMU, CAN_ID_POT);
    ESP_LOGI(TAG, "CAN TX pin: %d, RX pin: %d", CAN_TX_PIN, CAN_RX_PIN);

    // Create sensor reading task
    xTaskCreate(sensor_task, "sensor_task", 4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "Application started. Sending data every %d ms", SEND_PERIOD_MS);
}

