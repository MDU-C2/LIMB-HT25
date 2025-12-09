

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "esp_log.h"
#include "esp_err.h"

#include "driver/i2c.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "can_driver.h"

static const char *TAG = "can_tx";

// I2C Configuration for IMU
#define I2C_MASTER_NUM        0
#define I2C_MASTER_SDA_IO     7
#define I2C_MASTER_SCL_IO     6
#define I2C_MASTER_FREQ_HZ    400000
#define I2C_MASTER_TIMEOUT_MS 2000  // Increased timeout for reliability

#define LSM6DSO32_ADDR        0x6A
#define LSM6DSO32_WHO_AM_I    0x0F
#define LSM6DSO32_CTRL1_XL    0x10
#define LSM6DSO32_CTRL2_G     0x11
#define LSM6DSO32_CTRL3_C     0x12
#define LSM6DSO32_OUTX_L_A    0x28
#define LSM6DSO32_OUTY_L_A    0x2A
#define LSM6DSO32_OUTX_L_G    0x22
#define LSM6DSO32_OUTY_L_G    0x24

#define CAN_ID_IMU_DATA       0x100
#define IMU_TX_PERIOD_MS      50
#define CAN_TX_PIN           5
#define CAN_RX_PIN           4
#define CAN_BAUDRATE         500000

// IMU Raw data structure
typedef struct {
    int16_t accel_x;
    int16_t accel_y;
    //int16_t accel_z;
    int16_t gyro_x;
    int16_t gyro_y;
    //int16_t gyro_z;
} imu_raw_data_t;

// IMU CAN payload structure
// Note: CAN frames are limited to 8 bytes, so we use int16 values
// Scale factors: accel = ±4g (8192 counts/g), gyro = ±250 dps (16384 counts/dps)
typedef struct {
    int16_t accel_x;  // Raw accel X (will be scaled by receiver)
    int16_t accel_y;  // Raw accel Y
    int16_t gyro_x;   // Raw gyro X
    int16_t gyro_y;   // Raw gyro Y
} imu_can_payload_t;

// I2C and IMU functions (same as before)
static esp_err_t imu_register_read(uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, LSM6DSO32_ADDR, &reg_addr, 1, data, len, I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

static esp_err_t imu_register_write(uint8_t reg_addr, uint8_t data)
{
    uint8_t write_buf[2] = {reg_addr, data};
    return i2c_master_write_to_device(I2C_MASTER_NUM, LSM6DSO32_ADDR, write_buf, sizeof(write_buf), I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

static esp_err_t i2c_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    
    esp_err_t ret = i2c_param_config(I2C_MASTER_NUM, &conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure I2C");
        return ret;
    }

    return i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
}

static esp_err_t imu_init(void)
{
    esp_err_t ret;
    uint8_t who_am_i;

    ret = imu_register_read(LSM6DSO32_WHO_AM_I, &who_am_i, 1);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read WHO_AM_I register");
        return ret;
    }

    if (who_am_i != 0x6C) {
        ESP_LOGE(TAG, "Invalid WHO_AM_I register value: 0x%02X", who_am_i);
        return ESP_ERR_NOT_FOUND;
    }

    ESP_LOGI(TAG, "LSM6DSO32 detected (WHO_AM_I: 0x%02X)", who_am_i);

    ret = imu_register_write(LSM6DSO32_CTRL1_XL, 0x50); // 26 Hz ODR, high-performance mode
    if (ret != ESP_OK) return ret;

    ret = imu_register_write(LSM6DSO32_CTRL2_G, 0x50); // Gyroscope off
    if (ret != ESP_OK) return ret;

    ret = imu_register_write(LSM6DSO32_CTRL3_C, 0x04); // Continuous update mode
    if (ret != ESP_OK) return ret;


    vTaskDelay(pdMS_TO_TICKS(100)); // Wait for the IMU to stabilize
    ESP_LOGI(TAG, "LSM6DSO32 initialized");
    return ESP_OK;
}

static esp_err_t imu_read_data(imu_raw_data_t *data)
{
    uint8_t raw_bytes[8];
    esp_err_t ret;

    // Read accelerometer data (4 bytes: X_L, X_H, Y_L, Y_H)
    ret = imu_register_read(LSM6DSO32_OUTX_L_A, raw_bytes, 4);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read accelerometer: %s", esp_err_to_name(ret));
        return ret;
    }

    // Small delay between I2C reads to avoid bus conflicts
    vTaskDelay(pdMS_TO_TICKS(1));

    // Read gyroscope data (4 bytes: X_L, X_H, Y_L, Y_H)
    ret = imu_register_read(LSM6DSO32_OUTX_L_G, raw_bytes + 4, 4);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read gyroscope: %s", esp_err_to_name(ret));
        return ret;
    }

    // Parse data (little-endian)
    data->accel_x = (int16_t)((raw_bytes[1] << 8) | raw_bytes[0]);
    data->accel_y = (int16_t)((raw_bytes[3] << 8) | raw_bytes[2]);
    data->gyro_x = (int16_t)((raw_bytes[5] << 8) | raw_bytes[4]);
    data->gyro_y = (int16_t)((raw_bytes[7] << 8) | raw_bytes[6]);
    
    return ESP_OK;
}

// Main task using CAN driver
static void imu_tx_task(void *arg)
{
    imu_raw_data_t raw_data;
    imu_can_payload_t can_payload;
    TickType_t last_wake_time = xTaskGetTickCount();

    ESP_LOGI(TAG, "IMU TX task started");

    while (1) {
        esp_err_t ret = imu_read_data(&raw_data);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to read IMU data: %s (0x%x)", esp_err_to_name(ret), ret);
            // Add a small delay before retrying to let I2C bus recover
            vTaskDelay(pdMS_TO_TICKS(10));
            vTaskDelayUntil(&last_wake_time, pdMS_TO_TICKS(IMU_TX_PERIOD_MS));
            continue;
        }

        // Pack raw int16 values directly (fits in 8 bytes)
        can_payload.accel_x = raw_data.accel_x;
        can_payload.accel_y = raw_data.accel_y;
        can_payload.gyro_x = raw_data.gyro_x;
        can_payload.gyro_y = raw_data.gyro_y;

        ret = can_send(CAN_ID_IMU_DATA, (uint8_t *)&can_payload, sizeof(can_payload));
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to send IMU data: %d", esp_err_to_name(ret));
        }

        if (ret == ESP_OK) {
            // Convert to float for logging
            float accel_x_f = (float)can_payload.accel_x / 8192.0f;
            float accel_y_f = (float)can_payload.accel_y / 8192.0f;
            float gyro_x_f = (float)can_payload.gyro_x / 16384.0f;
            float gyro_y_f = (float)can_payload.gyro_y / 16384.0f;
            ESP_LOGI(TAG, "TX: accel_x=%.2f, accel_y=%.2f, gyro_x=%.2f, gyro_y=%.2f", 
                     accel_x_f, accel_y_f, gyro_x_f, gyro_y_f);
        } else {
            ESP_LOGE(TAG, "CAN TX failed: %s", esp_err_to_name(ret));
        }

        vTaskDelayUntil(&last_wake_time, pdMS_TO_TICKS(IMU_TX_PERIOD_MS));
    }
}

void app_main(void) 
{
    ESP_LOGI(TAG, "Starting CAN TX with IMU");

    ESP_ERROR_CHECK(i2c_init());
    // Delay to let I2C bus stabilize
    vTaskDelay(pdMS_TO_TICKS(200));
    
    ESP_ERROR_CHECK(imu_init());
    ESP_ERROR_CHECK(can_init(CAN_TX_PIN, CAN_RX_PIN, CAN_BAUDRATE));

    xTaskCreate(imu_tx_task, "imu_tx_task", 4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "Main task started");
    
}