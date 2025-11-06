/*
 * imu_driver.c
 *
 * Private implementation of the LSM6DSO32 driver.
 * Contains all the static helper functions and low-level I2C logic
 * required to operate the sensor.
 */
#include "imu_driver.h"     // Include our own header file
#include <math.h>           // Required for M_PI
#include "esp_log.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// --- Private Definitions ---
// These are only used inside this .c file

static const char *TAG = "IMU_DRIVER";

// I2C Configuration
#define I2C_MASTER_SCL_IO           5
#define I2C_MASTER_SDA_IO           4
#define I2C_MASTER_NUM              0
#define I2C_MASTER_FREQ_HZ          400000
#define I2C_MASTER_TIMEOUT_MS       1000

// Sensor Register Definitions
#define LSM6DSO32_SENSOR_ADDR       0x6A
#define LSM6DSO32_WHO_AM_I_REG      0x0F
#define LSM6DSO32_CTRL1_XL          0x10
#define LSM6DSO32_CTRL2_G           0x11
#define LSM6DSO32_CTRL3_C           0x12
#define LSM6DSO32_OUT_TEMP_L        0x20
#define LSM6DSO32_OUTX_L_G          0x22
#define LSM6DSO32_OUTX_L_A          0x28

// --- Private Function Prototypes ---
// These functions are 'static', meaning they are only visible
// within this file. They help keep the public functions clean.
static esp_err_t lsm6dso32_register_read(uint8_t reg_addr, uint8_t *data, size_t len);
static esp_err_t lsm6dso32_register_write_byte(uint8_t reg_addr, uint8_t data);
static esp_err_t i2c_master_init(void);


// --- Public Function Implementations ---
// These are the functions defined in the .h file

esp_err_t imu_init(void)
{
    // Step 1: Initialize the I2C bus driver
    if (i2c_master_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2C master init failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "I2C master initialized");

    // Step 2: Check if the sensor is connected
    uint8_t who_am_i;
    if (lsm6dso32_register_read(LSM6DSO32_WHO_AM_I_REG, &who_am_i, 1) != ESP_OK) {
         ESP_LOGE(TAG, "Failed to read WHO_AM_I register");
         return ESP_FAIL;
    }

    if (who_am_i != 0x6C) {
        ESP_LOGE(TAG, "LSM6DSO32 not found. WHO_AM_I = 0x%02X (Expected 0x6C)", who_am_i);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "LSM6DSO32 sensor found!");

    // Step 3: Configure the sensor
    // (These are the same settings from your friend's code)
    lsm6dso32_register_write_byte(LSM6DSO32_CTRL1_XL, 0x50); // ±4g range, 104 Hz ODR
    lsm6dso32_register_write_byte(LSM6DSO32_CTRL2_G, 0x50);  // ±250 dps range, 104 Hz ODR
    lsm6dso32_register_write_byte(LSM6DSO32_CTRL3_C, 0x04); // Enable address auto-increment
    
    vTaskDelay(pdMS_TO_TICKS(100)); // Wait for sensor to stabilize
    ESP_LOGI(TAG, "LSM6DSO32 configured and ready");
    
    return ESP_OK;
}

esp_err_t imu_read_data(lsm6dso32_data_t *data)
{
    // This buffer will hold the raw 16-bit values from the sensor
    uint8_t raw_data[14]; // Temp(2) + Gyro(6) + Accel(6) = 14 bytes
    
    // Read all 14 bytes at once, starting from the temperature register
    if (lsm6dso32_register_read(LSM6DSO32_OUT_TEMP_L, raw_data, 14) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read sensor data block");
        return ESP_FAIL;
    }
    
    // --- Data Conversion ---
    // Reconstruct the 16-bit signed integers from the two 8-bit bytes (Little Endian)
    
    // Temperature
    int16_t temp = (int16_t)((raw_data[1] << 8) | raw_data[0]);
    data->temperature = (temp / 256.0f) + 25.0f; // Conversion from datasheet
    
    // Gyroscope
    int16_t gyro_x = (int16_t)((raw_data[3] << 8) | raw_data[2]);
    int16_t gyro_y = (int16_t)((raw_data[5] << 8) | raw_data[4]);
    int16_t gyro_z = (int16_t)((raw_data[7] << 8) | raw_data[6]);
    // Convert to rad/s (using M_PI from math.h for better precision)
    data->gyro.x = (float)gyro_x * 250.0f / 32768.0f * (M_PI / 180.0f);
    data->gyro.y = (float)gyro_y * 250.0f / 32768.0f * (M_PI / 180.0f);
    data->gyro.z = (float)gyro_z * 250.0f / 32768.0f * (M_PI / 180.0f);
    
    // Accelerometer
    int16_t accel_x = (int16_t)((raw_data[9] << 8) | raw_data[8]);
    int16_t accel_y = (int16_t)((raw_data[11] << 8) | raw_data[10]);
    int16_t accel_z = (int16_t)((raw_data[13] << 8) | raw_data[12]);
    // Convert to m/s^2
    data->accel.x = (float)accel_x * 4.0f / 32768.0f * 9.81f;
    data->accel.y = (float)accel_y * 4.0f / 32768.0f * 9.81f;
    data->accel.z = (float)accel_z * 4.0f / 32768.0f * 9.81f;
    
    return ESP_OK;
}


// --- Private Helper Function Implementations ---

static esp_err_t i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    return i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0); // No buffers needed
}

static esp_err_t lsm6dso32_register_read(uint8_t reg_addr, uint8_t *data, size_t len)
{
    // I2C read: Write the register address, then read back 'len' bytes
    return i2c_master_write_read_device(I2C_MASTER_NUM, LSM6DSO32_SENSOR_ADDR, &reg_addr, 1, data, len, pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

static esp_err_t lsm6dso32_register_write_byte(uint8_t reg_addr, uint8_t data)
{
    // I2C write: Combine register address and data into a 2-byte buffer
    uint8_t write_buf[2] = {reg_addr, data};
    return i2c_master_write_to_device(I2C_MASTER_NUM, LSM6DSO32_SENSOR_ADDR, write_buf, sizeof(write_buf), pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}