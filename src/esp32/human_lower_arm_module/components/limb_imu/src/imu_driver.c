/*
 * imu_driver.c
 *
 * Private implementation of the LSM6DSO32 driver.
 * (See imu_driver.h for public interface details).
 * * Contains all the static helper functions and low-level I2C logic
 * required to operate the sensor.
 */
#include "imu_driver.h"    
#include <math.h>           
#include "esp_log.h"
#include "driver/i2c.h"    
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"  

// --- Private Definitions ---
// These are only used inside this .c file

static const char *TAG = "IMU_DRIVER";

// --- I2C Configuration ---
#define I2C_MASTER_SCL_IO       5        // GPIO pin for I2C Clock
#define I2C_MASTER_SDA_IO       4        // GPIO pin for I2C Data
#define I2C_MASTER_NUM          0        // I2C port number (0 or 1)
#define I2C_MASTER_FREQ_HZ      400000   // I2C master clock frequency (400kHz Fast Mode)
#define I2C_MASTER_TIMEOUT_MS   1000     // Default timeout for I2C operations

// --- Sensor Register Definitions ---
#define LSM6DSO32_SENSOR_ADDR       0x6A     // I2C device address //## -- add new sensor address if needed
#define LSM6DSO32_WHO_AM_I_REG    0x0F     // "Who Am I" register
#define LSM6DSO32_CTRL1_XL        0x10     // Accelerometer control register
#define LSM6DSO32_CTRL2_G         0x11     // Gyroscope control register
#define LSM6DSO32_CTRL3_C         0x12     // Control register 3 (settings)
#define LSM6DSO32_OUT_TEMP_L      0x20     // Start address for data registers
#define LSM6DSO32_OUTX_L_G        0x22     // Gyro data start address
#define LSM6DSO32_OUTX_L_A        0x28     // Accel data start address

// --- Sensor-specific magic numbers ---
#define LSM6DSO32_WHO_AM_I_VALUE  0x6C     // Expected value from WHO_AM_I register
#define ACCEL_SENSITIVITY_4G      (4.0f / 32768.0f) // Sensitivity for ±4g range
#define GYRO_SENSITIVITY_250DPS   (250.0f / 32768.0f) // Sensitivity for ±250dps range
#define GRAVITY_MS2               9.81f    // Standard gravity
#define DPS_TO_RADS               (M_PI / 180.0f)   // Degrees-per-second to radians-per-second

// --- Private Function Prototypes ---
// These functions are 'static', meaning they are only visible
// within this file. They help keep the public functions clean.

/**
 * @brief Reads a sequence of bytes from a specific register on the IMU.
 * @param reg_addr The register address to start reading from.
 * @param[out] data Pointer to the buffer to store the read data.
 * @param len Number of bytes to read.
 * @return ESP_OK on success.
 */
static esp_err_t lsm6dso32_register_read(uint8_t reg_addr, uint8_t *data, size_t len);

/**
 * @brief Writes a single byte to a specific register on the IMU.
 * @param reg_addr The register address to write to.
 * @param data The 8-bit data byte to write.
 * @return ESP_OK on success.
 */
static esp_err_t lsm6dso32_register_write_byte(uint8_t reg_addr, uint8_t data);

/**
 * @brief Initializes the ESP32's I2C master peripheral.
 * @return ESP_OK on success.
 */
static esp_err_t i2c_master_init(void);


// --- Public Function Implementations ---
// These are the functions defined in the .h file

/**
 * @brief Initializes the I2C bus and the LSM6DSO32 sensor.
 */
esp_err_t imu_init(void) 
{
    // Step 1: Initialize the I2C bus driver
    if (i2c_master_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2C master init failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "I2C master initialized");

    // Step 2: Check if the sensor is connected by reading WHO_AM_I //## -- validate second sensor if needed
    uint8_t who_am_i;
    if (lsm6dso32_register_read(LSM6DSO32_WHO_AM_I_REG, &who_am_i, 1) != ESP_OK) {
         ESP_LOGE(TAG, "Failed to read WHO_AM_I register");
         return ESP_FAIL;
    }

    if (who_am_i != LSM6DSO32_WHO_AM_I_VALUE) {
        ESP_LOGE(TAG, "LSM6DSO32 not found. WHO_AM_I = 0x%02X (Expected 0x%02X)", who_am_i, LSM6DSO32_WHO_AM_I_VALUE);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "LSM6DSO32 sensor found!");

    // Step 3: Configure the sensor //## -- configure second sensor if needed
    
    // Set Accelerometer: 104Hz ODR (0b0101), ±4g range (0b00)
    lsm6dso32_register_write_byte(LSM6DSO32_CTRL1_XL, 0x50); 
    
    // Set Gyroscope: 104Hz ODR (0b0101), ±250 dps range (0b00)
    lsm6dso32_register_write_byte(LSM6DSO32_CTRL2_G, 0x50);  
    
    // Enable block data update (0b0100)
    lsm6dso32_register_write_byte(LSM6DSO32_CTRL3_C, 0x04); 
    
    // Wait for the sensor to stabilize after configuration
    vTaskDelay(pdMS_TO_TICKS(100)); 
    ESP_LOGI(TAG, "LSM6DSO32 configured and ready");
    
    return ESP_OK;
}

/**
 * @brief Reads a full block of sensor data (Temp, Gyro, Accel)
 * and converts it to standard physical units.
 */
esp_err_t imu_read_data(lsm6dso32_data_t *data) //## -- add param for drive sensor address if needed
{
    // This buffer will hold the raw 16-bit (2-byte) values from the sensor
    uint8_t raw_data[14]; // Temp(2) + Gyro(6) + Accel(6) = 14 bytes
    
    // Read all 14 bytes in a single burst read, starting from the temperature register (0x20)
    // This is efficient and relies on the sensor's register auto-increment.
    if (lsm6dso32_register_read(LSM6DSO32_OUT_TEMP_L, raw_data, 14) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read sensor data block");
        return ESP_FAIL;
    }
    
    // --- Data Conversion ---
    // The sensor provides data as 16-bit signed integers (int16_t)
    // in Little Endian format (LSB first, then MSB).
    
    // Temperature (registers 0x20, 0x21)
    int16_t temp_raw = (int16_t)((raw_data[1] << 8) | raw_data[0]);
    data->temperature = (temp_raw / 256.0f) + 25.0f; // Conversion from datasheet
    
    // Gyroscope (registers 0x22 to 0x27)
    int16_t gyro_x_raw = (int16_t)((raw_data[3] << 8) | raw_data[2]);
    int16_t gyro_y_raw = (int16_t)((raw_data[5] << 8) | raw_data[4]);
    int16_t gyro_z_raw = (int16_t)((raw_data[7] << 8) | raw_data[6]);
    
    // Convert raw gyro data to radians per second (rad/s)
    data->gyro.x = (float)gyro_x_raw * GYRO_SENSITIVITY_250DPS * DPS_TO_RADS;
    data->gyro.y = (float)gyro_y_raw * GYRO_SENSITIVITY_250DPS * DPS_TO_RADS;
    data->gyro.z = (float)gyro_z_raw * GYRO_SENSITIVITY_250DPS * DPS_TO_RADS;
    
    // Accelerometer (registers 0x28 to 0x2D)
    int16_t accel_x_raw = (int16_t)((raw_data[9] << 8) | raw_data[8]);
    int16_t accel_y_raw = (int16_t)((raw_data[11] << 8) | raw_data[10]);
    int16_t accel_z_raw = (int16_t)((raw_data[13] << 8) | raw_data[12]);
    
    // Convert raw accel data to meters per second squared (m/s^2)
    data->accel.x = (float)accel_x_raw * ACCEL_SENSITIVITY_4G * GRAVITY_MS2;
    data->accel.y = (float)accel_y_raw * ACCEL_SENSITIVITY_4G * GRAVITY_MS2;
    data->accel.z = (float)accel_z_raw * ACCEL_SENSITIVITY_4G * GRAVITY_MS2;
    
    return ESP_OK;
}


// --- Private Helper Function Implementations ---

/**
 * @brief Initializes the I2C master peripheral on the ESP32.
 */
static esp_err_t i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE, // Enable internal pull-ups
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    return i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0); // No slave buffers
}

/**
 * @brief Wrapper function to read from the I2C device.
 */
static esp_err_t lsm6dso32_register_read(uint8_t reg_addr, uint8_t *data, size_t len) //## -- add param for drive sensor address if needed
{
    // Standard I2C read operation:
    // 1. Write the register address we want to read from.
    // 2. Read back 'len' bytes of data.
    return i2c_master_write_read_device(I2C_MASTER_NUM, LSM6DSO32_SENSOR_ADDR, &reg_addr, 1, data, len, pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

/**
 * @brief Wrapper function to write a byte to the I2C device.
 */
static esp_err_t lsm6dso32_register_write_byte(uint8_t reg_addr, uint8_t data) //## -- add param for drive sensor address if needed
{
    // Standard I2C write operation:
    // Combine register address and data into a 2-byte buffer.
    uint8_t write_buf[2] = {reg_addr, data};
    // Write the buffer to the device.
    return i2c_master_write_to_device(I2C_MASTER_NUM, LSM6DSO32_SENSOR_ADDR, write_buf, sizeof(write_buf), pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}