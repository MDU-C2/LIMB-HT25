#include "imu.h"

#include <string.h>
#include "esp_log.h"
#include "driver/i2c.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "IMU";

// LSM6DSO32 register addresses
#define LSM6DSO32_WHO_AM_I_REG      0x0F
#define LSM6DSO32_CTRL1_XL          0x10
#define LSM6DSO32_CTRL2_G           0x11
#define LSM6DSO32_CTRL3_C           0x12

// Data output registers
#define LSM6DSO32_OUTX_L_G          0x22
#define LSM6DSO32_OUTX_H_G          0x23
#define LSM6DSO32_OUTY_L_G          0x24
#define LSM6DSO32_OUTY_H_G          0x25
#define LSM6DSO32_OUTZ_L_G          0x26
#define LSM6DSO32_OUTZ_H_G          0x27
#define LSM6DSO32_OUTX_L_A          0x28
#define LSM6DSO32_OUTX_H_A          0x29
#define LSM6DSO32_OUTY_L_A          0x2A
#define LSM6DSO32_OUTY_H_A          0x2B
#define LSM6DSO32_OUTZ_L_A          0x2C
#define LSM6DSO32_OUTZ_H_A          0x2D

#define LSM6DSO32_WHO_AM_I_VALUE    0x6C
#define I2C_MASTER_TIMEOUT_MS        1000

// Static configuration storage
static imu_config_t s_imu_config;
static bool s_imu_initialized = false;

/**
 * @brief Read a sequence of bytes from LSM6DSO32 sensor registers
 */
static esp_err_t imu_register_read(uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(s_imu_config.i2c_port,
                                        s_imu_config.sensor_addr,
                                        &reg_addr, 1,
                                        data, len,
                                        I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

/**
 * @brief Write a byte to a LSM6DSO32 sensor register
 */
static esp_err_t imu_register_write_byte(uint8_t reg_addr, uint8_t data)
{
    uint8_t write_buf[2] = {reg_addr, data};
    return i2c_master_write_to_device(s_imu_config.i2c_port,
                                      s_imu_config.sensor_addr,
                                      write_buf, sizeof(write_buf),
                                      I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

/**
 * @brief Initialize I2C master
 */
static esp_err_t i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = s_imu_config.sda_pin,
        .scl_io_num = s_imu_config.scl_pin,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = s_imu_config.i2c_freq_hz,
    };

    esp_err_t ret = i2c_param_config(s_imu_config.i2c_port, &conf);
    if (ret != ESP_OK) {
        return ret;
    }

    return i2c_driver_install(s_imu_config.i2c_port, conf.mode, 0, 0, 0);
}

/**
 * @brief Configure LSM6DSO32 sensor registers
 */
static esp_err_t lsm6dso32_configure(void)
{
    esp_err_t ret;

    // Configure accelerometer
    ret = imu_register_write_byte(LSM6DSO32_CTRL1_XL, s_imu_config.accel_odr);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure accelerometer");
        return ret;
    }

    // Configure gyroscope
    ret = imu_register_write_byte(LSM6DSO32_CTRL2_G, s_imu_config.gyro_odr);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure gyroscope");
        return ret;
    }

    // Configure control register 3 (enable IF_INC for auto-increment)
    ret = imu_register_write_byte(LSM6DSO32_CTRL3_C, 0x04);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure CTRL3_C");
        return ret;
    }

    // Wait for sensor to stabilize
    vTaskDelay(pdMS_TO_TICKS(100));

    return ESP_OK;
}

esp_err_t imu_init(const imu_config_t *config)
{
    if (config == NULL) {
        ESP_LOGE(TAG, "Configuration cannot be NULL");
        return ESP_ERR_INVALID_ARG;
    }

    if (s_imu_initialized) {
        ESP_LOGW(TAG, "IMU already initialized");
        return ESP_OK;
    }

    // Copy configuration
    s_imu_config = *config;

    // Initialize I2C
    esp_err_t ret = i2c_master_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize I2C: %s", esp_err_to_name(ret));
        return ret;
    }

    // Verify sensor presence by reading WHO_AM_I register
    uint8_t who_am_i;
    ret = imu_register_read(LSM6DSO32_WHO_AM_I_REG, &who_am_i, 1);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read WHO_AM_I register: %s", esp_err_to_name(ret));
        i2c_driver_delete(s_imu_config.i2c_port);
        return ret;
    }

    if (who_am_i != LSM6DSO32_WHO_AM_I_VALUE) {
        ESP_LOGE(TAG, "Invalid WHO_AM_I value: 0x%02X (expected: 0x%02X)", 
                 who_am_i, LSM6DSO32_WHO_AM_I_VALUE);
        i2c_driver_delete(s_imu_config.i2c_port);
        return ESP_ERR_NOT_FOUND;
    }

    // Configure sensor
    ret = lsm6dso32_configure();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure sensor");
        i2c_driver_delete(s_imu_config.i2c_port);
        return ret;
    }

    s_imu_initialized = true;
    ESP_LOGI(TAG, "IMU initialized successfully");
    return ESP_OK;
}

esp_err_t imu_deinit(void)
{
    if (!s_imu_initialized) {
        return ESP_OK;
    }

    esp_err_t ret = i2c_driver_delete(s_imu_config.i2c_port);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to delete I2C driver: %s", esp_err_to_name(ret));
        return ret;
    }

    s_imu_initialized = false;
    ESP_LOGI(TAG, "IMU deinitialized");
    return ESP_OK;
}

esp_err_t imu_read_data(imu_data_t *data)
{
    if (data == NULL) {
        ESP_LOGE(TAG, "Data pointer cannot be NULL");
        return ESP_ERR_INVALID_ARG;
    }

    if (!s_imu_initialized) {
        ESP_LOGE(TAG, "IMU not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t raw_data[12]; // Gyro(6) + Accel(6) = 12 bytes
    esp_err_t ret;

    // Read gyroscope data (6 bytes)
    ret = imu_register_read(LSM6DSO32_OUTX_L_G, raw_data, 6);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read gyroscope data");
        return ret;
    }

    // Read accelerometer data (6 bytes)
    ret = imu_register_read(LSM6DSO32_OUTX_L_A, raw_data + 6, 6);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read accelerometer data");
        return ret;
    }

    // Convert gyroscope data (16-bit, ±250 dps range by default)
    data->gyro.x = (int16_t)((raw_data[1] << 8) | raw_data[0]);
    data->gyro.y = (int16_t)((raw_data[3] << 8) | raw_data[2]);
    data->gyro.z = (int16_t)((raw_data[5] << 8) | raw_data[4]);

    data->accel.x = (int16_t)((raw_data[7] << 8) | raw_data[6]);
    data->accel.y = (int16_t)((raw_data[9] << 8) | raw_data[8]);
    data->accel.z = (int16_t)((raw_data[11] << 8) | raw_data[10]);

    return ESP_OK;
}

bool imu_is_present(void)
{
    if (!s_imu_initialized) {
        return false;
    }

    uint8_t who_am_i;
    esp_err_t ret = imu_register_read(LSM6DSO32_WHO_AM_I_REG, &who_am_i, 1);
    if (ret != ESP_OK) {
        return false;
    }

    return (who_am_i == LSM6DSO32_WHO_AM_I_VALUE);
}

