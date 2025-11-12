#include "imu_driver.h"

#include <math.h>
#include <string.h>

#include "esp_log.h"
#include "esp_check.h"

#include "app_config.h"

static const char *TAG = "imu";

#define LSM6DSO32_REG_FUNC_CFG_ACCESS   0x01 // Function configuration access register
#define LSM6DSO32_REG_PIN_CTRL          0x02 // Pin control register
#define LSM6DSO32_REG_WHO_AM_I          0x0F // Who am I register
#define LSM6DSO32_REG_CTRL1_XL          0x10 // Accelerometer control register 1
#define LSM6DSO32_REG_CTRL2_G           0x11 // Gyroscope control register 2
#define LSM6DSO32_REG_CTRL3_C           0x12 // Control register 3
#define LSM6DSO32_REG_CTRL7_G           0x16 // Gyroscope control register 7
#define LSM6DSO32_REG_CTRL8_XL          0x17 // Accelerometer control register 8
#define LSM6DSO32_REG_CTRL9_XL          0x18 // Accelerometer control register 9
#define LSM6DSO32_REG_STATUS_REG        0x1E // Status register
#define LSM6DSO32_REG_OUT_TEMP_L        0x20 // Temperature output L
#define LSM6DSO32_REG_OUTX_L_G          0x22 // Gyro X output L
#define LSM6DSO32_REG_OUTX_L_A          0x28 // Accel X output L

static imu_driver_config_t s_cfg = {0}; // Configuration for the IMU driver
static bool s_initialized = false; // Flag to check if the IMU is initialized
static imu_orientation_t s_orientation = {0}; // Current orientation of the IMU
static bool s_orientation_initialized = false; // Flag to check if the orientation is initialized

/*
 * @brief Write a value to a register
 * @param reg The register to write to
 * @param value The value to write to the register
 * @return ESP_OK if the write was successful, otherwise an error code
 */
static esp_err_t imu_write_reg(uint8_t reg, uint8_t value)
{
    uint8_t data[2] = {reg, value};
    return i2c_master_write_to_device(s_cfg.i2c_port, s_cfg.i2c_addr, data, sizeof(data), pdMS_TO_TICKS(100));
}

/*
 * @brief Read a value from a register
 * @param reg The register to read from
 * @param value The value read from the register
 * @return ESP_OK if the read was successful, otherwise an error code
 */
static esp_err_t imu_read_reg(uint8_t reg, uint8_t *value)
{
    return i2c_master_write_read_device(s_cfg.i2c_port, s_cfg.i2c_addr, &reg, 1, value, 1, pdMS_TO_TICKS(100));
}

/*
 * @brief Read a sequence of values from a register
 * @param start_reg The register to start reading from
 * @param buffer The buffer to store the read values
 * @param length The number of values to read
 * @return ESP_OK if the read was successful, otherwise an error code
 */
static esp_err_t imu_read_multi(uint8_t start_reg, uint8_t *buffer, size_t length)
{
    return i2c_master_write_read_device(s_cfg.i2c_port, s_cfg.i2c_addr, &start_reg, 1, buffer, length, pdMS_TO_TICKS(100));
}

/*
 * @brief Configure the sensor
 * @return ESP_OK if the configuration was successful, otherwise an error code
 */
static esp_err_t configure_sensor(void)
{
    // Reset device
    ESP_RETURN_ON_ERROR(imu_write_reg(LSM6DSO32_REG_CTRL3_C, 0x01), TAG, "Failed to set SW reset");
    vTaskDelay(pdMS_TO_TICKS(10));

    // Enable block data update and auto register increment
    ESP_RETURN_ON_ERROR(imu_write_reg(LSM6DSO32_REG_CTRL3_C, 0x44), TAG, "Failed to configure CTRL3_C");

    // Enable accelerometer: 104 Hz, 4g, bandwidth ~50 Hz
    ESP_RETURN_ON_ERROR(imu_write_reg(LSM6DSO32_REG_CTRL1_XL, 0x4A), TAG, "Failed to configure CTRL1_XL");

    // Enable gyroscope: 104 Hz, 2000 dps
    ESP_RETURN_ON_ERROR(imu_write_reg(LSM6DSO32_REG_CTRL2_G, 0x4C), TAG, "Failed to configure CTRL2_G");

    // Enable gyroscope high-performance mode
    ESP_RETURN_ON_ERROR(imu_write_reg(LSM6DSO32_REG_CTRL7_G, 0x00), TAG, "Failed to configure CTRL7_G");

    // Accelerometer filter configuration (enable LPF2, cutoff ~45 Hz)
    ESP_RETURN_ON_ERROR(imu_write_reg(LSM6DSO32_REG_CTRL8_XL, 0x09), TAG, "Failed to configure CTRL8_XL");

    // Disable I3C, use single data conversion
    ESP_RETURN_ON_ERROR(imu_write_reg(LSM6DSO32_REG_CTRL9_XL, 0x00), TAG, "Failed to configure CTRL9_XL");

    return ESP_OK;
}

/*
 * @brief Initialize the IMU driver
 * @param config The configuration for the IMU driver
 * @return ESP_OK if the initialization was successful, otherwise an error code
 */
esp_err_t imu_driver_init(const imu_driver_config_t *config)
{
    // Check if the configuration is valid
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }

    // Set the configuration
    s_cfg = *config;

    // If the I2C address is not set, use the default address
    if (s_cfg.i2c_addr == 0) {
        s_cfg.i2c_addr = LSM6DSO32_I2C_ADDR;
    }

    // Configure the I2C interface
    i2c_config_t i2c_cfg = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = s_cfg.sda_gpio,
        .scl_io_num = s_cfg.scl_gpio,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = s_cfg.clk_speed_hz,
        .clk_flags = 0,
    };

    esp_err_t err = i2c_param_config(s_cfg.i2c_port, &i2c_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure I2C: %s", esp_err_to_name(err));
        return err;
    }

    // Install the I2C driver
    err = i2c_driver_install(s_cfg.i2c_port, i2c_cfg.mode, 0, 0, 0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Failed to install I2C driver: %s", esp_err_to_name(err));
        return err;
    }

    // Read the WHO_AM_I register
    uint8_t who_am_i = 0;
    ESP_RETURN_ON_ERROR(imu_read_reg(LSM6DSO32_REG_WHO_AM_I, &who_am_i), TAG, "Failed to read WHO_AM_I");
    if (who_am_i != 0x6C) {
        ESP_LOGW(TAG, "Unexpected WHO_AM_I value: 0x%02X", who_am_i);
    }

    // Configure the sensor
    ESP_RETURN_ON_ERROR(configure_sensor(), TAG, "Failed to configure sensor");

    // Set the initialized flag
    s_initialized = true;

    // Set the orientation initialized flag
    s_orientation_initialized = false;

    // Clear the orientation
    memset(&s_orientation, 0, sizeof(s_orientation));

    ESP_LOGI(TAG, "LSM6DSO32 initialized (addr 0x%02X)", s_cfg.i2c_addr);
    return ESP_OK;
}

/*
 * @brief Read the raw data from the IMU
 * @param sample The sample to store the raw data
 * @return ESP_OK if the read was successful, otherwise an error code
 */
esp_err_t imu_driver_read_raw(imu_raw_sample_t *sample)
{
    // Check if the IMU is initialized and the sample is valid
    if (!s_initialized || !sample) {
        return ESP_ERR_INVALID_STATE;
    }

    // Read the raw data from the IMU
    uint8_t buffer[14] = {0};
    esp_err_t err = imu_read_multi(LSM6DSO32_REG_OUT_TEMP_L, buffer, sizeof(buffer));
    if (err != ESP_OK) {
        return err;
    }

    // Store the raw data in the sample
    sample->temperature_raw = (int16_t)((buffer[1] << 8) | buffer[0]);
    for (int i = 0; i < 3; ++i) {
        sample->gyro_raw[i] = (int16_t)((buffer[3 + i * 2 + 1] << 8) | buffer[3 + i * 2]);
        sample->accel_raw[i] = (int16_t)((buffer[9 + i * 2 + 1] << 8) | buffer[9 + i * 2]);
    }

    return ESP_OK;
}

/*
 * @brief Compute the orientation from the raw data
 * @param sample The sample to compute the orientation from
 * @param dt_seconds The time difference between the samples
 * @param out The orientation to store the computed orientation
 */
static void compute_orientation_from_sample(const imu_raw_sample_t *sample, float dt_seconds, imu_orientation_t *out)
{
    // Constants for the gyro and accelerometer
    const float gyro_sensitivity_dps = 0.070f;        // 70 mdps/LSB at 2000 dps full-scale
    const float accel_sensitivity_g = 0.000122f * 4;  // 4 g full-scale, 0.122 mg/LSB
    const float alpha = 0.98f;                        // complementary filter constant

    // Convert the raw data to degrees per second and gravity
    float gyro_dps[3];
    float accel_g[3];
    for (int i = 0; i < 3; ++i) {
        gyro_dps[i] = sample->gyro_raw[i] * gyro_sensitivity_dps;
        accel_g[i] = sample->accel_raw[i] * accel_sensitivity_g;
    }

    // Compute the pitch and roll from the accelerometer
    float pitch_acc = atan2f(accel_g[0], sqrtf(accel_g[1] * accel_g[1] + accel_g[2] * accel_g[2])) * 57.29578f;
    float roll_acc = atan2f(accel_g[1], sqrtf(accel_g[0] * accel_g[0] + accel_g[2] * accel_g[2])) * 57.29578f;

    // If the orientation is not initialized, set the initial orientation
    if (!s_orientation_initialized) {
        s_orientation.pitch_deg = pitch_acc;
        s_orientation.roll_deg = roll_acc;
        s_orientation.yaw_deg = 0.0f;
        s_orientation_initialized = true;
    }

    // Update the orientation using the gyro and accelerometer
    s_orientation.pitch_deg = alpha * (s_orientation.pitch_deg + gyro_dps[1] * dt_seconds) + (1.0f - alpha) * pitch_acc;
    s_orientation.roll_deg = alpha * (s_orientation.roll_deg + gyro_dps[0] * dt_seconds) + (1.0f - alpha) * roll_acc;
    s_orientation.yaw_deg += gyro_dps[2] * dt_seconds;

    // Wrap the yaw around if it is greater than 180 degrees or less than -180 degrees
    if (s_orientation.yaw_deg > 180.0f) {
        s_orientation.yaw_deg -= 360.0f;
    } else if (s_orientation.yaw_deg < -180.0f) {
        s_orientation.yaw_deg += 360.0f;
    }

    // Store the orientation in the output
    if (out) {
        *out = s_orientation;
    }
}

/*
 * @brief Read the orientation from the IMU
 * @param dt_seconds The time difference between the samples
 * @param orientation_out The orientation to store the read orientation
 * @return ESP_OK if the read was successful, otherwise an error code
 */
esp_err_t imu_driver_read_orientation(float dt_seconds, imu_orientation_t *orientation_out)
{
    // Check if the IMU is initialized and the orientation is valid
    if (!s_initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    // Read the raw data from the IMU
    imu_raw_sample_t sample = {0};
    esp_err_t err = imu_driver_read_raw(&sample);
    if (err != ESP_OK) {
        return err;
    }

    // Compute the orientation from the raw data
    compute_orientation_from_sample(&sample, dt_seconds, orientation_out);
    return ESP_OK;
}

/*
 * @brief Read the temperature from the IMU
 * @param temperature_celsius The temperature to store the read temperature
 * @return ESP_OK if the read was successful, otherwise an error code
 */
esp_err_t imu_driver_read_temperature(float *temperature_celsius)
{
    // Check if the IMU is initialized and the temperature is valid
    if (!s_initialized || !temperature_celsius) {
        return ESP_ERR_INVALID_STATE;
    }

    // Read the raw data from the IMU
    imu_raw_sample_t sample = {0};
    esp_err_t err = imu_driver_read_raw(&sample);
    if (err != ESP_OK) {
        return err;
    }

    // Store the temperature in the output
    *temperature_celsius = 25.0f + ((float)sample.temperature_raw) / 256.0f;
    return ESP_OK;
}

