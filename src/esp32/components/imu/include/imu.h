#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c.h"
#include "esp_err.h"

// Table 44 in the LSM6DSO32 datasheet.
// How often the accelerometer refreshes its data.
typedef enum : uint8_t {
  IMU_ODR_XL_OFF = 0,
  IMU_ODR_XL_12_5_HZ = 1,
  IMU_ODR_XL_26_HZ = 2,
  IMU_ODR_XL_52_HZ = 3,
  IMU_ODR_XL_104_HZ = 4,
  IMU_ODR_XL_208_HZ = 5,
  IMU_ODR_XL_416_HZ = 6,
  IMU_ODR_XL_833_HZ = 7,
  IMU_ODR_XL_1660_HZ = 8,
  IMU_ODR_XL_3330_HZ = 9,
  IMU_ODR_XL_6660_HZ = 10,
} ImuAccelerometerOutputDataRate;

// Table 45 in the LSM6DSO32 datasheet.
// The +- range of the accelerometer in standard gravity (g).
typedef enum : uint8_t {
  IMU_FS_XL_4_G = 0,    // 0b0000
  IMU_FS_XL_8_G = 8,    // 0b1000
  IMU_FS_XL_16_G = 12,  // 0b1100
  IMU_FS_XL_32_G = 4,   // 0b0100
} ImuAccelerometerFullScaleRange;

// Table 48 in the LSM6DSO32 datasheet.
// How often the gyroscope refreshes its data.
typedef enum : uint8_t {
  IMU_ODR_G_OFF = 0,
  IMU_ODR_G_12_5_HZ = 1,
  IMU_ODR_G_26_HZ = 2,
  IMU_ODR_G_52_HZ = 3,
  IMU_ODR_G_104_HZ = 4,
  IMU_ODR_G_208_HZ = 5,
  IMU_ODR_G_416_HZ = 6,
  IMU_ODR_G_833_HZ = 7,
  IMU_ODR_G_1660_HZ = 8,
  IMU_ODR_G_3330_HZ = 9,
  IMU_ODR_G_6660_HZ = 10,
} ImuGyroscopeOutputDataRate;

// Table 47 in the LSM6DSO32 datasheet.
// The +- range of the gyroscope in degrees per second (dps).
typedef enum : uint8_t {
  IMU_FS_G_125_DPS = 2,    // 0b0010
  IMU_FS_G_250_DPS = 0,    // 0b0000
  IMU_FS_G_500_DPS = 4,    // 0b0100
  IMU_FS_G_1000_DPS = 8,   // 0b1000
  IMU_FS_G_2000_DPS = 12,  // 0b1100
} ImuGyroscopeFullScaleRange;

/**
 * @brief IMU raw accelerometer sample mg/LSB.
 */
typedef struct {
  int16_t x;
  int16_t y;
  int16_t z;
} ImuRawAccelVector;

/**
 * @brief IMU raw gyroscope sample in mdps/LSB.
 */
typedef struct {
  int16_t pitch;
  int16_t roll;
  int16_t yaw;
} ImuRawGyroVector;

/**
 * @brief Complete IMU sensor raw data
 */
typedef struct {
  ImuRawAccelVector accel;  // Accelerometer data in mg/LSB
  ImuRawGyroVector gyro;    // Gyroscope data in mdps/LSB
} ImuRawData;

/**
 * @brief IMU configuration structure
 */
typedef struct {
  i2c_port_t i2c_port;   // I2C port number
  uint8_t sda_pin;       // GPIO number for SDA
  uint8_t scl_pin;       // GPIO number for SCL
  uint32_t i2c_freq_hz;  // I2C clock frequency in Hz
  uint8_t sensor_addr;   // I2C address of the sensor (default: 0x6A)
  // Accelerometer measurement range (+-N mg)
  ImuAccelerometerFullScaleRange accel_range;
  // Gyroscope measurement range (+-N mdps)
  ImuGyroscopeFullScaleRange gyro_range;
  ImuAccelerometerOutputDataRate accel_odr;
  ImuGyroscopeOutputDataRate gyro_odr;
} ImuConfig;

/**
 * @brief Default IMU configuration
 */
#define IMU_CONFIG_DEFAULT()                                                  \
  (ImuConfig) {                                                               \
    .i2c_port = I2C_NUM_0, .sda_pin = 4, .scl_pin = 5, .i2c_freq_hz = 400000, \
    .sensor_addr = 0x6A, .accel_range = IMU_FS_XL_4_G,                        \
    .gyro_range = IMU_FS_G_250_DPS, .accel_odr = IMU_ODR_XL_208_HZ,           \
    .gyro_odr = IMU_ODR_G_208_HZ,                                             \
  }

/**
 * @brief Initialize the IMU component
 *
 * @param config Configuration structure for the IMU
 * @return esp_err_t ESP_OK on success
 */
esp_err_t imu_init(const ImuConfig* config);

/**
 * @brief Deinitialize the IMU component
 *
 * @return esp_err_t ESP_OK on success
 */
esp_err_t imu_deinit(void);

/**
 * @brief Read IMU sensor data
 *
 * @param data Pointer to ImuRawData structure to store the data
 * @return esp_err_t ESP_OK on success
 */
esp_err_t imu_read_data(ImuRawData* data);

/**
 * @brief Check if IMU sensor is present and responding
 *
 * @return true if sensor is detected, false otherwise
 */
bool imu_is_present(void);
