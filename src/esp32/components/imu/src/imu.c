#include "imu.h"

#include <string.h>

#include "driver/i2c.h"
#include "endian.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "IMU";

// LSM6DSO32 register addresses
#define LSM6DSO32_WHO_AM_I_REG 0x0F
#define LSM6DSO32_CTRL1_XL 0x10
#define LSM6DSO32_CTRL2_G 0x11
#define LSM6DSO32_CTRL3_C 0x12

// Data output registers
#define LSM6DSO32_OUTX_L_G 0x22
#define LSM6DSO32_OUTX_H_G 0x23
#define LSM6DSO32_OUTY_L_G 0x24
#define LSM6DSO32_OUTY_H_G 0x25
#define LSM6DSO32_OUTZ_L_G 0x26
#define LSM6DSO32_OUTZ_H_G 0x27
#define LSM6DSO32_OUTX_L_A 0x28
#define LSM6DSO32_OUTX_H_A 0x29
#define LSM6DSO32_OUTY_L_A 0x2A
#define LSM6DSO32_OUTY_H_A 0x2B
#define LSM6DSO32_OUTZ_L_A 0x2C
#define LSM6DSO32_OUTZ_H_A 0x2D

#define LSM6DSO32_WHO_AM_I_VALUE 0x6C
#define I2C_MASTER_TIMEOUT_MS 1000

// Static configuration storage
static ImuConfig s_imu_config;
static bool s_imu_initialized = false;
static float s_accel_lsb_value = 0.F;
static float s_gyro_lsb_value = 0.F;

/**
 * @brief Read a sequence of bytes from LSM6DSO32 sensor registers
 */
static esp_err_t imu_register_read(uint8_t reg_addr, uint8_t* data,
                                   size_t len) {
  // Before reading any data, we need to tell the IMU from which register to
  // read. We do that by starting the data transfer with writing the
  // register's address, after which any data we read during the same data
  // transfer will be from that register (table 14 in the LSM6DSO32
  // datasheet). These two steps are combined into one with the
  // i2c_master_write_read_device function.
  return i2c_master_write_read_device(
      s_imu_config.i2c_port, s_imu_config.sensor_addr, &reg_addr, 1, data, len,
      pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

/**
 * @brief Write a byte to a LSM6DSO32 sensor register
 */
static esp_err_t imu_register_write_byte(uint8_t reg_addr, uint8_t data) {
  // Before writing any data, we need to tell the IMU which register the data
  // should be written to. We do this by making sure the first byte that is
  // written in the data transfer is the register's address. This results in
  // any subsequent bytes written in the same data transfer to be written to
  // the specified register (table 12 in the LSM6DSO32 datasheet).
  uint8_t write_buf[2] = {reg_addr, data};
  return i2c_master_write_to_device(
      s_imu_config.i2c_port, s_imu_config.sensor_addr, write_buf,
      sizeof(write_buf), pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

/**
 * @brief Initialize I2C master
 */
static esp_err_t i2c_master_init(void) {
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
static esp_err_t lsm6dso32_configure(void) {
  esp_err_t ret;

  // Configure accelerometer (section 9.12 in the LSM6DSO32 datasheet).
  ret = imu_register_write_byte(
      LSM6DSO32_CTRL1_XL,
      (s_imu_config.accel_odr << 4) | s_imu_config.accel_range);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to configure accelerometer");
    return ret;
  }

  // Configure gyroscope (section 9.13 in the LSM6DSO32 datasheet).
  ret =
      imu_register_write_byte(LSM6DSO32_CTRL2_G, (s_imu_config.gyro_odr << 4) |
                                                     s_imu_config.gyro_range);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to configure gyroscope");
    return ret;
  }

  // Configure control register 3
  // (section 9.14 in the LSM6DSO32 datasheet).
  //
  // Enabling IF_INC automatically switches to reading from the register at
  // the next address for every byte read during a multi-byte read (this is
  // enabled by default, but just in case we set it explicitly). This allows
  // us to read all xyz values from both the accelerometer and the gyroscope
  // with a single read call.
  //
  // Enabling BDU makes sure that the most significant and least significant
  // bytes of the output registers read during multi-byte reads actually belong
  // to the same sample by preventing updates to the output register while it
  // is being read.
  enum {
    BDU = 0x40,     // 0b0100'0000.
    IF_INC = 0x04,  // 0b0000'0100
  };
  ret = imu_register_write_byte(LSM6DSO32_CTRL3_C, BDU | IF_INC);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to configure CTRL3_C");
    return ret;
  }

  // Wait for sensor to stabilize
  vTaskDelay(pdMS_TO_TICKS(100));

  return ESP_OK;
}

/**
 * @brief Returns the appropriate LSB value for the provided accelerometer
 * range
 */
static float imu_accel_range_to_lsb(
    ImuAccelerometerFullScaleRange accel_range) {
  // The raw data is expressed in terms of Least Significant Bits (LSBs), which
  // for the accelerometer represents some amount of mg, where g is standard
  // gravity (9.80665 m/s^2). The amount depends on the sensitivity, with the
  // below values representing mg/LSB for the supported sensitivities (Table 3
  // in the LSM6DSO32 datasheet).
  static const float LSB_4_G = 0.122F;
  static const float LSB_8_G = 0.244F;
  static const float LSB_16_G = 0.488F;
  static const float LSB_32_G = 0.976F;

  switch (accel_range) {
    case IMU_FS_XL_4_G: {
      return LSB_4_G;
    }
    case IMU_FS_XL_8_G: {
      return LSB_8_G;
    }
    case IMU_FS_XL_16_G: {
      return LSB_16_G;
    }
    case IMU_FS_XL_32_G: {
      return LSB_32_G;
    }
  }

  ESP_LOGE(TAG,
           "Invalid accel range passed to imu_accel_range_to_lsb: %d, "
           "defaulting to +-4 g",
           accel_range);
  return LSB_4_G;
}

/**
 * @brief Returns the appropriate LSB value for the provided gyroscope
 * range
 */
static float imu_gyro_range_to_lsb(ImuGyroscopeFullScaleRange gyro_range) {
  // The raw data is expressed in terms of Least Significant Bits (LSBs), which
  // for the gyroscope represent some amount of mdps. The amount depends on the
  // sensitivity, with the below values representing mdps/LSB for the supported
  // sensitivities (Table 3 in the LSM6DSO32 datasheet).
  static const float LSB_125_DPS = 4.375F;
  static const float LSB_250_DPS = 8.75F;
  static const float LSB_500_DPS = 17.50F;
  static const float LSB_1000_DPS = 35.F;
  static const float LSB_2000_DPS = 70.F;

  switch (gyro_range) {
    case IMU_FS_G_125_DPS: {
      return LSB_125_DPS;
    }
    case IMU_FS_G_250_DPS: {
      return LSB_250_DPS;
    }
    case IMU_FS_G_500_DPS: {
      return LSB_500_DPS;
    }
    case IMU_FS_G_1000_DPS: {
      return LSB_1000_DPS;
    }
    case IMU_FS_G_2000_DPS: {
      return LSB_2000_DPS;
    }
  }

  ESP_LOGE(TAG,
           "Invalid gyro range passed to imu_gyro_range_to_lsb: %d, "
           "defaulting to 250 dps",
           gyro_range);
  return LSB_250_DPS;
}

esp_err_t imu_init(const ImuConfig* config) {
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
    ESP_LOGE(TAG, "Invalid WHO_AM_I value: 0x%02X (expected: 0x%02X)", who_am_i,
             LSM6DSO32_WHO_AM_I_VALUE);
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

  // Cache the LSB values for the accelerometer and gyroscope.
  s_accel_lsb_value = imu_accel_range_to_lsb(s_imu_config.accel_range);
  s_gyro_lsb_value = imu_gyro_range_to_lsb(s_imu_config.gyro_range);

  s_imu_initialized = true;
  ESP_LOGI(TAG, "IMU initialized successfully");
  return ESP_OK;
}

esp_err_t imu_deinit(void) {
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

esp_err_t imu_read_data(ImuRawData* data) {
  if (data == NULL) {
    ESP_LOGE(TAG, "Data pointer cannot be NULL");
    return ESP_ERR_INVALID_ARG;
  }

  if (!s_imu_initialized) {
    ESP_LOGE(TAG, "IMU not initialized");
    return ESP_ERR_INVALID_STATE;
  }

  // Reading gyro and accelerometer data requires us to read from multiple
  // registers. However, because IF_INC is enabled in the CTRL3_C register,
  // the register address being read from is incremented for every new byte
  // that is read during a multi-byte read (9.14 in the LSM6DSO32 datasheet).
  // Since the relevant data registers span the address range 0x22-0x2D, we
  // can simply read 12 bytes starting from address 0x22 (OUTX_L_G) to get
  // both the gyro and accelerometer data with one read call (9.28-9.33 in
  // the LSM6DSO32 datasheet).

  // To avoid strict aliasing issues when reading the 16-bit values from the
  // buffer, we can let the buffer contain 6 `int16_t`s from the get-go
  // rather than 12 `uint8_t`s.
  int16_t raw_data[6];  // Gyro(6) + Accel(6) = 12 bytes or 6 16-bit ints.

  // Read gyroscope and accelerometer data (12 bytes)
  ESP_RETURN_ON_ERROR(imu_register_read(LSM6DSO32_OUTX_L_G, (uint8_t*)raw_data,
                                        sizeof(raw_data)),
                      TAG, "Failed to read gyro and accelerometer data");

  // The registers are ordered such that they are read as little endian
  // 16-bit integers. As such, we can use le16toh to convert them from
  // little endian to the host endianness.
  data->gyro.pitch = le16toh(raw_data[0]);
  data->gyro.roll = le16toh(raw_data[1]);
  data->gyro.yaw = le16toh(raw_data[2]);
  data->accel.x = le16toh(raw_data[3]);
  data->accel.y = le16toh(raw_data[4]);
  data->accel.z = le16toh(raw_data[5]);

  return ESP_OK;
}

bool imu_is_present(void) {
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

float imu_to_mg(int16_t raw_accel_value) {
  return raw_accel_value * s_accel_lsb_value;
}

float imu_to_mdps(int16_t raw_gyro_value) {
  return raw_gyro_value * s_gyro_lsb_value;
}

ImuAccelVector imu_to_mg_vector(ImuRawAccelVector raw_accel_vector) {
  return (ImuAccelVector){
      .x = imu_to_mg(raw_accel_vector.x),
      .y = imu_to_mg(raw_accel_vector.y),
      .z = imu_to_mg(raw_accel_vector.z),
  };
}

ImuGyroVector imu_to_mdps_vector(ImuRawGyroVector raw_gyro_vector) {
  return (ImuGyroVector){
      .pitch = imu_to_mdps(raw_gyro_vector.pitch),
      .roll = imu_to_mdps(raw_gyro_vector.roll),
      .yaw = imu_to_mdps(raw_gyro_vector.yaw),
  };
}

ImuData imu_to_mg_and_mdps(ImuRawData raw_data) {
  return (ImuData){
      .gyro = imu_to_mdps_vector(raw_data.gyro),
      .accel = imu_to_mg_vector(raw_data.accel),
  };
}
