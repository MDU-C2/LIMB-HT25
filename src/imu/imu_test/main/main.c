/* LSM6DSO32 I2C Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include "esp_log.h"
#include "driver/i2c.h"
#include "sdkconfig.h"
// #include "esp_driver_i2c.h"  // Not needed for legacy I2C driver
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"

static const char *TAG = "LSM6DSO32";

#define I2C_MASTER_SCL_IO           5                        /*!< GPIO number used for I2C master clock */
#define I2C_MASTER_SDA_IO           4                        /*!< GPIO number used for I2C master data  */
#define I2C_MASTER_NUM              0                         /*!< I2C master i2c port number, the number of i2c peripheral interfaces available will depend on the chip */
#define I2C_MASTER_FREQ_HZ          400000                    /*!< I2C master clock frequency */
#define I2C_MASTER_TX_BUF_DISABLE   0                         /*!< I2C master doesn't need buffer */
#define I2C_MASTER_RX_BUF_DISABLE   0                         /*!< I2C master doesn't need buffer */
#define I2C_MASTER_TIMEOUT_MS       1000

#define LSM6DSO32_SENSOR_ADDR       0x6A        /*!< Address of the LSM6DSO32 sensor (SDO=0) */
#define LSM6DSO32_WHO_AM_I_REG      0x0F        /*!< WHO_AM_I register address */
#define LSM6DSO32_CTRL1_XL          0x10        /*!< Accelerometer control register */
#define LSM6DSO32_CTRL2_G           0x11        /*!< Gyroscope control register */
#define LSM6DSO32_CTRL3_C           0x12        /*!< Control register 3 */

// Data output registers
#define LSM6DSO32_OUT_TEMP_L        0x20        /*!< Temperature output L */
#define LSM6DSO32_OUT_TEMP_H        0x21        /*!< Temperature output H */
#define LSM6DSO32_OUTX_L_G          0x22        /*!< Gyro X output L */
#define LSM6DSO32_OUTX_H_G          0x23        /*!< Gyro X output H */
#define LSM6DSO32_OUTY_L_G          0x24        /*!< Gyro Y output L */
#define LSM6DSO32_OUTY_H_G          0x25        /*!< Gyro Y output H */
#define LSM6DSO32_OUTZ_L_G          0x26        /*!< Gyro Z output L */
#define LSM6DSO32_OUTZ_H_G          0x27        /*!< Gyro Z output H */
#define LSM6DSO32_OUTX_L_A          0x28        /*!< Accel X output L */
#define LSM6DSO32_OUTX_H_A          0x29        /*!< Accel X output H */
#define LSM6DSO32_OUTY_L_A          0x2A        /*!< Accel Y output L */
#define LSM6DSO32_OUTY_H_A          0x2B        /*!< Accel Y output H */
#define LSM6DSO32_OUTZ_L_A          0x2C        /*!< Accel Z output L */
#define LSM6DSO32_OUTZ_H_A          0x2D        /*!< Accel Z output H */

// Add UART configuration
#define UART_NUM UART_NUM_0
#define UART_TX_PIN 1
#define UART_RX_PIN 3
#define UART_BAUD_RATE 115200
#define UART_BUF_SIZE 256

// Data structures
typedef struct {
    float x;
    float y;
    float z;
} imu_data_t;

typedef struct {
    imu_data_t accel;    // in m/s²
    imu_data_t gyro;     // in rad/s
    float temperature;   // in °C
} lsm6dso32_data_t;

/**
 * @brief Read a sequence of bytes from a LSM6DSO32 sensor registers
 */
static esp_err_t lsm6dso32_register_read(uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, LSM6DSO32_SENSOR_ADDR, &reg_addr, 1, data, len, I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
}

/**
 * @brief Write a byte to a LSM6DSO32 sensor register
 */
static esp_err_t lsm6dso32_register_write_byte(uint8_t reg_addr, uint8_t data)
{
    int ret;
    uint8_t write_buf[2] = {reg_addr, data};

    ret = i2c_master_write_to_device(I2C_MASTER_NUM, LSM6DSO32_SENSOR_ADDR, write_buf, sizeof(write_buf), I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);

    return ret;
}

/**
 * @brief i2c master initialization
 */
static esp_err_t i2c_master_init(void)
{
    int i2c_master_port = I2C_MASTER_NUM;

    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };

    i2c_param_config(i2c_master_port, &conf);

    return i2c_driver_install(i2c_master_port, conf.mode, I2C_MASTER_RX_BUF_DISABLE, I2C_MASTER_TX_BUF_DISABLE, 0);
}

/**
 * @brief Initialize LSM6DSO32 sensor
 */
static esp_err_t lsm6dso32_init(void)
{
    esp_err_t ret;
    
    // Configure accelerometer: ±4g range, 104 Hz ODR
    ret = lsm6dso32_register_write_byte(LSM6DSO32_CTRL1_XL, 0x50);
    if (ret != ESP_OK) return ret;
    
    // Configure gyroscope: ±250 dps range, 104 Hz ODR  
    ret = lsm6dso32_register_write_byte(LSM6DSO32_CTRL2_G, 0x50);
    if (ret != ESP_OK) return ret;
    
    // Configure control register 3 (enable IF_INC for auto-increment)
    ret = lsm6dso32_register_write_byte(LSM6DSO32_CTRL3_C, 0x04);
    if (ret != ESP_OK) return ret;
    
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_LOGI(TAG, "LSM6DSO32 initialized successfully");
    return ESP_OK;
}

/**
 * @brief Read LSM6DSO32 sensor data
 */
static esp_err_t lsm6dso32_read_data(lsm6dso32_data_t *data)
{
    uint8_t raw_data[14]; // Temperature(2) + Gyro(6) + Accel(6) = 14 bytes
    esp_err_t ret;
    
    // Read temperature data (2 bytes)
    ret = lsm6dso32_register_read(LSM6DSO32_OUT_TEMP_L, raw_data, 2);
    if (ret != ESP_OK) return ret;
    
    // Read gyroscope data (6 bytes)
    ret = lsm6dso32_register_read(LSM6DSO32_OUTX_L_G, raw_data + 2, 6);
    if (ret != ESP_OK) return ret;
    
    // Read accelerometer data (6 bytes)
    ret = lsm6dso32_register_read(LSM6DSO32_OUTX_L_A, raw_data + 8, 6);
    if (ret != ESP_OK) return ret;
    
    // Convert temperature data
    int16_t temp = (int16_t)((raw_data[1] << 8) | raw_data[0]);
    data->temperature = (temp / 256.0f) + 25.0f; // LSM6DSO32 temperature conversion
    
    // Convert gyroscope data (16-bit, ±250 dps range)
    int16_t gyro_x = (int16_t)((raw_data[3] << 8) | raw_data[2]);
    int16_t gyro_y = (int16_t)((raw_data[5] << 8) | raw_data[4]);
    int16_t gyro_z = (int16_t)((raw_data[7] << 8) | raw_data[6]);
    
    // Convert to rad/s (250 dps = 250 * π/180 rad/s)
    data->gyro.x = (float)gyro_x * 250.0f / 32768.0f * 3.14159f / 180.0f;
    data->gyro.y = (float)gyro_y * 250.0f / 32768.0f * 3.14159f / 180.0f;
    data->gyro.z = (float)gyro_z * 250.0f / 32768.0f * 3.14159f / 180.0f;
    
    // Convert accelerometer data (16-bit, ±4g range)
    int16_t accel_x = (int16_t)((raw_data[9] << 8) | raw_data[8]);
    int16_t accel_y = (int16_t)((raw_data[11] << 8) | raw_data[10]);
    int16_t accel_z = (int16_t)((raw_data[13] << 8) | raw_data[12]);
    
    // Convert to m/s² (4g = 4 * 9.81 m/s²)
    data->accel.x = (float)accel_x * 4.0f / 32768.0f * 9.81f;
    data->accel.y = (float)accel_y * 4.0f / 32768.0f * 9.81f;
    data->accel.z = (float)accel_z * 4.0f / 32768.0f * 9.81f;
    
    return ESP_OK;
}

/**
 * @brief Initialize UART
 */
 static esp_err_t uart_init(void) {
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    
    uart_driver_install(UART_NUM, UART_BUF_SIZE * 2, 0, 0, NULL, 0);
    uart_param_config(UART_NUM, &uart_config);
    uart_set_pin(UART_NUM, UART_TX_PIN, UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    return ESP_OK;
}

/**
 * @brief Send IMU data to UART
 */
static void send_imu_data_json(const lsm6dso32_data_t *data) {
    printf("{\"accel\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f},\"gyro\":{\"x\":%.3f,\"y\":%.3f,\"z\":%.3f},\"temp\":%.1f}\n",
        data->accel.x, data->accel.y, data->accel.z,
        data->gyro.x, data->gyro.y, data->gyro.z,
        data->temperature);
}


void app_main(void)
{
    uint8_t data;
    esp_err_t ret;

    // Disable logging on UART0 to keep JSON output clean
    esp_log_level_set("*", ESP_LOG_NONE);
    
    ESP_ERROR_CHECK(i2c_master_init());
    //ESP_LOGI(TAG, "I2C initialized successfully");

    //ESP_ERROR_CHECK(uart_init());
    //ESP_LOGI(TAG, "UART initialized successfully");

    /* Read the LSM6DSO32 WHO_AM_I register, on power up the register should have the value 0x6C */
    ESP_ERROR_CHECK(lsm6dso32_register_read(LSM6DSO32_WHO_AM_I_REG, &data, 1));
    //ESP_LOGI(TAG, "WHO_AM_I = 0x%02X (expected: 0x6C)", data);

    if (data != 0x6C) {
        //ESP_LOGE(TAG, "WHO_AM_I register value incorrect. Expected 0x6C, got 0x%02X", data);
        return;
    }

    ESP_ERROR_CHECK(lsm6dso32_init());

    /* Read LSM6DSO32 sensor data in a loop */
    lsm6dso32_data_t imu_data;
    while (1) {
        ret = lsm6dso32_read_data(&imu_data);
        if (ret == ESP_OK) {

            // Send data via UART
            send_imu_data_json(&imu_data);

        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
