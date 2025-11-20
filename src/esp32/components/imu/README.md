# IMU

LSM6DSO32 IMU sensor driver for ESP32-C3. Provides accelerometer and gyroscope data via I2C.

## Features

- Accelerometer readings (m/s²)
- Gyroscope readings (rad/s)
- Configurable ranges and output data rates
- I2C communication

## Usage

```c
#include "imu.h"

// Initialize with default config
imu_config_t config = IMU_CONFIG_DEFAULT();
imu_init(&config);

// Or customize configuration
imu_config_t custom_config = {
    .i2c_port = I2C_NUM_0,
    .sda_pin = 4,
    .scl_pin = 5,
    .i2c_freq_hz = 400000,
    .sensor_addr = 0x6A,
    .accel_range = 4,      // 4g range
    .gyro_range = 250,     // 250 dps range
    .accel_odr = 0x50,
    .gyro_odr = 0x50,
};
imu_init(&custom_config);

// Check if sensor is present
if (imu_is_present()) {
    // Read sensor data
    imu_data_t data;
    imu_read_data(&data);
    
    // Access accelerometer: data.accel.x, data.accel.y, data.accel.z
    // Access gyroscope: data.gyro.x, data.gyro.y, data.gyro.z
}

// Cleanup
imu_deinit();
```

## Dependencies

- `driver` - For I2C driver support

