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
ImuConfig config = IMU_CONFIG_DEFAULT();
imu_init(&config);

// Or customize configuration
ImuConfig custom_config = {
    .i2c_port = I2C_NUM_0,
    .sda_pin = 4,
    .scl_pin = 5,
    .i2c_freq_hz = 400000,
    .sensor_addr = 0x6A,
    .accel_range = IMU_FS_XL_4_G,   // +-4 g range
    .gyro_range = IMU_FS_G_250_DPS, // +-250 dps range
    .accel_odr = IMU_ODR_XL_208_HZ,
    .gyro_odr = IMU_ODR_G_416_HZ,
};
imu_init(&custom_config);

// Check if sensor is present
if (imu_is_present()) {
    // Read sensor data
    ImuRawData data;
    imu_read_data(&data);
    
    // Access accelerometer: data.accel.x, data.accel.y, data.accel.z
    // Access gyroscope: data.gyro.pitch, data.gyro.roll, data.gyro.yaw
}

// Cleanup
imu_deinit();
```

## Dependencies

- `driver` - For I2C driver support

