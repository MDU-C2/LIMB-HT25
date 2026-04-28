# IMU

LSM6DSO32 IMU sensor driver for ESP32-C3. Provides accelerometer and gyroscope data via I²C.

> [!NOTE]
> This component only supports a single IMU per device. The ESP32-C3 only has a single I²C
> controller, but it is technically possible to connect two IMUs to the same I²C bus by
> connecting the SDO pin to the supply voltage
> ([section 5.1-5.1.1 in the datasheet](https://www.st.com/resource/en/datasheet/lsm6dso32.pdf#page=34)).
> However, using two IMU's on the same I²C bus would require modifying the component to support
> that functionality. See commits [830ce5a](https://github.com/MDU-C2/LIMB-HT25/commit/830ce5afcdf231bf2c67dd96277c1b9fc28e1762) and
> [e19678f](https://github.com/MDU-C2/LIMB-HT25/commit/e19678fbbb36785de4c6ecd20c485283c2859f15) for a previous implementation supporting using
> two IMUs on the same bus.

> [!WARNING]
> This component currently requires exclusive access to the ESP32-C3's I²C controller.

## Features

- Accelerometer readings ($\text{m}g$ &mdash; milli-standard gravity)
- Gyroscope readings (mdps &mdash; millidegrees per second)
- Configurable measurement ranges and output data rates
- Uses I²C communication

## Usage

```c
#include "imu.h"

// Initialize with default config
ImuConfig config = IMU_CONFIG_DEFAULT();
imu_init(&config);

// Or customize configuration
ImuConfig custom_config = {
    .i2c_port = I2C_NUM_0,
    .sda_pin = GPIO_NUM_4,
    .scl_pin = GPIO_NUM_5,
    .i2c_freq_hz = 400000,
     // Use the secondary address only if the SDO/SA0 pin is connected to the supply voltage.
    .sensor_addr = IMU_ADDRESS_SECONDARY,
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

    // Access raw accelerometer values: data.accel.x, data.accel.y, data.accel.z
    // Access raw gyroscope values: data.gyro.pitch, data.gyro.roll, data.gyro.yaw

    // Convert raw values to mg and mdps individually...
    float accel_x = imu_to_mg(data.accel.x);
    float gyro_yaw = imu_to_mdps(data.gyro.yaw);

    // Or convert the entire data struct in one go.
    ImuData converted_data = imu_to_mg_and_mdps(data);
    accel_x = converted_data.accel.x;
    gyro_yaw = converted_data.gyro.yaw;
}

// Cleanup
imu_deinit();
```

## Dependencies

- `driver` - For I2C driver support

## Hardware info
More info about the LSM6DSO32 IMU sensor can be found in
[its datasheet](https://www.st.com/resource/en/datasheet/lsm6dso32.pdf).

## Conversions from raw values to $\text{m}g$ and mdps

The raw values read from `imu_read_data` shouldn't be used directly
since they are expressed in Least Significant Bits (LSBs), which is the
smallest unit of measurement that the IMU outputs. For acceleration,
the LSB represents some amount of $\text{m}g$ ($g$ representing
[standard gravity](https://en.wikipedia.org/wiki/Standard_gravity)).
For the gyroscope, the LSB instead represents mdps (dps being degrees
per second). The exact value of the LSB depends on the measurement range
(also referred to simply as Full Scale (FS) in the data sheet) used for
the acceleration or the gyroscope. The LSB values for each FS are shown in
[table 3](https://www.st.com/resource/en/datasheet/lsm6dso32.pdf#page=25)
of the datasheet. Converting the raw values to $\text{m}g$ or mdps is
then just a matter of multiplying them by the correct LSB value. This
functionality is provided by the `imu_to_mg` and `imu_to_mdps` family
of functions.

