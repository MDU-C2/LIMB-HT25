# Potentiometer

ESP32-C3 potentiometer driver using ADC manager for shared ADC access.

## Features

- Reads potentiometer values via ADC
- Supports raw, voltage (mV), and normalized (0-1000) readings
- Uses ADC manager for shared ADC unit access
- Automatic ADC calibration support

## Usage

```c
#include "potentiometer.h"

// Initialize with default config (GPIO0, ADC_CHANNEL_0)
potentiometer_init(NULL);

// Or with custom config
potentiometer_config_t config = {
    .gpio_pin = 0,
    .adc_channel = ADC_CHANNEL_0,
    .adc_atten = ADC_ATTEN_DB_12,
    .adc_bitwidth = ADC_BITWIDTH_DEFAULT,
};
potentiometer_init(&config);

// Read values
int raw;
int voltage_mv;
uint16_t normalized;

potentiometer_read_raw(&raw);
potentiometer_read_voltage(&voltage_mv);
potentiometer_read_normalized(&normalized);

// Cleanup
potentiometer_deinit();
```

## Dependencies

- `adc_manager` - For shared ADC unit management
- `esp_adc` - For ADC calibration

