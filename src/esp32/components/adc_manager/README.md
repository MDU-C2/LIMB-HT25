# ADC Manager

Shared ADC unit manager for ESP32-C3. Manages ADC_UNIT_1 initialization and channel registration to allow multiple sensors to share the same ADC unit without conflicts.

## Features

- Single ADC unit initialization (ADC_UNIT_1)
- Channel registration with automatic reuse detection
- One sensor per channel constraint
- Simple handle-based API (handle = channel number)

## Usage

```c
#include "adc_manager.h"

// Initialize ADC manager (optional, auto-initializes on first register)
adc_mgr_init();

// Register a channel
adc_oneshot_chan_cfg_t cfg = {
    .bitwidth = ADC_BITWIDTH_DEFAULT,
    .atten = ADC_ATTEN_DB_12,
};
adc_mgr_handle_t handle = adc_mgr_register_channel(ADC_CHANNEL_0, &cfg);

// Read from channel
int raw_value;
adc_mgr_read(handle, &raw_value);

// Cleanup (optional, only if you want to explicitly deinitialize)
adc_mgr_deinit();
```

## Notes

- Handles are channel numbers (e.g., ADC_CHANNEL_0 returns handle 0)
- Only one sensor can be registered per channel
- Returns -1 on error, valid channel number on success

