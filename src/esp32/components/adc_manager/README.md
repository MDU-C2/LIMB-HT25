# ADC Manager

Shared ADC unit manager for ESP32-C3. Manages ADC_UNIT_1 initialization and channel registration to allow multiple sensors to share the same ADC unit without conflicts.

## Features

- Single ADC unit initialization (ADC_UNIT_1)
- Channel registration with separate sample rates.
- One sensor per channel constraint.
- Reading all samples since the last time adc_mgr_read was called.

## Usage

```c
#include "adc_manager.h"

void app_main(void) {
  // 1. First you set up which channels and sample rates you want to use.
  enum {
    kEmgChannel = ADC_CHANNEL_0,
    kPiezoChannel = ADC_CHANNEL_1,
  };

  AdcMgrChannelConfig channel_configs[] = {
      {
          .channel = kEmgChannel,
          .sample_rate = 4000,
      },
      {
          .channel = kPiezoChannel,
          .sample_rate = 100,
      },
  };

  AdcMgrConfig mgr_config = {
      .channel_configs = channel_configs,
      .channel_configs_len = 2,
      .ms_worth_of_buffer_size = 100,
  };

  // 2. Then you initialize the ADC manager.
  adc_mgr_init(mgr_config);

  // 3. Then you provide the buffers that the ADC manager should write its
  // results into.
  enum {
    kEmgBufCapacity = 1024,
    kPiezoBufCapacity = 1024,
  };

  uint16_t emg_underlying_buf[kEmgBufCapacity] = {0};
  uint16_t piezo_underlying_buf[kPiezoBufCapacity] = {0};

  AdcMgrReadResults results = {
      .channel_buffers =
          {
              // 4. The index corresponds to the channel, so make sure the 0th
              // index is given the buffer for ADC channel 0, etc.
              [kEmgChannel] =
                  {
                      .data = emg_underlying_buf,
                      .capacity = kEmgBufCapacity,
                  },
              [kPiezoChannel] =
                  {
                      .data = piezo_underlying_buf,
                      .capacity = kPiezoBufCapacity,
                  },
          },
  };

  // 4. Then you tell the ADC manager to write any read values to the buffers
  // you provided in the AdcMgrReadResults variable.
  adc_mgr_read(&results, 0);

  // 5. Use the values however you want.
  AdcMgrChannelBuffer* emg_channel_buffer =
      &results.channel_buffers[kEmgChannel];
  AdcMgrChannelBuffer* piezo_channel_buffer =
      &results.channel_buffers[kPiezoChannel];

  // Print piezo readings.
  for (int i = 0; i < piezo_channel_buffer->length; ++i) {
    uint16_t value = piezo_channel_buffer->data[i];
    // Do something with the value.
  }

  // Print EMG readings.
  for (int i = 0; i < emg_channel_buffer->length; ++i) {
    uint16_t value = emg_channel_buffer->data[i];
    // Do something with the value.
  }

  // 6. Make sure you update the length of the buffers after you have used the
  // values!
  piezo_channel_buffer->length = 0;
  emg_channel_buffer->length = 0;

  // 7. Deinitialize the ADC manager when you're done with it.
  adc_mgr_deinit();
}
```

A full example program can be found in the `example` directory.

## Notes
You will get any available samples for all channels you registered in `adc_mgr_init`.
Make sure you handle all the samples before your next call to `adc_mgr_read`.

