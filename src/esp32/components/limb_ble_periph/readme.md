A component for setting up the Bluetooth LE peripheral used in the LIMB project.

> [!IMPORTANT]
> This component requires that the Bluetooth component is enabled using NimBLE as the host.
> Use `idf.py menuconfig` to enable `Component config > Bluetooth > Bluetooth > Host > Nimble - BLE only`.
>
> It is also recommended to increase the tick rate in FreeRTOS to 1 kHz via
> `Component config > FreeRTOS > Kernel > configTICK_RATE_HZ`.

To use, first start a FreeRTOS task using the provided `BleTask` function in `limb_ble_periph.h`.
Then, you can get access to the tx buffer used for the different sensors using the
`get_emg_buf`, `get_imu_buf`, and `get_piezo_buf` functions in `sensors_service.h`.
After writing data to a tx buffer, you can call the `TryNotifyEmgSubscribers`,
`TryNotifyImuSubscribers`, and `TryNotifyPiezoSubscribers` functions to send a
notification of the corresponding sensor buffer to the central if it is subscribed.

In `sensors_service.h`, there are enums for the details regarding the different
sensors; their frequencies, window sizes, window overlaps, and total amount of
bytes per sample for the readings from all the sensors of the same type. These
should be changed to match the values that are actually used for the sensors.

Here is an example program that sends EMG data using a FreeRTOS task:
```c
#include "freertos/FreeRTOS.h"
#include "limb_ble_periph.h"
#include "sensors_service.h"

// Assuming we have some function that can fill a buffer with some number of
// continuous EMG readings.
void FillEmgData(uint8_t* out_buf, uint16_t buf_size);

// Sending IMU or piezo data works in the same way, just 
void SendEmgDataTask([[maybe_unused]] void* arg) {
  CharacteristicBuffer emg_buf = get_emg_buf();

  while (true) {
    FillEmgData(emg_buf.data, emg_buf.size)
    TryNotifyEmgSubscribers();
  }

  vTaskDelete(NULL);
}

void app_main(void) {
  xTaskCreate(BleTask, "BleTask", 4096, NULL, 5, NULL);
  xTaskCreate(SendEmgDataTask, "SendEmgDataTask", 4096, NULL, 5, NULL);
}
```
