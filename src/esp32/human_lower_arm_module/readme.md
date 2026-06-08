# Human lower arm module

This is the program for the human lower arm module located on the cuff
with EMG, IMU, and Piezo sensors. Its purpose is to provide subscribable
Bluetooth Low Energy notifications for the data from the EMG, IMU,
and Piezo sensors using the [limb_ble_periph component](../components/limb_ble_periph/).

## Steps for building and flashing the program

The steps for building and flashing the ESP32-C3-Zero with the program are as follows:
```sh
idf.py set-target esp32c3
idf.py menuconfig # In case you need to configure some settings.
idf.py build flash # Build and then flash the ESP32-C3-Zero connected to the computer.
```

Once you've set the target and set your configuration in menuconfig, you can just run the last command
from then on.

## Configuration

This program requires some configuration using `idf.py menuconfig`. There is also some
LIMB specific configuration available under the `LIMB config` submenu.

### Enabling Bluetooth Low Energy via NimBLE

The program uses the [limb_ble_periph component](../components/limb_ble_periph/) which depends on
the NimBLE component. However, it's not possible for the limb_ble_periph component to automatically
enable the NimBLE component due to Kconfig shenanigans, so it has to be enabled by this module instead.
It can be enabled using `idf.py menuconfig` in
`Component config -> Bluetooth -> Bluetooth -> Host -> NimBLE - BLE only`.

### Disabling the IMU
The program can also be configured to not use the IMU by running `idf.py menuconfig` and making sure
`LIMB config -> Enable IMU` is unchecked.

> [!note]
> Disabling the IMU is useful in case it's disconnected from the ESP-C3-Zero for whatever reason,
> since the program normally won't start if it can't connect to the IMU.

### Reduce logging
The [limb_ble_periph component](../components/limb_ble_periph/) might end up logging excessively in some scenarios,
which could end up triggering the task watchdog causing the program to stop working. To avoid this,
the logging level can be lowered by running `idf.py menuconfig` and setting
`Component config -> Log -> Log Level -> Default log verbosity`.

## What the module looks like

The module is located on the cuff, as shown in the image:

![Image of the module on the cuff for the human arm](/res/human_lower_arm_module.jpg)
