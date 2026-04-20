# Microcontrollers used in the LIMB project

The microcontrollers used in this project are [ESP32-C3-Zeros](https://www.waveshare.com/esp32-c3-zero.htm).
There are 5 of them used in total, 4 for the robot arm and 1 for the cuff on the human arm.

The ones on the robot arm communicate with the NVIDIA Jetson AGX Orin using a CAN bus. They
send potentiometer, pressure, and IMU data to the Orin and receive motor actuation commands from it.
For more information regarding the CAN bus, look in [docs/can/can.md](./can.md).

The one on the human arm communicates with the Orin using Bluetooth Low Energy. It collects
data from the EMG, IMU, and Piezo sensors and sends the data as notifications that the Orin
subscribes to. For more inforamation regarding the use of BLE, look in [docs/bluetooth_low_energy.md](./ble.md).

The programs for the microcontrollers can be found in the [src/esp32/](../src/esp32/) directory.

## [ESP32-C3-Zero](https://www.waveshare.com/esp32-c3-zero.htm)

The pinout for the ESP32-C3-Zero is shown in the following image:

![ESP32-C3-Zero pinout image](https://www.waveshare.com/img/devkit/ESP32-C3-Zero/ESP32-C3-Zero-details-inter.jpg)

More information can be found in its [datasheet](https://documentation.espressif.com/esp32-c3_datasheet_en.pdf)
and [technical reference manual](https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf).

> [!warning]
> The ESP32-C3-Zero have some pins that can be used to affect the boot procedure called
> [strapping pins](https://documentation.espressif.com/esp32-c3_datasheet_en.pdf#section.3).
> The way this is achieved is by changing the voltage level provided to the pins during the
> boot sequence. It can also be the case that they output some signal on the pin during
> the flashing process. This output signal could affect the thing connected to the pin, e.g.
> a servo that interprets the signal as a pulse width and starts moving (this has happened
> to us and is an issue since we've
> [modded the servos to be continuous](./servo_modification.md)).
> As such, it is best to avoid using them if possible, or otherwise make sure that
> they are used in a safe way.
>
> Other pins to be wary of are
> [the GPIO18 and GPIO19 pins which are used for USB serial communication](https://documentation.espressif.com/esp32-c3_datasheet_en.pdf#subsubsection.2.3.4)
> and
> [the GPIO20 and GPIO 21 pins which are used for UART communication](https://documentation.espressif.com/esp32-c3_datasheet_en.pdf#subsubsection.2.3.4).
> These allow you to interface with the ESP32-C3-Zero via a USB cable (e.g. flashing or monitoring
> output), which means they should also ideally be avoided.

## [ESP-IDF](https://github.com/espressif/esp-idf)
ESP-IDF is the official development framework used for ESP32. This project has been
developed using version 5.5.1, for which the docs can be found
[here](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32c3/index.html).

