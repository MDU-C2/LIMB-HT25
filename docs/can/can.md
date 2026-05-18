# CAN

A CAN bus is used to allow the NVIDIA Jetson AGX Orin to communicate with the ESP32-C3-Zeros
distributed on the robot arm.

## CAN bus connections
The CAN bus is composed of a twisted pair running along the arm.
The Orin and all ESP32-C3-Zeros are connected to the CAN bus using
SN65HVD232D CAN transceivers ([datasheet](https://www.ti.com/lit/gpn/sn65hvd232)),
with the transceivers at both ends of the bus containing 120&Omega; terminating resistors.

## Using CAN with the NVIDIA Jetson AGX Orin Developer Kit

See [docs/jetson_agx_orin.md](../jetson_agx_orin.md) for information
regarding connecting the Orin to a CAN bus.

## Using CAN with the ESP32-C3-Zero

The ESP32-C3-Zeros support something called Two-Wire Automotive Interface (TWAI),
which is compatible with CAN frames, so it's basically CAN by another name. The
[TWAI driver](https://docs.espressif.com/projects/esp-idf/en/v5.4.4/esp32/api-reference/peripherals/twai.html)
in ESP-IDF is used to create the component in
[src/esp32/components/can_driver/](../../src/esp32/components/can_driver/) which
we use to send and receive messages over the CAN bus.

> [!note]
> The ESP-IDF TWAI driver used is from ESP-IDF v5.4.4, not the updated one from v5.5.1.

## More information
More information regarding the layout of the CAN messages can be found in
[docs/can/message_data_layout.md](./message_data_layout.md).

