# LIMB

LIMB is a project at Mälardalen University with the aim of creating a bionic arm
to be used in the rehabilitation process of stroke survivors.

## Platforms used

The microcontrollers used are ESP32-C3-Zeros and the main computer is an [NVIDIA Jetson AGX Orin Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/).

## Structure of repository

The key parts of the repository are:

- [`scripts/`](./scripts/)
  - Contains a script to set up the virtual CAN interfaces on the Orin.
- [`src/esp32/`](./src/esp32)
  - Contains the programs for the ESP32-C3-Zero modules controlling the motors and reading from the sensors on the robot arm.
- [`src/layers/`](./src/layers)
  - Contains the main program running on the AGX Orin, reading sensor values over the CAN bus from the ESP32s and over Bluetooth Low Energy from the user before figuring out how the robot arm should behave and sending out the new joint angles on the CAN bus to the ESP32s.

