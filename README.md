# LIMB

LIMB is a student project at [Mälardalen University](https://www.mdu.se/) with the aim of creating a bionic arm
to be used in the rehabilitation process of stroke survivors. The arm was
developed in collaboration with [Universidad de Antioquia](http://www.udea.edu.co/),
[Universidad Tecnológica de Panamá](https://utp.ac.pa/), and
[UC Berkeley](https://www.berkeley.edu/).

![A photo of the bionic arm built in the LIMB project](/res/bionic_arm.jpg)

## Platforms used

The microcontrollers used are ESP32-C3-Zeros and the main computer is an [NVIDIA Jetson AGX Orin Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/).

## Controlling the arm

To control the arm first connect to the NVIDIA Jetson AGX Orin. See
the [AGX Orin documentation](/docs/jetson_agx_orin.md) for more details on how to connect to it.
Then you can use CAN messages to control the arm and get information from its sensors.
See the [CAN documentation](/docs/can/can.md) for more details regarding sending and receiving
CAN messages using the AGX Orin. There are various
ready-made python scripts in the [`/src/`](/src/) directory that can be used to control the arm.
See [`/src/README.md`](/src/README.md) for more details on how to run the python scripts.

## Structure of repository

The key parts of the repository are:

- [`scripts/`](./scripts/)
  - Contains a script to set up the virtual CAN interfaces on the Orin.
- [`src/esp32/`](./src/esp32)
  - Contains the programs for the ESP32-C3-Zero modules controlling the motors and reading from the sensors on the robot arm.
- [`src/layers/`](./src/layers)
  - Contains the main program running on the AGX Orin, reading sensor values over the CAN bus from the ESP32s and over Bluetooth Low Energy from the user before figuring out how the robot arm should behave and sending out the new joint angles on the CAN bus to the ESP32s.

