# Hand pressure module

This program reads data from the pressure sensors on the fingers and from the IMU on the hand.

## Steps for building and flashing the program

The steps for building and flashing the ESP32-C3-Zero with the program are as follows:
```sh
idf.py set-target esp32c3
idf.py menuconfig # In case you need to configure some settings.
idf.py build flash # Build and then flash the ESP32-C3-Zero connected to the computer.
```

Once you've set the target and set your configuration in menuconfig, you can just run the last command
from then on.

## Monitoring the program

If you are connected to the ESP-C3-Zero from a computer via its USB-C
port, you can monitor its output to see if things are working as they
should by running:

```sh
idf.py monitor
```

## Configuration

The program has LIMB specific settings that can be configured using `idf.py menuconfig`.

### Force reenabling CAN when entering a Bus Off state
The program can be configured to automatically reenable the CAN bus in case a Bus Off state occurs
by running `idf.py menuconfig` and checking `LIMB config ---> Force reenable CAN on bus off`.
The CAN bus is noisy enough that this module risks entering a Bus Off state regularly, so
it's probably a good idea to check it.

> [!warning]
> The Bus Off state happens for a reason, so be careful if you force reenable the bus.
> Read the [CAN driver components readme] for more information.

### Disable the IMU
The program can also be configured to not use the IMU by running `idf.py menuconfig` and making sure
`LIMB config -> Enable IMU` is unchecked.

> [!note]
> Disabling the IMU is useful in case it's disconnected from the ESP-C3-Zero for whatever reason,
> since the program normally won't start if it can't connect to the IMU.

## Location of the module on the arm

The module is located on the back of the hand, as shown in the image (the ESP-C3-Zero is located on the
backside of the PCB):

![Image of the hand of the robot arm with the elbow module on its back](/res/hand_pressure_module.jpg)

## PCB details

The files for the PCB of the module are located in the [`/designs/KiCad/LIMB_Hand/`](/designs/KiCad/LIMB_Hand/) directory.
