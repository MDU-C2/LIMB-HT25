# Robot elbow node

This program is responsible for controlling the stepper motor for the elbow and reading from the IMU for the upper arm.

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
by running `idf.py menuconfig` and checking `LIMB config -> Force reenable CAN on bus off`.
The CAN bus is noisy enough that this module risks entering a Bus Off state regularly, so
it's probably a good idea to check it.

> [!warning]
> The Bus Off state happens for a reason, so be careful if you force reenable the bus.
> Read the [CAN driver components readme] for more information.

### Disabling the IMU
The program can also be configured to not use the IMU by running `idf.py menuconfig` and making sure
`LIMB config -> Enable IMU` is unchecked.

> [!note]
> Disabling the IMU is useful in case it's disconnected from the ESP-C3-Zero for whatever reason,
> since the program normally won't start if it can't connect to the IMU.

## Stepper motor

The stepper motor is a [Joy-IT NEMA17-04](https://joy-it.net/en/products/NEMA17-04)
controlled using a [Pololu DRV8825 stepper motor driver](https://www.pololu.com/product/2133).
The stepper driver is current limited to 1 A by setting the reference voltage to
0.5 V using the process explained on its [product
page](https://www.pololu.com/product/2133).

## Motor joint and speed limits in CAN actuation messages

The [stepper](/src/esp32/components/stepper/) component will automatically clamp
the angle and speed to the minimum and maximum limits, so it's safe to
send any joint angle and speed.

## CAN messages

The program receives CAN messages with a recipient ID of `CAN_RECIPIENT_ROBOT_ELBOW`:

| CAN message name                       | CAN ID |
| --                                     | --     |
| `CAN_ID_ROBOT_ELBOW_UP_DOWN_ACTUATION` | 0x240 |
| `CAN_ID_ROBOT_ELBOW_UP_DOWN_STOP`      | 0x140 |

Furthermore, it sends the CAN messages:

| CAN message name                           | CAN ID |
| --                                         | --     |
| `CAN_ID_ROBOT_ELBOW_UP_DOWN_POTENTIOMETER` | 0x4A0  |
| `CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_PITCH`    | 0x5A0  |
| `CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_ROLL`     | 0x5A1  |
| `CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_YAW`      | 0x5A2  |
| `CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_X`       | 0x5A3  |
| `CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Y`       | 0x5A4  |
| `CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Z`       | 0x5A5  |

## Location of the module on the arm

The module is located behind the front plate, as shown in the image:

![Image of the robot arm with the elbow module located inside its front plate](/res/elbow_module.jpg)

## PCB details

The files for the PCB of the module are located in the [`/designs/KiCad/LIMB_Biceps/`](/designs/KiCad/LIMB_Biceps/) directory.
