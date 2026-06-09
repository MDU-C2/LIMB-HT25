# Hand motor module

This is the program that controls the servo motors for the fingers and wrist rotation.
Additionally, it reads IMU values for the lower arm.

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

## CAN messages

The program responds to the CAN messages with `CAN_RECIPIENT_ROBOT_WRIST` as the recipient:

| CAN message name                            | CAN ID |
| --                                          | --     |
| `CAN_ID_ROBOT_LOWER_ARM_ROTATION_ACTUATION` | 0x260  |
| `CAN_ID_ROBOT_THUMB_ACTUATION`              | 0x261  |
| `CAN_ID_ROBOT_INDEX_ACTUATION`              | 0x262  |
| `CAN_ID_ROBOT_MIDDLE_ACTUATION`             | 0x263  |
| `CAN_ID_ROBOT_RING_ACTUATION`               | 0x264  |
| `CAN_ID_ROBOT_PINKY_ACTUATION`              | 0x265  |

The program sends the CAN messages:

| CAN message name                            | CAN ID |
| --                                          | --     |
| `CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_PITCH`     | 0x5A6  |
| `CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_ROLL`      | 0x5A7  |
| `CAN_ID_ROBOT_LOWER_ARM_IMU_GYRO_YAW`       | 0x5A8  |
| `CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_X`        | 0x5A9  |
| `CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_Y`        | 0x5AA  |
| `CAN_ID_ROBOT_LOWER_ARM_IMU_ACCEL_Z`        | 0x5AB  |

## Servo motors

The fingers are controlled using [HS422 servo motors](https://hitecrcd.com/hs-422-deluxe-standard-servo/)
while the wrist is controlled using a
[Whadda WPK601 servo](https://www.velleman.eu/products/view/270-robot-digital-double-shaft-servo-kit-wpk601/?id=460528).

## Location of the module on the arm

The module is located in the triceps, as shown in the image:

![Image of the robot arm with the hand motor module located in its triceps](/res/hand_motor_module.jpg)

## PCB details

The files for the PCB of the module are located in the [`/designs/KiCad/LIMB_Lowerarm/`](/designs/KiCad/LIMB_Lowerarm/) directory.
