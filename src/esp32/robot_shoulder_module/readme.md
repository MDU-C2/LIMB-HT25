# Robot shoulder node

This program is responsible for controlling three motors in the arm:

- Shoulder up/down modded continuous servo motor
- Shoulder left/right modded continuous servo motor
- Upper arm rotation stepper motor

> [!CAUTION]
> Make sure the system is not powered when you run `idf.py flash` or `idf.py monitor` on
> the shoulder node!
> When you flash or monitor, the ESP-C3-Zero ends up sending signals on the pins connected to the
> PWM data for the servos. This means the servos end up moving, and since they are modified to be
> continuous, they don't stop moving until the flashing is finished or the monitor is connected!

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

The program can be configured to automatically reenable the CAN bus in case a Bus Off state occurs
by running `idf.py menuconfig` and checking `LIMB config ---> Force reenable CAN on bus off`.
The CAN bus is noisy enough that this module risks entering a Bus Off state regularly, so
it's probably a good idea to check it.

> [!warning]
> The Bus Off state happens for a reason, so be careful if you force reenable the bus.
> Read the [CAN driver components readme] for more information.

## Servo motors

The servo motors are both of the model JX Servo PDI-HV2060MG. For the LIMB project, both servo motors have
been modified to support continuous rotation. As such, the
[continuous servo component](/src/esp32/components/continuous_servo/) should be used instead of the
[regular servo component](/src/esp32/components/servo/).

More information regarding the continuous rotation modification can be found
[in the documentation](/docs/servo_modification.md).

## Stepper motor

The stepper motor is a [Joy-IT NEMA17-04](https://joy-it.net/en/products/NEMA17-04)
controlled using a [Pololu DRV8825 stepper motor driver](https://www.pololu.com/product/2133).

## Motor joint and speed limits in CAN actuation messages

The [continuous servo](/src/esp32/components/continuous_servo/) and [stepper](/src/esp32/components/stepper/) components
will automatically clamp the angle and speed to the minimum and maximum limits, so it's safe to send
any joint angles and speeds.

## CAN messages

The program receives CAN messages with a recipient ID of `CAN_RECIPIENT_ROBOT_SHOULDER`:

| CAN message name                             | CAN ID |
| --                                           | --     |
| `CAN_ID_ROBOT_SHOULDER_UP_DOWN_STOP`         | 0x120  |
| `CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_STOP`      | 0x121  |
| `CAN_ID_ROBOT_UPPER_ARM_ROTATION_STOP`       | 0x122  |
| `CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION`    | 0x220  |
| `CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION` | 0x221  |
| `CAN_ID_ROBOT_UPPER_ARM_ROTATION_ACTUATION`  | 0x222  |

Furthermore, it sends the CAN messages:

| CAN message name                                | CAN ID |
| --                                              | --     |
| `CAN_ID_ROBOT_ELBOW_UP_DOWN_POTENTIOMETER`      | 0x4A0  |
| `CAN_ID_ROBOT_UPPER_ARM_ROTATION_POTENTIOMETER` | 0x4A1  |
| `CAN_ID_ROBOT_SHOULDER_UP_DOWN_POTENTIOMETER`   | 0x4A2  |

## Location of the module on the arm

The module is located on the back of the torso, as shown in the image:

![Image of the shoulder module located at the back of the torso of the robot arm](/res/shoulder_module.jpg)

## PCB details

The files for the PCB of the module are located in the [`/designs/KiCad/LIMB_Shoulder/`](/designs/KiCad/LIMB_Shoulder/) directory.
