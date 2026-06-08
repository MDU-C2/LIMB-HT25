# Robot shoulder node

This program is responsible for controlling three motors in the arm:

- Shoulder up/down servo motor
- Shoulder left/right servo motor
- Upper arm rotation stepper motor

The program receives CAN messages with a recipient ID of `CAN_RECIPIENT_ROBOT_SHOULDER`, meaning:

- `CAN_ID_ROBOT_SHOULDER_UP_DOWN_ACTUATION`
- `CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_ACTUATION`
- `CAN_ID_ROBOT_UPPER_ARM_ROTATION_ACTUATION`
- `CAN_ID_ROBOT_SHOULDER_UP_DOWN_STOP`
- `CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_STOP`
- `CAN_ID_ROBOT_UPPER_ARM_ROTATION_STOP`

Furthermore, it sends the CAN messages:

- `CAN_ID_ROBOT_SHOULDER_UP_DOWN_POTENTIOMETER`
- `CAN_ID_ROBOT_SHOULDER_LEFT_RIGHT_POTENTIOMETER`
- `CAN_ID_ROBOT_UPPER_ARM_ROTATION_POTENTIOMETER`

# Servo motors

The servo motors are both of the model JX Servo PDI-HV2060MG. For the LIMB project, both servo motors have
been modified to support continuous rotation. As such, the
[continuous servo component](../components/continuous_servo/) should be used instead of the
[regular servo component](../components/servo/).

More information regarding the continuous rotation modification can be found
[in the documentation](../../../docs/servo_modification.md).

# Stepper motor

The stepper motor is a [Joy-IT NEMA17-04](https://joy-it.net/en/products/NEMA17-04)
controlled using a [Pololu DRV8825 stepper motor driver](https://www.pololu.com/product/2133).
