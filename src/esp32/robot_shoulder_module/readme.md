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
had their internal potentiometers removed and replaced with $2.5\ \text{k}\Omega$ resistors FIXME: MORE DETAILS.
This has the effect of making the servo motor think its potentiometer is always centered, in our case corresponding
to a pulse width of $1500\ \mu\text{s}$. By writing a pulse width higher or lower than $1500\ \mu\text{s}$, the servo
will infinitely rotate either clockwise or counterclockwise. The magnitude of the difference between the pulse
width used and $1500\ \mu\text{s}$ determines the speed of rotation.

# Stepper motor

The stepper motor is a NEMA17-04 controlled using a DRV8825 stepper motor driver.
