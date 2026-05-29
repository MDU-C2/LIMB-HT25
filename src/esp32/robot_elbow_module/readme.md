# Robot elbow node

This program is responsible for controlling the stepper motor for the elbow and reading from the IMU for the upper arm.

The program receives CAN messages with a recipient ID of `CAN_RECIPIENT_ROBOT_ELBOW`, meaning:

- `CAN_ID_ROBOT_ELBOW_UP_DOWN_ACTUATION`
- `CAN_ID_ROBOT_ELBOW_UP_DOWN_STOP`

Furthermore, it sends the CAN messages:

- `CAN_ID_ROBOT_ELBOW_UP_DOWN_POTENTIOMETER`
- `CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_PITCH`
- `CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_ROLL`
- `CAN_ID_ROBOT_UPPER_ARM_IMU_GYRO_YAW`
- `CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_X`
- `CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Y`
- `CAN_ID_ROBOT_UPPER_ARM_IMU_ACCEL_Z`

# Stepper motor

The stepper motor is a NEMA17-04 controlled using a DRV8825 stepper motor driver.
