# CAN Message data layouts
This document shows the basic data layout for the different types of
CAN messages. An x in the byte representation means the byte is unused.

## Motor stops
Motor stops don't send any data, just receiving the message is the
indication to immediately stop the motor.

## Motor actuations
Little endian 4-byte float angle and little endian 4-byte float velocity.

### Byte representation of CAN message data field (8 bytes)
| byte  |  0       |      1       |      2       |  3       | 4           | 5               | 6               | 7      |
| ----- | --       | ------       | ------       | --       | -           | -               | -               | -      |
| value | ang lo   | ang mid lo   | ang mid hi   | ang hi   | vel lo      | vel mid lo      | vel mid hi      | vel hi |

## Potentiometers
Little endian 4-byte float angle.

### Byte representation of CAN message data field (8 bytes)
| byte  |  0 |      1 |      2 |  3 | 4 | 5 | 6 | 7 |
| ----- | -- | ------ | ------ | -- | - | - | - | - |
| value | lo | mid lo | mid hi | hi | x | x | x | x |

## Pressure sensors
Little endian 2-byte integer value representing millivolts.

### Byte representation of CAN message data field (8 bytes)
| byte  |  0 |  1 | 2 | 3 | 4 | 5 | 6 | 7 |
| ----- | -- | -- | - | - | - | - | - | - |
| value | lo | hi | x | x | x | x | x | x |

## IMUs
6 Little endian 4-byte float values.

The IMU samples consist of two triplets: (pitch, roll, yaw) for the gyro and (x, y, z)
for the accelerometer. Each value in a triplet is a 4-byte floating point value,
meaning we have 12 bytes per triplet for a total of 24 bytes. Since a
CAN message is limited to 8 bytes of data, we have to split the samples into multiple
messages. For simplicity's sake, we're letting each of the six floating point values be its own
CAN message.

### Byte representation of CAN message data field (8 bytes)
| byte  |    0     |    1         |    2         |    3     |    4 |    5 | 6 | 7 |
| ----- | -------- | ------------ | ------------ | -------- | ---- | ---- | - | - |
| value | float lo | float mid lo | float mid hi | float hi | x    | x    | x | x |

## Intelligent gripper commands

Single byte representing a boolean value for if the hand should be in its intelligent grip state or not.

### Byte representation of CAN message data field (8 bytes)
| byte  | 0    | 1  | 2  | 3 | 4 | 5 | 6 | 7 |
| -     | -    | -  | -  | - | - | - | - | - |
| value | bool | x  | x  | x | x | x | x | x |
