# CAN Message data layouts
This document shows the basic data layout for the different types of
CAN messages.

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
| value | lo | mid lo | mid hi | hi | 0 | 0 | 0 | 0 |

## Pressure sensors
Little endian 2-byte integer value.

### Byte representation of CAN message data field (8 bytes)
| byte  |  0 |  1 | 2 | 3 | 4 | 5 | 6 | 7 |
| ----- | -- | -- | - | - | - | - | - | - |
| value | lo | hi | 0 | 0 | 0 | 0 | 0 | 0 |

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
| value | float lo | float mid lo | float mid hi | float hi | 0    | 0    | 0 | 0 |

## EMGs
Little endian 2-byte integer value.

### Byte representation of CAN message data field (8 bytes)
| byte  |  0 |  1 | 2 | 3 | 4 | 5 | 6 | 7 |
| ----- | -- | -- | - | - | - | - | - | - |
| value | lo | hi | 0 | 0 | 0 | 0 | 0 | 0 |

## Intelligent gripper commands

Since the higher level commands for the intelligent gripper just tell it to
change state, the reception of the message is enough information for the
gripper to act upon it. Therefore the message itself can be empty.
