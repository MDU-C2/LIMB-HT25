# CAN message IDs and their rationale
This document describes which CAN messages are used in the system and what their IDs are.
It also goes into some more detail regarding the decisions behind the prioritization
of the messages along with calculations showing the feasibility of the network
with regards to potential message latency.

## CAN message ID allocation strategy
We want to allow for easy filtering of relevant messages. We also want to
prioritize different types of messages. By splitting up the 11 ID bits into
three sections representing the message type, the recipient node, and a generic ID
we're able to both prioritize and easily filter the messages.

The layout we use is the following, with $x$ belonging to the message type section,
$y$ belonging to the recipient section, and $z$ belonging to the generic section:
| Bit     | 10  | 9   | 8   | 7   | 6   | 5   | 4   | 3   | 2   | 1   | 0   |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Section | $x$ | $x$ | $x$ | $y$ | $y$ | $y$ | $z$ | $z$ | $z$ | $z$ | $z$ |

For the message types, we get the following masks:

| Message type  | Binary value               | Hexadecimal value |
|---------------|----------------------------|-------------------|
| Stop          | $\texttt{001\ XXXX\ XXXX}$ | $\texttt{0x1XX}$  |
| Actuation     | $\texttt{010\ XXXX\ XXXX}$ | $\texttt{0x2XX}$  |
| Potentiometer | $\texttt{100\ XXXX\ XXXX}$ | $\texttt{0x4XX}$  |
| IMU           | $\texttt{101\ XXXX\ XXXX}$ | $\texttt{0x5XX}$  |
| Pressure      | $\texttt{111\ XXXX\ XXXX}$ | $\texttt{0x7XX}$  |

For the recipients, we get the following masks:

| Node           | Binary value               | Hexadecimal value |
|----------------|----------------------------|-------------------|
| Robot shoulder | $\texttt{XXX\ 001X\ XXXX}$ | $\texttt{0xX2X}$  |
| Robot elbow    | $\texttt{XXX\ 010X\ XXXX}$ | $\texttt{0xX4X}$  |
| Robot wrist    | $\texttt{XXX\ 011X\ XXXX}$ | $\texttt{0xX6X}$  |
| Robot hand     | $\texttt{XXX\ 100X\ XXXX}$ | $\texttt{0xX8X}$  |
| Robot AGX Orin | $\texttt{XXX\ 101X\ XXXX}$ | $\texttt{0xXAX}$  |

This way we prioritize stop messages and actuation messages the most. We also
prioritize messages intended for the robot shoulder and robot elbows the most,
since they are the strongest motors and could cause the most damage if their
current movement target is not up to date.

## Robot messages
These are the messages that are send over the robot arm's CAN bus.

### Stop messages
| CAN ID | Message                                   |
| ------ | ----------------------------------------  |
| $\texttt{0x120}$ | Shoulder up/down servo stop     |
| $\texttt{0x121}$ | Shoulder left/right servo stop  |
| $\texttt{0x122}$ | Upper arm rotation stepper stop |
| $\texttt{0x140}$ | Elbow up/down stepper stop      |
| $\texttt{0x160}$ | Lower arm rotation servo stop   |
| $\texttt{0x161}$ | Finger servos stop              |
| $\texttt{0x162}$ | Thumb servo stop                |
| $\texttt{0x163}$ | Index servo stop                |
| $\texttt{0x164}$ | Middle servo stop               |
| $\texttt{0x165}$ | Ring servo stop                 |
| $\texttt{0x166}$ | Pinky servo stop                |

### Actuation messages
| CAN ID | Message                                        |
| ------ | ---------------------------------------------- |
| $\texttt{0x220}$ | Shoulder up/down servo actuation     |
| $\texttt{0x221}$ | Shoulder left/right servo actuation  |
| $\texttt{0x222}$ | Upper arm rotation stepper actuation |
| $\texttt{0x240}$ | Elbow up/down stepper actuation      |
| $\texttt{0x260}$ | Lower arm rotation servo actuation   |
| $\texttt{0x261}$ | Thumb servo actuation                |
| $\texttt{0x262}$ | Index servo actuation                |
| $\texttt{0x263}$ | Middle servo actuation               |
| $\texttt{0x264}$ | Ring servo actuation                 |
| $\texttt{0x265}$ | Pinky servo actuation                |
| $\texttt{0x266}$ | Hand set grip state                  |

### Potentiometer messages

| CAN ID | Message                                             |
| ------ | --------------------------------------------------- |
| $\texttt{0x4A0}$  | Elbow up/down stepper potentiometer      |
| $\texttt{0x4A1}$  | Upper arm rotation stepper potentiometer |
| $\texttt{0x4A2}$  | Shoulder up/down servo potentiometer     |
| $\texttt{0x4A3}$  | Shoulder left/right servo potentiometer  |

### IMU messages
| CAN ID            | Message                            |
| ------            | ---------------------------------- |
| $\texttt{0x5A0}$  | IMU robot upper arm gyro pitch     |
| $\texttt{0x5A1}$  | IMU robot upper arm gyro roll      |
| $\texttt{0x5A2}$  | IMU robot upper arm gyro yaw       |
| $\texttt{0x5A3}$  | IMU robot upper arm acceleration x |
| $\texttt{0x5A4}$  | IMU robot upper arm acceleration y |
| $\texttt{0x5A5}$  | IMU robot upper arm acceleration z |
| $\texttt{0x5A6}$  | IMU robot lower arm gyro pitch     |
| $\texttt{0x5A7}$  | IMU robot lower arm gyro roll      |
| $\texttt{0x5A8}$  | IMU robot lower arm gyro yaw       |
| $\texttt{0x5A9}$  | IMU robot lower arm acceleration x |
| $\texttt{0x5AA}$  | IMU robot lower arm acceleration y |
| $\texttt{0x5AB}$  | IMU robot lower arm acceleration z |
| $\texttt{0x5AC}$  | IMU robot hand gyro pitch          |
| $\texttt{0x5AD}$  | IMU robot hand gyro roll           |
| $\texttt{0x5AE}$  | IMU robot hand gyro yaw            |
| $\texttt{0x5AF}$  | IMU robot hand acceleration x      |
| $\texttt{0x5B0}$  | IMU robot hand acceleration y      |
| $\texttt{0x5B1}$  | IMU robot hand acceleration z      |

### Pressure messages
| CAN ID | Message                                  |
| ------ | ---------------------------------------- |
| $\texttt{0x7A0}$  | Thumb pressure sensor                    |
| $\texttt{0x7A1}$  | Index pressure sensor                    |
| $\texttt{0x7A2}$  | Middle pressure sensor                   |
| $\texttt{0x7A3}$  | Ring pressure sensor                     |
| $\texttt{0x7A4}$  | Pinky pressure sensor                    |
