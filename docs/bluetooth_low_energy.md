# Bluetooth Low Energy

The ESP32-C3-Zero on the human arm communicates with the NVIDIA Jetson AGX Orin using
Bluetooth 5 Low Energy (BLE) notifications, providing characteristics for EMG, IMU, and
piezo data.

## Crash course in BLE

BLE is built around the idea of periodic connection events where two
devices, a Central and a Peripheral, have the opportunity to alternate
sharing data between each other.

The Peripheral creates what are called &ldquo;characteristics,&rdquo;
which represent something that the Central can read from and/or write
to. A characteristic can, for example, represent the latest EMG sensor
value provided by the Peripheral. These characteristics can also be
subscribed to, meaning that the Central will end up receiving the
latest value that can be read from the characteristic as soon as the
Peripheral can send it to the Central, without the Central making an
explicit read request.

The period at which connection events occur between the Central and
Peripheral is called the connection interval. At the start of every
connection event, the Central sends a packet to the Peripheral. If the
Central wants to read from a characteristic, this packet could signify
a read request, for example. The Peripheral then responds with a packet
based on what the Central sent, which could be the data contained in the
characteristic, for example, if the Central sent a read request or has
previously subscribed to the characteristic. The Central and Peripheral
then alternate sending (potentially empty) packets between each other
until either the end of the connection event or there are no more packets
to send from either device.

More information about connection events can be found in section 4.5,
Vol 6, Part B of the
[Bluetooth core specification](https://www.bluetooth.com/specifications/specs/core-specification-5-4/).

## Parts of the code that acts as the BLE Central and Peripheral

The Python code for the NVIDIA Jetson AGX Orin that acts as the BLE Central can be found in
[src/ble_central](../src/hardware/ble/). The C code that acts as the BLE Peripheral can be found
in [src/esp32/components/limb_ble_periph/](../src/esp32/components/limb_ble_periph/).

## Lowering latency

We want the connection between the human sensors and the robot arm
to be as responsive as possible, and as such
we want to lower the latency of the BLE communication as much as
possible. To achieve this goal we do multiple things.

- Increase the bitrate using 2M Phy.
- Increase the LL Data PDU length using LE Data Packet Length Extension.
- Increase the ATT MTU.
- Use notifications to avoid resending dropped packets.
- Lower the connection interval.

By default, the LL Data PDU length is 27 B, meaning that trying to send
packets larger than 27 B will actually split them up into multiple packets
with a maximum size of 27 B. This limit can be increased by enabling LE
Data Packet Length Extension and setting the maximum LL Data PDU length
to a larger value (max is 251 B as per Table 4.6 in Vol 6, Part B of the
[Bluetooth core specification](https://www.bluetooth.com/specifications/specs/core-specification-5-4/)).

The ATT MTU is the maximum size of the ATT packet contained in an L2CAP packet.
Given a maximum LL Data PDU length of 251 B, the maximum size of the L2CAP packet is 251
B, with 4 of those bytes being headers, meaning that the maximum size of the ATT MTU is
251 B - 4 B = 247 B.

## Traffic calculations

Given a bit rate of 2 Mbps and a connection interval of 10 ms with
frequencies of 4 kHz for the EMG, 100 Hz for the IMU, and 1 kHz for
the piezo, these calculations show the viability of sending the sensor
data as notifications for each period over BLE. The sensor data gets
encapsulated in multiple headers before being sent over BLE. First an
ATT header of 3 B is added
[[Table 3.38, Vol 3, Part F]](https://www.bluetooth.com/specifications/specs/core-specification-5-4/).
This is followed by an L2CAP header of 4 B
[[Figure 3.1, Vol 3, Part A]](https://www.bluetooth.com/specifications/specs/core-specification-5-4/),
an LL Data PDU header of 2 B
[[Figure 2.24, Vol 6, Part B]](https://www.bluetooth.com/specifications/specs/core-specification-5-4/),
and finally an LL packet header of 9 B
[[Figure 2.1, Vol 6, Part B]](https://www.bluetooth.com/specifications/specs/core-specification-5-4/)
for a total of 18 extra bytes.

The sizes of the different packets are shown in the table below:

| Data                                                | Size |
| ----                                                | ----                        |
| EMG data + metadata size per packet                 | 80 B + 4 B = 84 B           |
| Encapsulated as EMG LL packet                       | 84 B + 18 B = 102 B.        |
| IMU data + metadata size per packet                 | 24 B + 4 B = 28 B.          |
| Encapsulated as IMU LL packet                       | 28 B + 18 B = 46 B.         |
| Piezo data + metadata size per packet               | 20 B + 4 B = 24 B.          |
| Encapsulated as piezo LL packet                     | 24 B + 18 B = 42 B.         |
| Empty packet size (LL Data PDU + LL packet headers) | 2 B + 9 B = 11 B.           |

Since all values are below 251 B, they fit within a single LL packet each. The
transmission delay of each individual packet is shown in the table below:

| Data                | Transmission delay                                                                                         |
| ---                 | --                                                                                                         |
| EMG                 | 102 B / 2 Mbps = 408 &mu;s                                                                                 |
| IMU                 | 46 B / 2 Mbps = 184 &mu;s                                                                                  |
| Piezo               | 42 B / 2 Mbps = 168 &mu;s                                                                                  |
| Empty packet        | 11 B / 2 Mbps = 44 &mu;s                                                                                   |
| Inter-frame spacing | 150 &mu;s [[4.1.1, Vol 6, Part B]](https://www.bluetooth.com/specifications/specs/core-specification-5-4/) |

Assuming a normal connection event when the Central is subscribed to
notifications for all sensors, the Central will send empty packets and
the Peripheral will send EMG, IMU, and piezo data packets. An example of a
full transmission sequence of packets during the connection event is then:

`EMPTY + IFS + EMG + IFS + EMPTY + IFS + IMU + IFS + EMPTY + IFS + PIEZO + IFS + EMPTY`

The time it takes to send all packets in the connection event is then:

$$
4 \cdot 44\ \mu\text{s} + 6 \cdot 150\ \mu\text{s} + 408\ \mu\text{s} + 184\ \mu\text{s} + 168\ \mu\text{s} = 1.836\ \text{ms}
$$

1.836 ms is well within the deadline of 10 ms, so the amount of data
being sent should not be a problem.
