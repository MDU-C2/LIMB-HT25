# CAN

A CAN bus is used to allow the NVIDIA Jetson AGX Orin to communicate with the ESP32-C3-Zeros
distributed on the robot arm.

## CAN version
The CAN version used for this project is CAN 2.0, meaning only classical
frames supporting up to 8 bytes of data per message are used. This is
a hard limit since the ESP32-C3-Zeros used in the project
[don't support FD or XL frames](https://docs.espressif.com/projects/esp-idf/en/v5.4.4/esp32c3/api-reference/peripherals/twai.html#overview).
Regarding the CAN IDs, standard frames allowing for 11-bit IDs are used
instead of extended frames allowing for 29-bit IDs. However, this is
not a hard limit and could be changed in the future if necessary.

## CAN bus connections
The CAN bus is composed of a twisted pair running along the arm.
The Orin and all ESP32-C3-Zeros are connected to the CAN bus using
SN65HVD232D CAN transceivers ([datasheet](https://www.ti.com/lit/gpn/sn65hvd232)),
with the transceivers at both ends of the bus containing 120&Omega; terminating resistors.

## CAN bitrate
Since the CAN bus is on the order of 1 meter, we're able to use
a bitrate of 1 Mbit/s
([Source, see Table 1](https://www.ti.com/lit/an/slla270/slla270.pdf)).

## Using CAN with the NVIDIA Jetson AGX Orin Developer Kit

See [docs/jetson_agx_orin.md](../jetson_agx_orin.md) for information
regarding connecting the Orin to a CAN bus.

## Using CAN with the ESP32-C3-Zero

The ESP32-C3-Zeros support something called Two-Wire Automotive Interface (TWAI),
which is compatible with CAN frames, so it's basically CAN by another name. The
[TWAI driver](https://docs.espressif.com/projects/esp-idf/en/v5.4.4/esp32/api-reference/peripherals/twai.html)
in ESP-IDF is used to create the component in
[src/esp32/components/can_driver/](../../src/esp32/components/can_driver/) which
we use to send and receive messages over the CAN bus.

> [!note]
> The ESP-IDF TWAI driver used is from ESP-IDF v5.4.4, not the updated one from v5.5.1.

## Controlling the robot arm using CAN
Each joint connected to a motor can be controlled by sending
actuation CAN messages over the CAN bus. An overview
of the CAN message IDs used and their data layout can be
found in [docs/can/message_ids.md](./message_ids.md) and
[docs/can/message_data_layout.md](./message_data_layout.md).
In case the documentation ends up out of date, the definitions for the
message IDs in the codebase are in
[src/esp32/components/can_driver/include/can_driver.h](../../src/esp32/components/can_driver/include/can_driver.h)
and the data layout is defined by where the CAN messages are received
in the relevant module in [src/esp32/](../../src/esp32/).

In practice, the way you will end up sending the CAN messages is from
the Orin. Once you've set up the CAN network interface on the Orin
([there's a script you can run](../../scripts/agx_setup_can.sh)),
you can send CAN messages over the bus using the interface.

As an example, the elbow joint is controlled by sending a CAN
message with the ID 0x240 with the data being a little endian float
value for the angle in degrees followed by another little endian
float value for the angular velocity in degrees per second. The
limits for the supported angles and velocity can be found in the
configuration structs used when initializing the stepper motor in
[src/esp32/robot_elbow_module/](../../src/esp32/robot_elbow_module/).
As long as the configuration is correct, the received angles and velocities
should be clamped to the supported limits, meaning that if you send an angle
of 60&deg; to a motor that is configured to only allow a maximum of 50&deg;,
the angle targeted will be 50&deg;.

Here is a simple example python program using
[python-can](https://github.com/hardbyte/python-can) that sends an
actuation CAN message to the elbow stepper:
```python
import struct

import can

# Open a connection to the CAN interface.
# NOTE: Make sure the bitrate is the same as the other microcontrollers
# connected to the CAN bus use (1 Mbit/s in our case).
with can.Bus(interface='socketcan', channel='can0', bitrate=1_000_000) as bus:
    ELBOW_ACTUATION_CAN_ID = 0x240
    elbow_angle_deg = 40.0
    elbow_velocity_dps = 20.0
    # Making sure that the floats are packed in 8 bytes in little endian byte order.
    can_message_data = struct.pack('<ff', elbow_angle_deg, elbow_velocity_dps)
    msg = can.Message(
        arbitration_id=ELBOW_ACTUATION_CAN_ID,
        data=can_message_data,
        is_extended_id=False,
    )

    try:
        bus.send(msg)
        print("Successfully sent message on the CAN bus")
    except can.CanError as e:
        print(f"Some error occurred: {e}")
```

## Receiving CAN messages from the robot arm
There are various sensors on the robot arm that you might want to use. These can be received
on the Orin in a similar fashion to how actuation messages are sent in the previous section.
The possible CAN message IDs are in [docs/can/message_ids.md](./message_ids.md) and their
data layouts are in [docs/can/message_data_layout.md](./message_data_layout.md).

Here is a simple example python program using
[python-can](https://github.com/hardbyte/python-can) that reads the
elbow joint's current angle from a potentiometer CAN message.
```python
import struct

import can

# Open a connection to the CAN interface.
# NOTE: Make sure the bitrate is the same as the other microcontrollers
# connected to the CAN bus use (1 Mbit/s in our case).
with can.Bus(interface='socketcan', channel='can0', bitrate=1_000_000) as bus:
    try:
        msg = bus.recv()
        ELBOW_POTENTIOMETER_CAN_ID = 0x4A0
        if msg.arbitration_id == ELBOW_POTENTIOMETER_CAN_ID:
            # There's a trailing comma since unpack returns a tuple even though
            # it's only one value,
            elbow_angle, = struct.unpack('<f', msg.data)
            print(f"Successfully received angle {elbow_angle} degrees on the CAN bus")
    except can.CanError as e:
        print(f"Some error occurred: {e}")
```

