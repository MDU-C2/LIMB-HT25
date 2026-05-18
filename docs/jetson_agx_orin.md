# NVIDIA Jetson AGX Orin Developer Kit
The device used for coordinating and controlling the rest of the system is an
[NVIDIA Jetson AGX Orin Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/).

## Connecting to the Orin
You can access the Orin either by connecting it to a keyboard, mouse, and monitor, or by connecting to it using `ssh`.

### Connecting via ssh
You can use the USB-C port next to the 40-pin header to
[connect to the Orin using a virtual ethernet connection](https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/howto.html#up-stream-facing-port-ufp).
The Orin will be given the IP address 192.168.55.1 and your computer will be given the address 192.168.55.100.
You should then be able to `ssh` into the Orin using `ssh bigboyorin@192.168.55.1` and entering its password.

Connecting the Orin to your computer using the USB-C port next to the 40-pin header also makes the
Orin show up as a storage device that contains, among other things, some readme files regarding
setting up the Orin. If you require more information, that is one place to check.

## Running python
The Orin uses an old version of Ubuntu without access to newer python versions. See [src/README.md](../src/README.md#python-on-the-agx-orin) for more information.

## Connecting the Orin to a CAN bus
The Orin has two CAN controllers built-in, but no CAN transceivers.
Once the CAN transceiver has been connected to the correct pins in the
[40-pin header](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/agx_orin/jetson_agx_orin_devkit_carrier_board_specification_sp#page=37),
the correct registers have to be written to, CAN kernel drivers have to be enabled,
and a virtual CAN interface has to be created. This process is taken care of by the
[scripts/agx_setup_can.sh](../scripts/agx_setup_can.sh) script, creating the `can0`
interface for the CAN0 pins and the `can1` interface for the CAN1 pins.

More information regarding setting up and using CAN on the Orin can be found in the
[NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/HR/ControllerAreaNetworkCan.html).

## Connecting to the internet
The Orin has an Ethernet port that can be used to connect it to the internet. It also has Wi-Fi
functionality, but it doesn't support WPA2-Enterprise which is what eduroam uses to secure the connection,
meaning you'll have to find some other way to connect it over Wi-Fi (such as a mobile hotspot).

## More info
You can find more information in the
[NVIDIA Jetson AGX Orin Developer Kit Carrier Specification](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/agx_orin/jetson_agx_orin_devkit_carrier_board_specification_sp#page=37)
and the [NVIDIA Jetson AGX Orin Developer Kit User Guide](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html).
