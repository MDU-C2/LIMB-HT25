
## Preparations before controlling the arm

Before attempting to control the arm, there's some setup that you will
have to do.

- Make sure that the ESP32s on the arm are flashed with the correct programs (the programs are found in the [`esp32/`](./esp32) directory).
- [Set up the CAN interface on the Orin](#setting-up-the-can-interface).

## Setting up the CAN interface

In the [scripts folder](../scripts/) there is a file called [`agx_setup_can.sh`](../scripts/agx_setup_can.sh).
Running the script on the AGX Orin sets up two virtual CAN interfaces according to NVIDIA's developer guide,
`can0` and `can1`. Check which pins in the 40-pin expansion header the CAN transceiver uses
to know which interface is used (`CAN0_DIN`, `CAN0_DOUT` for `can0` and `CAN1_DIN`, `CAN1_DOUT` for `can1`.
The pin mapping can be found in
[section 3.3 of the Jetson AGX Orin Module Carrier Board specification](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/agx_orin/jetson_agx_orin_devkit_carrier_board_specification_sp#page=37))

## Python on the AGX Orin

The AGX Orin is running an old version of Ubuntu (20.04.6). As such, the python version available through `apt` is also old.
To make sure we are actually able to use newer python libraries, we use [`uv`](https://docs.astral.sh/uv/) to install python 3.10.18 and
manage the virtual environment to run the python programs.

## Running the main program

Running python programs is done by running `uv run your-python-program.py`.
The main program is located in [`layers/`](./layers/). For more information about what it does, check its [readme](./layers/README.md).
Run it on the AGX Orin using `uv run layers/main.py`, optionally with arguments at the end of the command.

