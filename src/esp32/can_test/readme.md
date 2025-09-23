A loopback test for a single ESP32-C3-Zero MC performed while shorting the GPIO 4 and 5 pins. Based on the [TWAI document](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-reference/peripherals/twai.html#hardware-connection) in the ESP-IDF programming guide.

## Usage
First set up ESP-IDF according to their [getting started guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/get-started/index.html).

Short the GPIO 4 and 5 pins on the ESP, connect it to your computer and run the following commands in this directory.
```sh
idf.py set-target esp32c3
idf.py build
idf.py flash monitor
```

If the monitor can't find the correct serial port, you can run:
```sh
idf.py flash monitor -p [path_to_port]
```

In my case running linux it's `/dev/ttyACM0`. If you run windows, maybe try looking [here](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/get-started/windows-setup.html#connect-your-device) for more info.

If successful, the output should look similar to:
```
Ready to do CAN IO!
Successful TX!
Received message: "Hello10"
Times entered callbacks: RX(1) TX(1) ERR(0)
Restarting in 10 seconds...
Successful TX!
Received message: "Hello10"
Times entered callbacks: RX(1) TX(1) ERR(0)
Restarting in 9 seconds...
```
