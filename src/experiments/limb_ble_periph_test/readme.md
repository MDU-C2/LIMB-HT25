A test program for setting up a Bluetooth LE peripheral that supports reading
and being notified of mock data.

Steps for running program:
```sh
idf.py set-target esp32c3
idf.py menuconfig # Enable BT and NimBLE.
idf.py build flash monitor
```

Then connect to the peripheral using a BLE central device. You can read or
be notified of the different characteristics.
