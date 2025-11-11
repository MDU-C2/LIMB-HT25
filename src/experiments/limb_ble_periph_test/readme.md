A test program for setting up a Bluetooth LE peripheral that supports reading
and being notified of mock data.

Just run the program using `idf.py build flash monitor` (or using your `idf.py`
incantation of choice) and then connect to it using a BLE central device. Then
you can read or be notified of its mock EMG, IMU and piezo values.
