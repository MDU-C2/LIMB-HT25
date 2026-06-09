# Pointers for ESP-IDF programs
The general steps when you want to run a program is as follows:
```sh
idf.py set-target esp32c3 # Or whatever type of esp32 chip you use.
idf.py menuconfig # Make sure you enable all the features that your program uses.
idf.py build flash monitor # Actually build and monitor output.
```

If you want to use one of our components, add the path to the
`components` directory to your app's `CMakeLists.txt`, e.g.:
```cmake
cmake_minimum_required(VERSION 3.16)

# NOTE: Add the following variable to use our components.
# Make sure the path you pass is correct relative to the cmake project's root dir.
set(EXTRA_COMPONENT_DIRS "../../esp32/components")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# NOTE: If you add our components directory as shown above, you also need to
# enable a minimum build so that you only build the components you actually
# use. The limb_ble_periph component requires that NimBLE bluetooth is enabled
# to even compile, so if you don't have a minimal build the limb_ble_periph
# component will fail to build unless you've enabled the NimBLE bluetooth
# component in your app.
idf_build_set_property(MINIMAL_BUILD ON)

project(esp_project_name)
```

If you use a component that relies on some feature, make sure to use
`idf.py menuconfig` to enable all necessary configurations for your program.
For example, if your program uses NimBLE (directly or indirectly via a component),
you have to enable NimBLE through `Component config > Bluetooth > Bluetooth > Host > NimBLE - BLE only`.
The modules also have LIMB specific configurations in `idf.py menuconfig`
under `LIMB config` that should be taken into account.

As a general rule FreeRTOS should probably also be modified to use
a 1000 Hz tick rate via menuconfig: `Component config > FreeRTOS >
Kernel > configTICK_RATE_HZ = 1000`. This is set by default using
`Kconfig.projbuild` files for all modules.

Once the programs should be deployed, make sure to change the optimization level from debug via menuconfig:
`Compiler options > Optimization Level = Optimize for performance`.

