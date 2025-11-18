# Components
This directory contains all the ESP-IDF components created for this project. As long as
you follow the directions in the `esp32` directory's readme regarding adding this `components`
directory to your app's `EXTRA_COMPONENT_DIRS`, you will have access to all the components
in this directory in your app.

## How to create and use components
At its core, a component is defined by a `CMakeLists.txt` file with the following contents:
```cmake
idf_component_register(
  SRCS # After this we declare all source files used by the component.
    src1.c
    src2.c
  INCLUDE_DIRS # After this we declare all directories containing public include files.
    include_directory
  REQUIRES # After this we declare all components used as dependencies by the component.
    component_name
)
```

More information about what arguments can be provided to `idf_component_register`
can be found [here](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/build-system.html#cmake-component-register).

A typical directory structure can look something like this:
```
.
├── CMakeLists.txt
├── include
│   ├── gap.h
│   ├── limb_ble_periph.h
│   └── sensors_service.h
├── readme.md
└── src
    ├── gap.c
    ├── limb_ble_periph.c
    └── sensors_service.c
```
with `CMakeLists.txt` having the contents:
```cmake
idf_component_register(
  SRCS
    src/limb_ble_periph.c
    src/sensors_service.c
    src/gap.c
  REQUIRES
    bt
  PRIV_REQUIRES
    nvs_flash
    esp_driver_gpio
  INCLUDE_DIRS
    include
)
```
