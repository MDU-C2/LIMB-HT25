# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "../../esp32/components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(limb_ble_periph_test)
