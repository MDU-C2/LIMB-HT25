cmake_minimum_required(VERSION 3.16)

# Set target to ESP32-C3
set(IDF_TARGET esp32c3)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(can_rx)

