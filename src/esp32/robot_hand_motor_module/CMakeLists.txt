cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "../components")

# Incluye las herramientas de construcción del IDF
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

idf_build_set_property(MINIMAL_BUILD ON)
# Define el nombre del proyecto
project(servo_control)
