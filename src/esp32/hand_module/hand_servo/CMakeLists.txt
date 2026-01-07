cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "../../components")

# Incluye las herramientas de construcción del IDF
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Define el nombre del proyecto
project(servo_control)