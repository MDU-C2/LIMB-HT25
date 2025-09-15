cmake_minimum_required(VERSION 3.12...3.20)

project(
  limb-bionic-arm
  VERSION 0.1.0
  DESCRIPTION
    "An intelligent bionic arm controlled using EMG, IMU, vision, pressure, and slip sensors."
  HOMEPAGE_URL "https://github.com/MDU-C2/LIMB-HT25"
  LANGUAGES C CXX)

add_executable(limb-bionic-arm src/hello_world.cc)

target_include_directories(
  limb-bionic-arm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(limb-bionic-arm PUBLIC cxx_std_11)
set_target_properties(limb-bionic-arm PROPERTIES CXX_EXTENSIONS OFF)

target_compile_options(
  limb-bionic-arm
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-Wall
    -Wextra
    -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4
    /permissive->)
