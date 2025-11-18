idf_component_register(
  SRCS
  "src/potentiometer.c"
  REQUIRES
  driver
  esp_adc
  INCLUDE_DIRS
  "include")

