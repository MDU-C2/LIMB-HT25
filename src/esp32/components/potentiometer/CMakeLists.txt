idf_component_register(
  SRCS
  "src/potentiometer.c"
  REQUIRES
  adc_manager
  esp_adc
  INCLUDE_DIRS
  "include")

