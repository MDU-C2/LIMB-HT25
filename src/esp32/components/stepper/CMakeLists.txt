idf_component_register(
  SRCS
  "src/stepper.c"
  REQUIRES
  adc_manager
  esp_adc
  driver
  freertos
  INCLUDE_DIRS
  "include")

