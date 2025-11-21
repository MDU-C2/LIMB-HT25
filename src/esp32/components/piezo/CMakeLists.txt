idf_component_register(
  SRCS
  "src/piezo.c"
  REQUIRES
  adc_manager
  esp_adc
  INCLUDE_DIRS
  "include")

