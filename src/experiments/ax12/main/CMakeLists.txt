idf_component_register(
  SRCS
  "main.c"
  PRIV_REQUIRES
  esp_driver_uart
  esp_driver_gpio
  esp_driver_ledc
  driver
  freertos
)

