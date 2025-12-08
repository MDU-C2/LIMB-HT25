idf_component_register(
  SRCS
  "main.c"
  "AX_servo.c"
  PRIV_REQUIRES
  esp_driver_uart
  esp_driver_gpio
  esp_driver_ledc
  driver
  freertos
)

