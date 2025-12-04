idf_component_register(
  SRCS
  "main.c"
  PRIV_REQUIRES
  imu
  potentiometer
  driver
  freertos
  esp_driver_twai
)

