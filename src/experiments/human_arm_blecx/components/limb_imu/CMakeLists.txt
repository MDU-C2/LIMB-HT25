idf_component_register(SRCS "src/imu_driver.c" "src/imu_service.c"
                       INCLUDE_DIRS "include"
                       REQUIRES freertos esp_system
                       PRIV_REQUIRES log driver esp_timer)