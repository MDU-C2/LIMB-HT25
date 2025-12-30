idf_component_register(SRCS "main.c"
                       REQUIRES adc_service imu_service ble_service
                       PRIV_REQUIRES freertos log)