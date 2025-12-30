idf_component_register(SRCS "src/ble_service.c"
                       INCLUDE_DIRS "include"
                       REQUIRES freertos
                       PRIV_REQUIRES log nvs_flash limb_ble_periph adc_service imu_service esp_timer)