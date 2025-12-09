idf_component_register(SRCS "src/ble_service.c"
                       INCLUDE_DIRS "include"
                       REQUIRES freertos
                       PRIV_REQUIRES log nvs_flash bt driver limb_ble_periph limb_emg_adc limb_imu)