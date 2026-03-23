idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES log freertos nvs_flash limb_emg_adc limb_imu ble_service)