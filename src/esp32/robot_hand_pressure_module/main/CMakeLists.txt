idf_component_register(SRCS "main.c"
                       REQUIRES adc_service can_driver imu limb_utils
                       PRIV_REQUIRES freertos log)
