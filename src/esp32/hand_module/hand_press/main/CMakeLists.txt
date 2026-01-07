idf_component_register(SRCS "main.c"
                       REQUIRES adc_service can_driver
                       PRIV_REQUIRES freertos log)