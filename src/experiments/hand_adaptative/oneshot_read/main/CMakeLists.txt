idf_component_register(SRCS "can_driver.c" "oneshot_read_main.c"
                    PRIV_REQUIRES esp_adc esp_timer
                    REQUIRES driver
                    INCLUDE_DIRS ".")
