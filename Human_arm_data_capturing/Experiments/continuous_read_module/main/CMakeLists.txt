idf_component_register(SRCS "adc_emg_driver.c" "continuous_read_main.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_adc esp_system freertos log)


                
