idf_component_register(SRCS "src/emg_driver.c" "src/emg_service.c"
                       INCLUDE_DIRS "include"
                       REQUIRES freertos esp_system
                       PRIV_REQUIRES log esp_adc)