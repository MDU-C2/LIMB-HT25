idf_component_register(SRCS "main.c" 
                            "adc_emg_driver.c" 
                            "imu_driver.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp_adc driver esp_system log freertos esp_timer)