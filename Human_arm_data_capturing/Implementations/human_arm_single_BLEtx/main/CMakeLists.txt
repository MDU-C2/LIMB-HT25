idf_component_register(SRCS "main.c"
                            "adc_emg_driver.c"
                            "imu_driver.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_adc driver esp_system log freertos esp_timer limb_ble_periph nvs_flash bt)