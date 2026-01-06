idf_component_register(SRCS "src/adc_service.c"

                       INCLUDE_DIRS "include"

                       REQUIRES freertos esp_system adc_manager limb_utils
                       
                       PRIV_REQUIRES log esp_timer)
