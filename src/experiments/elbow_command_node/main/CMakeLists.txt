idf_component_register(
    SRCS
        "main.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        can_driver
        driver
        esp_driver_gpio
        freertos
)

