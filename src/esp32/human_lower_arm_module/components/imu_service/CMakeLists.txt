idf_component_register(
    SRCS "src/imu_service.c"
    INCLUDE_DIRS "include"
    REQUIRES imu
    PRIV_REQUIRES esp_timer
)