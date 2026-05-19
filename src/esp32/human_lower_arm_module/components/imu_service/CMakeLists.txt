idf_component_register(
    SRCS "src/imu_service.c"
    INCLUDE_DIRS "include"
    REQUIRES imu limb_ble_periph limb_utils
    PRIV_REQUIRES esp_timer
)
