#include "can_driver.h"
#include "esp_log.h"

static const char *TAG = "CAN_DRIVER";

esp_err_t can_init(int tx_pin, int rx_pin, int baudrate) {
    // General configuration
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(tx_pin, rx_pin, TWAI_MODE_NORMAL);
    g_config.tx_queue_len = 5;
    g_config.rx_queue_len = 5;

    // Timing configuration based on baudrate
    twai_timing_config_t t_config;
    if (baudrate == 1000000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_1MBITS();
        t_config = temp;
    } else if (baudrate == 800000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_800KBITS();
        t_config = temp;
    } else if (baudrate == 500000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_500KBITS();
        t_config = temp;
    } else if (baudrate == 250000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_250KBITS();
        t_config = temp;
    } else if (baudrate == 125000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_125KBITS();
        t_config = temp;
    } else if (baudrate == 100000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_100KBITS();
        t_config = temp;
    } else if (baudrate == 50000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_50KBITS();
        t_config = temp;
    } else if (baudrate == 25000) {
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_25KBITS();
        t_config = temp;
    } else {
        // Default to 500kbps
        twai_timing_config_t temp = TWAI_TIMING_CONFIG_500KBITS();
        t_config = temp;
    }

    // Filter configuration (accept all messages)
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

    // Install and start TWAI driver
    esp_err_t ret = twai_driver_install(&g_config, &t_config, &f_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install TWAI driver");
        return ret;
    }

    ret = twai_start();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start TWAI driver");
        return ret;
    }

    ESP_LOGI(TAG, "CAN initialized (TX=%d, RX=%d, %d bps)", tx_pin, rx_pin, baudrate);
    return ESP_OK;
}

esp_err_t can_send(uint32_t id, const uint8_t *data, uint8_t len) {
    twai_message_t message;
    message.identifier = id;
    message.data_length_code = len;
    message.flags = 0;  // Standard frame, data frame
    
    for (int i = 0; i < len && i < 8; i++) {
        message.data[i] = data[i];
    }
    
    return twai_transmit(&message, pdMS_TO_TICKS(1000));
}

esp_err_t can_receive(uint32_t *id, uint8_t *data, uint8_t *len, int timeout_ms) {
    twai_message_t message;
    esp_err_t ret = twai_receive(&message, pdMS_TO_TICKS(timeout_ms));
    
    if (ret == ESP_OK) {
        *id = message.identifier;
        *len = message.data_length_code;
        
        for (int i = 0; i < message.data_length_code && i < 8; i++) {
            data[i] = message.data[i];
        }
    }
    
    return ret;
}

void can_deinit(void) {
    twai_stop();
    twai_driver_uninstall();
    ESP_LOGI(TAG, "CAN deinitialized");
}
