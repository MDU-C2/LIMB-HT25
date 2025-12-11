#include "can_driver.h"
#include "esp_log.h"
#include "driver/twai.h"

static const char *TAG = "CAN_DRIVER";


uint16_t create_filter_mask(CanMessageTypeFilterMask msg_type_filter_mask,
                            CanRecipientNodeFilterMask recipient_node_filter_mask,
                            CanGenericFilterMask generic_filter_mask) {
    return msg_type_filter_mask | recipient_node_filter_mask | generic_filter_mask;
}

esp_err_t can_init(int tx_pin, int rx_pin, int baudrate, const CanMsgFilter *filter) {
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

    // Filter configuration (accept all messages by default)
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();
    enum {
        // The id field in the CAN frame is 11 bits wide and, in the acceptance
        // code/mask, is located in the 11 most significant bits. By taking a
        // value with the 11 least significant bits set and shifting it 21 bits,
        // we get a value with the 11 most significant bits set and nothing else.
        BITS_TO_SHIFT_ID = 21U,
        // All bits set except for the most significant 11 bits.
        ID_BITS_MASK = ~(0x7FFU << BITS_TO_SHIFT_ID),
    };
    if (filter != NULL) {
        f_config.acceptance_code = filter->id << BITS_TO_SHIFT_ID;
        // The ignore_mask determines which ID bits should be set or unset.
        // However, we also want to make sure we ignore any other bits in the
        // frame (the RTR bit and data bits) so we only filter on the ID. To
        // do that, we have to make sure all other bits are set (ID_BITS_MASK).
        // We then also have to shift the provided ignore mask so it covers the
        // 11 most significant bits instead of the 11 least significant bits.
        f_config.acceptance_mask = ID_BITS_MASK | (filter->ignore_mask << BITS_TO_SHIFT_ID);
        f_config.single_filter = true;
    }

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
    twai_message_t message = {
        .identifier = id,
        .data_length_code = len,
        .flags = 0,  // Standard frame, data frame
    };
    
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
        
        // Since we don't support CAN FD, we know the maximum size of the
        // message data is 8 bytes, meaning we can do a direct copy by
        // treating the data buffer as an 8-byte integer. 
        // NOTE: We assume that the user has provided a buffer that's at least
        // 8 bytes.
        *(uint64_t*)data = *(uint64_t*)message.data;
    }
    
    return ret;
}

void can_deinit(void) {
    twai_stop();
    twai_driver_uninstall();
    ESP_LOGI(TAG, "CAN deinitialized");
}
