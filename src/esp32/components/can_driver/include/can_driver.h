#pragma once
#include "esp_err.h"

// Allows for filtering which messages to accept from the CAN bus.
// It follows the same format as in the [ESP-IDF documentation](
// https://docs.espressif.com/projects/esp-idf/en/v5.4.3/esp32c3/api-reference/peripherals/twai.html#acceptance-filter
// ), but it only filters on the id, not any other parts of the frame.
// 
// For example, if the ID is 0b000'0000'1000 and the mask is 0b000'0000'0111,
// then all IDs that match 0b000'0000'1xxx will be allowed and all other ids
// will be filtered.
typedef struct {
    uint32_t id;
    // Set bits are ignored. Unset MUST match.
    uint32_t ignore_mask;
} CanMsgFilter;

// Initialize the CAN bus
esp_err_t can_init(int tx_pin, int rx_pin, int baudrate, const CanMsgFilter* filter);

// Send a CAN message
esp_err_t can_send(uint32_t id, const uint8_t *data, uint8_t len);

// Try to receive a message (blocking for timeout_ms)
esp_err_t can_receive(uint32_t *id, uint8_t *data, uint8_t *len, int timeout_ms);

// Stop and deinitialize the node
void can_deinit(void);
