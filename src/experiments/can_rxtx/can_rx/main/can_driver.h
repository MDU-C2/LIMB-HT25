#pragma once
#include "esp_err.h"
#include "driver/twai.h"


// Initialize the CAN bus
esp_err_t can_init(int tx_pin, int rx_pin, int baudrate);

// Send a CAN message
esp_err_t can_send(uint32_t id, const uint8_t *data, uint8_t len);

// Try to receive a message (blocking for timeout_ms)
esp_err_t can_receive(uint32_t *id, uint8_t *data, uint8_t *len, int timeout_ms);

// Stop and deinitialize the node
void can_deinit(void);