/**
 * @file AX_servo.c
 * @author JanG175
 * @brief DYNAMIXEL AX-12A SERIAL PROTOCOL LIBRARY
 *
 * @copyright Apache 2.0
 */

#include "AX_servo.h"

#include <stdio.h>

static const bool abort_on = true;  // enable abort after timeout

static const char* TAG = "AX_servo";

/**
 * @brief calculate checksum of packet
 *
 * @param packet pointer to packet
 * @param len length of packet
 * @return checksum of packet
 */
static uint8_t AX_servo_calc_checksum(uint8_t* packet, uint32_t len) {
  uint8_t checksum = 0;

  for (uint32_t i = 2; i < len - 1; i++) {
    checksum += packet[i];
  }

  checksum = (~checksum) & 0xFF;

  return checksum;
}

/**
 * @brief check if response is valid
 *
 * @param ID ID of servo
 * @param response pointer to response
 * @param len length of response
 * @return true - response is valid, false - response is invalid
 */
static bool AX_servo_check_response(uint8_t ID, uint8_t* response,
                                    uint32_t len) {
  uint8_t checksum = AX_servo_calc_checksum(response, len);

  if (response[0] == AX_START && response[1] == AX_START && response[2] == ID &&
      response[len - 1] == checksum) {
    if (response[4] == 0)
      return true;
    else {
      ESP_LOGE(TAG, "Servo %u error code: %u", ID, response[4]);
      return false;
    }
  } else {
    ESP_LOGE(TAG, "Servo %u, response is invalid", ID);
    return false;
  }
}

/**
 * @brief send packet to servo
 *
 * @param AX struct with UART parameters
 * @param packet pointer to packet
 * @param len length of packet
 */
static void AX_servo_send_packet(AX_conf_t AX_conf, uint8_t* packet,
                                 uint32_t len) {
  // send uint8_t array to servo byte by byte
  uart_write_bytes(AX_conf.uart, (const uint8_t*)packet, len);
  ESP_ERROR_CHECK(uart_wait_tx_done(AX_conf.uart, AX_UART_TIMEOUT_MS));
}

/**
 * @brief receive response from servo
 *
 * @param AX struct with UART parameters
 * @param response pointer to response packet
 * @param len length of response packet
 */
static void AX_servo_receive_response(AX_conf_t AX_conf, uint8_t* response,
                                      uint32_t len) {
  uint32_t buf = 0;
  uint8_t data[len];

  // receive uint8_t array from servo
  buf = uart_read_bytes(AX_conf.uart, data, len, AX_UART_TIMEOUT_MS);
  uart_flush(AX_conf.uart);

  if (buf == len) {
    for (uint32_t i = 0; i < buf; i++) {
      response[i] = data[i];
    }
  } else {
    ESP_LOGE(TAG, "UART read error");

    for (uint32_t i = 0; i < len; i++) response[i] = 0;
  }
}

/**
 * @brief init ESP32 UART connection
 *
 * @param AX struct with UART parameters
 */
void AX_servo_init(AX_conf_t AX_conf) {
  uart_config_t uart_config = {.baud_rate = AX_conf.baudrate,
                               .data_bits = UART_DATA_8_BITS,
                               .parity = UART_PARITY_DISABLE,
                               .stop_bits = UART_STOP_BITS_1,
                               .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
                               .rx_flow_ctrl_thresh = 0,
                               .source_clk = UART_SCLK_DEFAULT};

  // If UART already has a driver, delete it first
  if (uart_is_driver_installed(AX_conf.uart) == true) {
    ESP_ERROR_CHECK(uart_driver_delete(AX_conf.uart));
  }

  // Install UART driver with both TX and RX buffers
  ESP_ERROR_CHECK(uart_driver_install(AX_conf.uart,
                                      2048,  // RX buffer size
                                      2048,  // TX buffer size
                                      0, NULL, 0));

  // Apply UART settings
  ESP_ERROR_CHECK(uart_param_config(AX_conf.uart, &uart_config));

  // Set TX and RX pins, no RTS/CTS
  ESP_ERROR_CHECK(uart_set_pin(AX_conf.uart, AX_conf.tx_pin, AX_conf.rx_pin,
                               UART_PIN_NO_CHANGE,    // RTS not used
                               UART_PIN_NO_CHANGE));  // CTS not used

  // Normal UART mode (not RS485)
  ESP_ERROR_CHECK(uart_set_mode(AX_conf.uart, UART_MODE_UART));

  // IMPORTANT: invert TX because of NPN open-collector
  // Your transistor inverts the signal once, so we invert it in hardware
  ESP_ERROR_CHECK(uart_set_line_inverse(AX_conf.uart, UART_SIGNAL_TXD_INV));
}

/**
 * @brief deinit ESP32 UART connection
 *
 * @param AX struct with UART parameters
 */
void AX_servo_deinit(AX_conf_t AX_conf) {
  if (uart_is_driver_installed(AX_conf.uart) == true)
    ESP_ERROR_CHECK(uart_driver_delete(AX_conf.uart));
}

/**
 * @brief ping servo
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 */
void AX_servo_ping(AX_conf_t AX_conf, uint8_t ID) {
  uint32_t len = 6;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_READ_DATA;
  packet[4] = AX_PING;
  packet[5] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo goal position
 *
 * @param AX struct with UART parameters
 * @param id servo ID
 * @param pos goal position
 */
void AX_servo_set_pos(AX_conf_t AX_conf, uint8_t ID, uint16_t pos) {
  uint8_t pos_H, pos_L;
  pos_H = pos >> 8;
  pos_L = pos & 0xFF;

  uint32_t len = 9;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_GOAL_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_GOAL_POSITION_L;
  packet[6] = pos_L;
  packet[7] = pos_H;
  packet[8] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo goal position with desired speed
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param pos goal position
 * @param speed speed
 */
void AX_servo_set_pos_w_spd(AX_conf_t AX_conf, uint8_t ID, uint16_t pos,
                            uint16_t speed) {
  uint8_t pos_H, pos_L, spd_H, spd_L;
  pos_H = pos >> 8;
  pos_L = pos & 0xFF;
  spd_H = speed >> 8;
  spd_L = speed & 0xFF;

  uint32_t len = 11;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_GOAL_SP_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_GOAL_POSITION_L;
  packet[6] = pos_L;
  packet[7] = pos_H;
  packet[8] = spd_L;
  packet[9] = spd_H;
  packet[10] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief enable servo endless rotation
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param status true - enable, false - disable
 */
void AX_servo_set_endless(AX_conf_t AX_conf, uint8_t ID, bool status) {
  if (status == true) {
    uint32_t len = 9;
    uint8_t packet[len];

    packet[0] = AX_START;
    packet[1] = AX_START;
    packet[2] = ID;
    packet[3] = AX_GOAL_LENGTH;
    packet[4] = AX_WRITE_DATA;
    packet[5] = AX_CCW_ANGLE_LIMIT_L;
    packet[6] = OFF;
    packet[7] = OFF;
    packet[8] = AX_servo_calc_checksum(packet, len);

    AX_servo_send_packet(AX_conf, packet, len);
  } else {
    AX_conf_turn(AX_conf, ID, 0, 0);

    uint32_t len = 9;
    uint8_t packet[len];

    packet[0] = AX_START;
    packet[1] = AX_START;
    packet[2] = ID;
    packet[3] = AX_GOAL_LENGTH;
    packet[4] = AX_WRITE_DATA;
    packet[5] = AX_CCW_ANGLE_LIMIT_L;
    packet[6] = AX_CCW_AL_L;
    packet[7] = AX_CCW_AL_H;
    packet[8] = AX_servo_calc_checksum(packet, len);

    AX_servo_send_packet(AX_conf, packet, len);
  }
}

/**
 * @brief turn servo endlessly
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param side LEFT (CCW) or RIGHT (CW)
 * @param speed speed
 */
void AX_conf_turn(AX_conf_t AX_conf, uint8_t ID, uint8_t side, uint16_t speed) {
  if (side == LEFT) {
    uint8_t spd_H, spd_L;
    spd_H = speed >> 8;
    spd_L = speed & 0xFF;

    uint32_t len = 9;
    uint8_t packet[len];

    packet[0] = AX_START;
    packet[1] = AX_START;
    packet[2] = ID;
    packet[3] = AX_SPEED_LENGTH;
    packet[4] = AX_WRITE_DATA;
    packet[5] = AX_GOAL_SPEED_L;
    packet[6] = spd_L;
    packet[7] = spd_H;
    packet[8] = AX_servo_calc_checksum(packet, len);

    AX_servo_send_packet(AX_conf, packet, len);
  } else if (side == RIGHT) {
    uint8_t spd_H, spd_L;
    spd_H = (speed >> 8) + 4;
    spd_L = speed & 0xFF;

    uint32_t len = 9;
    uint8_t packet[len];

    packet[0] = AX_START;
    packet[1] = AX_START;
    packet[2] = ID;
    packet[3] = AX_SPEED_LENGTH;
    packet[4] = AX_WRITE_DATA;
    packet[5] = AX_GOAL_SPEED_L;
    packet[6] = spd_L;
    packet[7] = spd_H;
    packet[8] = AX_servo_calc_checksum(packet, len);

    AX_servo_send_packet(AX_conf, packet, len);
  }
}

/**
 * @brief set servo max torque
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param torque max torque
 */
void AX_servo_set_max_torque(AX_conf_t AX_conf, uint8_t ID, uint16_t torque) {
  uint8_t max_torque_frame_H = torque >> 8;
  uint8_t max_torque_frame_L = torque & 0xFF;

  uint32_t len = 9;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_MT_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_MAX_TORQUE_L;
  packet[6] = max_torque_frame_L;
  packet[7] = max_torque_frame_H;
  packet[8] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo torque status
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param status true - enable, false - disable
 */
void AX_conf_torque_status(AX_conf_t AX_conf, uint8_t ID, uint8_t status) {
  uint32_t len = 8;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_TORQUE_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_TORQUE_ENABLE;
  packet[6] = status;
  packet[7] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief get servo current position
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @return current position
 */
uint16_t AX_servo_get_pos(AX_conf_t AX_conf, uint8_t ID) {
  uint8_t pos_frame_L = 0;
  uint8_t pos_frame_H = 0;

  uint32_t len_w = 8;
  uint8_t packet[len_w];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_POS_LENGTH;
  packet[4] = AX_READ_DATA;
  packet[5] = AX_PRESENT_POSITION_L;
  packet[6] = AX_BYTE_READ_POS;
  packet[7] = AX_servo_calc_checksum(packet, len_w);

  uint32_t len_r = 8;
  uint8_t response[len_r];
  uint32_t cnt = 0;

  do {
    AX_servo_send_packet(AX_conf, packet, len_w);
    AX_servo_receive_response(AX_conf, response, len_r);
    cnt++;
  } while ((AX_servo_check_response(ID, response, len_r) == false) &&
           (cnt < AX_UART_MAX_REPEAT));

  if (cnt < AX_UART_MAX_REPEAT) {
    pos_frame_L = response[5];
    pos_frame_H = response[6];
  } else {
    if (abort_on == true) {
      ESP_LOGE(TAG, "Servo %u not responding - aborting...", ID);
      AX_servo_deinit(AX_conf);
      abort();
    }
  }

  return (pos_frame_H << 8) + pos_frame_L;
}

/**
 * @brief set servo angle limits
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param CWLimit clockwise limit
 * @param CCWLimit counter clockwise limit
 */
void AX_servo_set_angle_limit(AX_conf_t AX_conf, uint8_t ID, uint16_t CWLimit,
                              uint16_t CCWLimit) {
  uint8_t CW_H = CWLimit >> 8;
  uint8_t CW_L = CWLimit & 0xFF;
  uint8_t CCW_H = CCWLimit >> 8;
  uint8_t CCW_L = CCWLimit & 0xFF;

  uint32_t len = 12;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_CCW_CW_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_CW_ANGLE_LIMIT_L;
  packet[6] = CW_L;
  packet[7] = CW_H;
  packet[8] = AX_CCW_ANGLE_LIMIT_L;
  packet[9] = CCW_L;
  packet[10] = CCW_H;
  packet[11] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo LED level
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param status LED level
 */
void AX_servo_set_led(AX_conf_t AX_conf, uint8_t ID, uint8_t status) {
  uint32_t len = 8;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_LED_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_LED;
  packet[6] = status;
  packet[7] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief read value from servo register
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param reg register number
 * @param reg_len register length
 * @return register value
 */
uint16_t AX_servo_read_register(AX_conf_t AX_conf, uint8_t ID, uint8_t reg,
                                uint8_t reg_len) {
  uint8_t reg_frame = 0;
  uint32_t len_w = 8;
  uint8_t packet[len_w];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_REG_WRITE;
  packet[4] = AX_READ_DATA;
  packet[5] = reg;
  packet[6] = reg_len;
  packet[7] = AX_servo_calc_checksum(packet, len_w);

  uint32_t len_r = 7;
  uint8_t response[len_r];
  uint32_t cnt = 0;

  do {
    AX_servo_send_packet(AX_conf, packet, len_w);
    AX_servo_receive_response(AX_conf, response, len_r);
    cnt++;
  } while ((AX_servo_check_response(ID, response, len_r) == false) &&
           (cnt < AX_UART_MAX_REPEAT));

  if (cnt < AX_UART_MAX_REPEAT) {
    switch (reg_len) {
      case 1:
        reg_frame = response[5];
        break;
      case 2:
        reg_frame = response[5];
        reg_frame += response[6] << 8;
        break;
    }
  } else {
    if (abort_on == true) {
      ESP_LOGE(TAG, "Servo %u not responding - aborting...", ID);
      AX_servo_deinit(AX_conf);
      abort();
    }
  }

  return reg_frame;
}

/**
 * @brief write value to servo register
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param reg register number
 * @param reg_val register value
 */
void AX_servo_write_register(AX_conf_t AX_conf, uint8_t ID, uint8_t reg,
                             uint8_t reg_val) {
  uint32_t len = 8;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_REG_WRITE;
  packet[4] = AX_WRITE_DATA;
  packet[5] = reg;
  packet[6] = reg_val;
  packet[7] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief servo factory reset
 * (reset sets id to 1 and baudrate to 1000000 and sets back original values of
 * them)
 * [!!! might not work properly - esp32 does not like 10M baud !!!]
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 */
void AX_servo_reset(AX_conf_t AX_conf, uint8_t ID) {
  uint32_t len = 6;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_RESET_LENGTH;
  packet[4] = AX_RESET;
  packet[5] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);

  // reset sets id to 1 and baudrate to 1000000

  uint32_t baud = AX_conf.baudrate;
  AX_servo_deinit(AX_conf);
  AX_conf.baudrate = 1000000;

  vTaskDelay(10 / portTICK_PERIOD_MS);

  AX_servo_init(AX_conf);
  vTaskDelay(10 / portTICK_PERIOD_MS);

  AX_servo_set_ID(AX_conf, 1, ID);
  vTaskDelay(10 / portTICK_PERIOD_MS);

  AX_servo_set_BD(AX_conf, ID, baud);
}

/**
 * @brief set servo ID
 *
 * @param AX struct with UART parameters
 * @param ID current servo ID
 * @param newID new servo ID
 */
void AX_servo_set_ID(AX_conf_t AX_conf, uint8_t ID, uint8_t newID) {
  uint32_t len = 8;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_ID_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_ID;
  packet[6] = newID;
  packet[7] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo baudrate
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param baud new baudrate
 */
void AX_servo_set_BD(AX_conf_t AX_conf, uint8_t ID, uint8_t BAUD) {
  uint32_t len = 8;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_BD_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_BAUD_RATE;
  packet[6] = BAUD;
  packet[7] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief read servo temperature
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @return temperature
 */
int16_t AX_servo_get_temperature(AX_conf_t AX_conf, uint8_t ID) {
  uint16_t temp_frame = 0;
  uint32_t len_w = 8;
  uint8_t packet[len_w];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_TEM_LENGTH;
  packet[4] = AX_READ_DATA;
  packet[5] = AX_PRESENT_TEMPERATURE;
  packet[6] = AX_BYTE_READ;
  packet[7] = AX_servo_calc_checksum(packet, len_w);

  uint32_t len_r = 7;
  uint8_t response[len_r];
  uint32_t cnt = 0;

  do {
    AX_servo_send_packet(AX_conf, packet, len_w);
    AX_servo_receive_response(AX_conf, response, len_r);
    cnt++;
  } while ((AX_servo_check_response(ID, response, len_r) == false) &&
           (cnt < AX_UART_MAX_REPEAT));

  if (cnt < AX_UART_MAX_REPEAT) {
    temp_frame = response[5];
  } else {
    if (abort_on == true) {
      ESP_LOGE(TAG, "Servo %u not responding - aborting...", ID);
      AX_servo_deinit(AX_conf);
      abort();
    }
  }

  return temp_frame;
}

/**
 * @brief read if servo is moving
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @return 1 - moving, 0 - not moving
 */
uint8_t AX_servo_is_moving(AX_conf_t AX_conf, uint8_t ID) {
  uint8_t mov_byte = 0;
  uint32_t len_w = 8;
  uint8_t packet[len_w];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_MOVING_LENGTH;
  packet[4] = AX_READ_DATA;
  packet[5] = AX_MOVING;
  packet[6] = AX_BYTE_READ;
  packet[7] = AX_servo_calc_checksum(packet, len_w);

  uint32_t len_r = 7;
  uint8_t response[len_r];
  uint32_t cnt = 0;

  do {
    AX_servo_send_packet(AX_conf, packet, len_w);
    AX_servo_receive_response(AX_conf, response, len_r);
    cnt++;
  } while ((AX_servo_check_response(ID, response, len_r) == false) &&
           (cnt < AX_UART_MAX_REPEAT));

  if (cnt < AX_UART_MAX_REPEAT) {
    mov_byte = response[5];
  } else {
    if (abort_on == true) {
      ESP_LOGE(TAG, "Servo %u not responding - aborting...", ID);
      AX_servo_deinit(AX_conf);
      abort();
    }
  }

  return mov_byte;
}

/**
 * @brief read servo current torque
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @return current torque
 */
uint16_t AX_servo_get_load(AX_conf_t AX_conf, uint8_t ID) {
  uint8_t load_byte_L = 0;
  uint8_t load_byte_H = 0;

  uint32_t len_w = 8;
  uint8_t packet[len_w];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_POS_LENGTH;
  packet[4] = AX_READ_DATA;
  packet[5] = AX_PRESENT_LOAD_L;
  packet[6] = AX_BYTE_READ_POS;
  packet[7] = AX_servo_calc_checksum(packet, len_w);

  uint32_t len_r = 8;
  uint8_t response[len_r];
  uint32_t cnt = 0;

  do {
    AX_servo_send_packet(AX_conf, packet, len_w);
    AX_servo_receive_response(AX_conf, response, len_r);
    cnt++;
  } while ((AX_servo_check_response(ID, response, len_r) == false) &&
           (cnt < AX_UART_MAX_REPEAT));

  if (cnt < AX_UART_MAX_REPEAT) {
    load_byte_L = response[5];
    load_byte_H = response[6];
  } else {
    if (abort_on == true) {
      ESP_LOGE(TAG, "Servo %u not responding - aborting...", ID);
      AX_servo_deinit(AX_conf);
      abort();
    }
  }

  return (load_byte_H << 8) + load_byte_L;
}

/**
 * @brief set servo shutdown alarm
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param salarm shutdown alarm number
 */
void AX_servo_set_shutdown_alarm(AX_conf_t AX_conf, uint8_t ID,
                                 uint8_t salarm) {
  uint32_t len = 8;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_SALARM_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_ALARM_SHUTDOWN;
  packet[6] = salarm;
  packet[7] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo compliance margin
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param CWCMargin clockwise compliance margin
 * @param CCWCMargin counter clockwise compliance margin
 */
void AX_servo_set_C_margin(AX_conf_t AX_conf, uint8_t ID, uint8_t CWCMargin,
                           uint8_t CCWCMargin) {
  uint32_t len = 10;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_CM_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_CW_COMPLIANCE_MARGIN;
  packet[6] = CWCMargin;
  packet[7] = AX_CCW_COMPLIANCE_MARGIN;
  packet[8] = CCWCMargin;
  packet[9] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo compliance slope
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param CWCSlope clockwise compliance slope
 * @param CCWCSlope counter clockwise compliance slope
 */
void AX_servo_set_C_slope(AX_conf_t AX_conf, uint8_t ID, uint8_t CWCSlope,
                          uint8_t CCWCSlope) {
  uint32_t len = 10;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_CS_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_CW_COMPLIANCE_SLOPE;
  packet[6] = CWCSlope;
  packet[7] = AX_CCW_COMPLIANCE_SLOPE;
  packet[8] = CCWCSlope;
  packet[9] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}

/**
 * @brief set servo punch
 *
 * @param AX struct with UART parameters
 * @param ID servo ID
 * @param punch punch
 */
void AX_servo_set_punch(AX_conf_t AX_conf, uint8_t ID, uint16_t punch) {
  uint8_t punch_H = punch >> 8;
  uint8_t punch_L = punch & 0xFF;

  uint8_t len = 9;
  uint8_t packet[len];

  packet[0] = AX_START;
  packet[1] = AX_START;
  packet[2] = ID;
  packet[3] = AX_PUNCH_LENGTH;
  packet[4] = AX_WRITE_DATA;
  packet[5] = AX_PUNCH_L;
  packet[6] = punch_L;
  packet[7] = punch_H;
  packet[8] = AX_servo_calc_checksum(packet, len);

  AX_servo_send_packet(AX_conf, packet, len);
}
