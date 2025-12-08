/**
 * @file AX_servo.h
 * @author JanG175
 * @brief DYNAMIXEL AX-12A SERIAL PROTOCOL LIBRARY
 * 
 * @copyright Apache 2.0
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_log.h"

// important AX-12 constants
// EEPROM AREA
#define AX_MODEL_NUMBER_L           0
#define AX_MODEL_NUMBER_H           1
#define AX_VERSION                  2
#define AX_ID                       3
#define AX_BAUD_RATE                4
#define AX_RETURN_DELAY_TIME        5
#define AX_CW_ANGLE_LIMIT_L         6
#define AX_CW_ANGLE_LIMIT_H         7
#define AX_CCW_ANGLE_LIMIT_L        8
#define AX_CCW_ANGLE_LIMIT_H        9
#define AX_SYSTEM_DATA2             10
#define AX_LIMIT_TEMPERATURE        11
#define AX_DOWN_LIMIT_VOLTAGE       12
#define AX_UP_LIMIT_VOLTAGE         13
#define AX_MAX_TORQUE_L             14
#define AX_MAX_TORQUE_H             15
#define AX_RETURN_LEVEL             16
#define AX_ALARM_LED                17
#define AX_ALARM_SHUTDOWN           18
#define AX_OPERATING_MODE           19
#define AX_DOWN_CALIBRATION_L       20
#define AX_DOWN_CALIBRATION_H       21
#define AX_UP_CALIBRATION_L         22
#define AX_UP_CALIBRATION_H         23

// RAM AREA
#define AX_TORQUE_ENABLE            24
#define AX_LED                      25
#define AX_CW_COMPLIANCE_MARGIN     26
#define AX_CCW_COMPLIANCE_MARGIN    27
#define AX_CW_COMPLIANCE_SLOPE      28
#define AX_CCW_COMPLIANCE_SLOPE     29
#define AX_GOAL_POSITION_L          30
#define AX_GOAL_POSITION_H          31
#define AX_GOAL_SPEED_L             32
#define AX_GOAL_SPEED_H             33
#define AX_TORQUE_LIMIT_L           34
#define AX_TORQUE_LIMIT_H           35
#define AX_PRESENT_POSITION_L       36
#define AX_PRESENT_POSITION_H       37
#define AX_PRESENT_SPEED_L          38
#define AX_PRESENT_SPEED_H          39
#define AX_PRESENT_LOAD_L           40
#define AX_PRESENT_LOAD_H           41
#define AX_PRESENT_VOLTAGE          42
#define AX_PRESENT_TEMPERATURE      43
#define AX_REGISTERED_INSTRUCTION   44
#define AX_PAUSE_TIME               45
#define AX_MOVING                   46
#define AX_LOCK                     47
#define AX_PUNCH_L                  48
#define AX_PUNCH_H                  49

// Status Return Levels
#define AX_RETURN_NONE              0
#define AX_RETURN_READ              1
#define AX_RETURN_ALL               2

// Instruction Set
#define AX_PING                     1
#define AX_READ_DATA                2
#define AX_WRITE_DATA               3
#define AX_REG_WRITE                4
#define AX_ACTION                   5
#define AX_RESET                    6
#define AX_SYNC_WRITE               131

// Specials
#define OFF                         0
#define ON                          1
#define LEFT                        0
#define RIGHT                       1
#define AX_BYTE_READ                1
#define AX_BYTE_READ_POS            2
#define AX_RESET_LENGTH             2
#define AX_ACTION_LENGTH            2
#define AX_ID_LENGTH                4
#define AX_LR_LENGTH                4
#define AX_SRL_LENGTH               4
#define AX_RDT_LENGTH               4
#define AX_LEDALARM_LENGTH          4
#define AX_SALARM_LENGTH            4
#define AX_TL_LENGTH                4
#define AX_VL_LENGTH                6
#define AX_CM_LENGTH                6
#define AX_CS_LENGTH                6
#define AX_CCW_CW_LENGTH            8
#define AX_BD_LENGTH                4
#define AX_TEM_LENGTH               4
#define AX_MOVING_LENGTH            4
#define AX_RWS_LENGTH               4
#define AX_VOLT_LENGTH              4
#define AX_LED_LENGTH               4
#define AX_TORQUE_LENGTH            4
#define AX_POS_LENGTH               4
#define AX_GOAL_LENGTH              5
#define AX_MT_LENGTH                5
#define AX_PUNCH_LENGTH             5
#define AX_SPEED_LENGTH             5
#define AX_GOAL_SP_LENGTH           7
#define AX_ACTION_CHECKSUM          250
#define BROADCAST_ID                254
#define AX_START                    255
#define AX_CCW_AL_L                 255 
#define AX_CCW_AL_H                 3
#define TIME_OUT                    10
#define TX_MODE                     1
#define RX_MODE                     0
#define LOCK                        1

// Baud Rates
#define AX_BAUD_1000000             1
#define AX_BAUD_500000              3
#define AX_BAUD_400000              4
#define AX_BAUD_250000              7
#define AX_BAUD_200000              9
#define AX_BAUD_115200              16
#define AX_BAUD_57600               34
#define AX_BAUD_19200               103
#define AX_BAUD_9600                207

// UART timeout constants
#define AX_UART_MAX_REPEAT          50
#define AX_UART_TIMEOUT_MS          (20 / portTICK_PERIOD_MS)

typedef struct AX_conf_t
{
    uart_port_t uart;
    gpio_num_t tx_pin;
    gpio_num_t rx_pin;
    gpio_num_t rts_pin;
    uint32_t baudrate;
} AX_conf_t;


void AX_servo_init(AX_conf_t AX_conf);

void AX_servo_deinit(AX_conf_t AX_conf);

void AX_servo_ping(AX_conf_t AX_conf, uint8_t ID);

void AX_servo_set_pos(AX_conf_t AX_conf, uint8_t id, uint16_t pos);

void AX_servo_set_pos_w_spd(AX_conf_t AX_conf, uint8_t ID, uint16_t pos, uint16_t speed);

void AX_servo_set_endless(AX_conf_t AX_conf, uint8_t ID, bool status);

void AX_conf_turn(AX_conf_t AX_conf, uint8_t ID, uint8_t side, uint16_t speed);

void AX_servo_set_max_torque(AX_conf_t AX_conf, uint8_t ID, uint16_t torque);

void AX_conf_torque_status(AX_conf_t AX_conf, uint8_t ID, uint8_t status);

uint16_t AX_servo_get_pos(AX_conf_t AX_conf, uint8_t ID);

void AX_servo_set_angle_limit(AX_conf_t AX_conf, uint8_t ID, uint16_t CWLimit, uint16_t CCWLimit);

void AX_servo_set_led(AX_conf_t AX_conf, uint8_t ID, uint8_t status);

uint16_t AX_servo_read_register(AX_conf_t AX_conf, uint8_t ID, uint8_t reg, uint8_t reg_len);

void AX_servo_write_register(AX_conf_t AX_conf, uint8_t ID, uint8_t reg, uint8_t reg_val);

void AX_servo_reset(AX_conf_t AX_conf, uint8_t ID);

void AX_servo_set_ID(AX_conf_t AX_conf, uint8_t ID, uint8_t newID);

void AX_servo_set_BD(AX_conf_t AX_conf, uint8_t ID, uint8_t BAUD);

int16_t AX_servo_get_temperature(AX_conf_t AX_conf, uint8_t ID);

uint8_t AX_servo_is_moving(AX_conf_t AX_conf, uint8_t ID);

uint16_t AX_servo_get_load(AX_conf_t AX_conf, uint8_t ID);

void AX_servo_set_shutdown_alarm(AX_conf_t AX_conf, uint8_t ID, uint8_t salarm);

void AX_servo_set_C_margin(AX_conf_t AX_conf, uint8_t ID, uint8_t CWCMargin, uint8_t CCWCMargin);

void AX_servo_set_C_slope(AX_conf_t AX_conf, uint8_t ID, uint8_t CWCSlope, uint8_t CCWCSlope);

void AX_servo_set_punch(AX_conf_t AX_conf, uint8_t ID, uint16_t punch);
