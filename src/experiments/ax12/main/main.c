#include "AX_servo.h"

#define SERVO_ID 1

void app_main(void)
{
    AX_conf_t ax = {
        .uart     = UART_NUM_1,
        .tx_pin   = GPIO_NUM_7,
        .rx_pin   = GPIO_NUM_6,
        .rts_pin  = UART_PIN_NO_CHANGE,
        .baudrate = 1000000   // or whatever Wizard says
    };

    AX_servo_init(ax);

    vTaskDelay(pdMS_TO_TICKS(500)); // let servo boot

    uint8_t led_on[]  = {0xFF, 0xFF, SERVO_ID, 0x04, 0x03, 0x19, 0x01, 0x00};
    uint8_t led_off[] = {0xFF, 0xFF, SERVO_ID, 0x04, 0x03, 0x19, 0x00, 0x00};

    // compute checksums:
    uint8_t sum = SERVO_ID + 0x04 + 0x03 + 0x19 + 0x01;
    led_on[7]  = (~sum) & 0xFF;

    sum = SERVO_ID + 0x04 + 0x03 + 0x19 + 0x00;
    led_off[7] = (~sum) & 0xFF;

    while (1) {
        uart_write_bytes(ax.uart, (const char *)led_on,  sizeof(led_on));
        vTaskDelay(pdMS_TO_TICKS(500));

        uart_write_bytes(ax.uart, (const char *)led_off, sizeof(led_off));
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

