#include "driver/ledc.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// =================================
#define BUTTON_Next_GPIO  1    // GPIO for button to move to next position (active low)
#define BUTTON_Prev_GPIO  0    // GPIO for button to move to previous position (
#define SERVO_GPIO        5           // PWM pin to MG90S signal (orange)
#define SERVO_FREQ_HZ     50          // 50 Hz = 20 ms period
#define SERVO_RES_BITS    13          // resolution; 13 bits = 8191 ticks

#define SERVO_MIN_US      500        // 500 
#define SERVO_MAX_US      2500        // 2500 
#define SERVO_MIN_DEG     0
#define SERVO_MAX_DEG     180
// =================================

#define SERVO_PERIOD_US   (1000000UL / SERVO_FREQ_HZ)
#define SERVO_MAX_DUTY    ((1U << SERVO_RES_BITS) - 1)

uint32_t us_to_duty(uint32_t us);
void servo_init(void);
void servo_write_us(uint32_t pulse_us);
void servo_write_deg(int deg);
int servo_button_control(int current_angle, bool mode);
