#include "mg90s.h"

uint32_t us_to_duty(uint32_t us)
{
    if (us < SERVO_MIN_US) us = SERVO_MIN_US;
    if (us > SERVO_MAX_US) us = SERVO_MAX_US;
    // Duty is proportion of high time within full PWM period
    return (uint32_t)((uint64_t)SERVO_MAX_DUTY * us / SERVO_PERIOD_US);
}

void servo_init(void)
{
    ledc_timer_config(&(ledc_timer_config_t){
        .speed_mode       = LEDC_LOW_SPEED_MODE,
        .duty_resolution  = SERVO_RES_BITS,
        .timer_num        = LEDC_TIMER_0,
        .freq_hz          = SERVO_FREQ_HZ,
        .clk_cfg          = LEDC_AUTO_CLK,
    });

    ledc_channel_config(&(ledc_channel_config_t){
        .gpio_num       = SERVO_GPIO,
        .speed_mode     = LEDC_LOW_SPEED_MODE,
        .channel        = LEDC_CHANNEL_0,
        .timer_sel      = LEDC_TIMER_0,
        .duty           = 0,
        .hpoint         = 0,
        .intr_type      = LEDC_INTR_DISABLE,
    });

    gpio_set_direction(BUTTON_Next_GPIO, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON_Next_GPIO, GPIO_PULLUP_ONLY);
    gpio_set_direction(BUTTON_Prev_GPIO, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON_Prev_GPIO, GPIO_PULLUP_ONLY);
}

void servo_write_us(uint32_t pulse_us)
{
    uint32_t duty = us_to_duty(pulse_us);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
}

void servo_write_deg(int deg)
{
    if (deg < SERVO_MIN_DEG) deg = SERVO_MIN_DEG;
    if (deg > SERVO_MAX_DEG) deg = SERVO_MAX_DEG;
    const float span_us = (float)(SERVO_MAX_US - SERVO_MIN_US);
    uint32_t us = (uint32_t)(SERVO_MIN_US + span_us * (deg - SERVO_MIN_DEG) /
                             (SERVO_MAX_DEG - SERVO_MIN_DEG));
    servo_write_us(us);
}

int servo_button_control(int current_angle, bool mode)
{

    servo_write_deg(current_angle);

    if (mode == 0) {
        if (gpio_get_level(BUTTON_Next_GPIO) == 0) { // Button pressed (active low)
            current_angle += 45; // Move 45 degrees forward
            if (current_angle > SERVO_MAX_DEG) {
                current_angle = SERVO_MAX_DEG;
            }
            servo_write_deg(current_angle);
            vTaskDelay(pdMS_TO_TICKS(300)); // Debounce delay
        }
        if (gpio_get_level(BUTTON_Prev_GPIO) == 0) { // Button pressed (active low)
            current_angle -= 45; // Move 45 degrees backward
            if (current_angle < SERVO_MIN_DEG) {
                current_angle = SERVO_MIN_DEG;
            }
            servo_write_deg(current_angle);
            vTaskDelay(pdMS_TO_TICKS(300)); // Debounce delay
        }
        printf("Current Angle: %d\n", current_angle);
        vTaskDelay(pdMS_TO_TICKS(50)); // Polling delay
    }
    else {
        // Automatic mode can be implemented here
        if (gpio_get_level(BUTTON_Next_GPIO) == 0){
            current_angle = SERVO_MAX_DEG;
            printf("Gripping\n");
            servo_write_deg(current_angle);
            vTaskDelay(pdMS_TO_TICKS(300)); // Debounce delay
        } 
        if (gpio_get_level(BUTTON_Prev_GPIO) == 0) 
        {
            current_angle = SERVO_MIN_DEG;
            printf("Releasing\n");
            servo_write_deg(current_angle);
            vTaskDelay(pdMS_TO_TICKS(300)); // Debounce delay
        }
    }
    return current_angle;

}