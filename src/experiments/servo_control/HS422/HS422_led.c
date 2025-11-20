#include "HS422_led.h"

static const char *TAG = "HS422_LEDC";

// Servo configurations - customize each servo individually
static servo_config_t servos[NUM_SERVOS] = {
    // Thumb servo
    {
        .gpio_pin = THUMB_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_0,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .direction = SERVO_DIR_REVERSE,
        .name = "Thumb"
    },
    // Index finger
    {
        .gpio_pin = INDEX_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_1,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .direction = SERVO_DIR_REVERSE,
        .name = "Index"
    },
    // Middle finger
    {
        .gpio_pin = MID_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_2,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .direction = SERVO_DIR_REVERSE,
        .name = "Middle"
    },
    // Ring finger
    {
        .gpio_pin = RING_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_3,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .direction = SERVO_DIR_NORMAL,
        .name = "Ring"
    },
    // Pinky finger
    {
        .gpio_pin = PINKY_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_4,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .direction = SERVO_DIR_NORMAL,
        .name = "Pinky"
    }
};

// Convert microseconds to duty cycle
uint32_t us_to_duty(uint32_t us)
{
    if (us < SERVO_MIN_US) us = SERVO_MIN_US;
    if (us > SERVO_MAX_US) us = SERVO_MAX_US;
    return (uint32_t)((uint64_t)SERVO_MAX_DUTY * us / SERVO_PERIOD_US);
}

// Initialize all servos
esp_err_t servo_led_init(void)
{
    ESP_LOGI(TAG, "Initializing LEDC for %d servos", NUM_SERVOS);
    
    // Configure LEDC timer (shared by all servos)
    ledc_timer_config_t ledc_timer = {
        .speed_mode       = LEDC_LOW_SPEED_MODE,
        .duty_resolution  = SERVO_RES_BITS,
        .timer_num        = LEDC_TIMER_0,
        .freq_hz          = SERVO_FREQ_HZ,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&ledc_timer));
    
    ESP_LOGI(TAG, "Timer configured");
    // Configure each servo channel individually
    for (int i = 0; i < NUM_SERVOS; i++) {
        ESP_LOGI(TAG, "Configuring %s on GPIO%d, Channel %d", 
                 servos[i].name, servos[i].gpio_pin, servos[i].ledc_channel);
        
        ledc_channel_config_t channel_config1 = {
            .gpio_num       = servos[i].gpio_pin,
            .speed_mode     = LEDC_LOW_SPEED_MODE,
            .channel        = servos[i].ledc_channel,
            .intr_type      = LEDC_INTR_DISABLE,
            .timer_sel      = LEDC_TIMER_0,
            .duty           = 0,
            .hpoint         = 0
        };
        
        ESP_ERROR_CHECK(ledc_channel_config(&channel_config1));
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    ESP_LOGI(TAG, "All channels configured, setting initial positions");
    
    // Set initial positions after all channels are configured
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_write_deg_channel(i, 90);  // Start at center position
        vTaskDelay(pdMS_TO_TICKS(50));   // Small delay between servo movements
    }

    ESP_LOGI(TAG, "All servos initialized at neutral position");
    return ESP_OK;
}

// Write angle to specific servo channel
void servo_write_deg_channel(int channel, int deg)
{
    if (channel < 0 || channel >= NUM_SERVOS) return;
    
    servo_config_t *servo = &servos[channel];
    // Clamp angle
    if (deg < servo->min_angle) deg = servo->min_angle;
    if (deg > servo->max_angle) deg = servo->max_angle;
    
    
    // Convert angle to pulse width
    uint32_t us = servo->min_pulse_us + 
                  ((deg - servo->min_angle) * (servo->max_pulse_us - servo->min_pulse_us)) / 
                  (servo->max_angle - servo->min_angle);
    
    // Set duty cycle
    uint32_t duty = us_to_duty(us);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, servo->ledc_channel, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, servo->ledc_channel);
    
    ESP_LOGI(TAG, "%s -> %d° (%lu us)", servo->name, deg, us);
}

// Write same angle to all servos
void servo_write_all_deg(int deg)
{
    ESP_LOGI(TAG, "Setting all servos to %d°", deg);
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_write_deg_channel(i, deg);
    }
}

// Close all fingers (180 degrees)
void close_all_fingers(void)
{
    ESP_LOGI(TAG, "Closing all fingers");
    for (int i = 0; i < NUM_SERVOS; i++) {
        if(servos[i].direction == SERVO_DIR_NORMAL) {
            servo_write_deg_channel(i, servos[i].max_angle);
        } else {
            servo_write_deg_channel(i, servos[i].min_angle);
        }
    }
}


// Open all fingers (0 degrees)
void open_all_fingers(void)
{
    ESP_LOGI(TAG, "Opening all fingers");
    for (int i = 0; i < NUM_SERVOS; i++) {
        if(servos[i].direction == SERVO_DIR_NORMAL) {
            servo_write_deg_channel(i, servos[i].min_angle);
        } else {
            servo_write_deg_channel(i, servos[i].max_angle);
        }
    }
}