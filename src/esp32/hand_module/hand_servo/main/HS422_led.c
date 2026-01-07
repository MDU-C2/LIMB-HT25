#include "HS422_led.h"

static const char *TAG = "HS422_LEDC";

static int dedos[NUM_SERVOS] = {THUMB_SERVO_GPIO, INDEX_SERVO_GPIO, MID_SERVO_GPIO, RING_SERVO_GPIO, PINKY_SERVO_GPIO};

// Servo configurations - customize each servo individually
static servo_config_t servos[NUM_SERVOS] = {
    // Thumb servo
    {
        .gpio_pin = THUMB_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_0,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1100,
        .max_pulse_us = 1700,
        .direction = SERVO_DIR_REVERSE,
        .name = "Thumb"
    },
    // Index finger
    {
        .gpio_pin = INDEX_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_1,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1100,
        .max_pulse_us = 1700,
        .direction = SERVO_DIR_REVERSE,
        .name = "Index"
    },
    // Middle finger
    {
        .gpio_pin = MID_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_2,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1100,
        .max_pulse_us = 1700,
        .direction = SERVO_DIR_REVERSE,
        .name = "Middle"
    },
    // Ring finger
    {
        .gpio_pin = RING_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_3,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1100,
        .max_pulse_us = 1700,
        .direction = SERVO_DIR_REVERSE,
        .name = "Ring"
    },
    // Pinky finger
    {
        .gpio_pin = PINKY_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_4,
        .max_angle = 180,
        .min_angle = 0,
        .min_pulse_us = 1200,
        .max_pulse_us = 1700,
        .direction = SERVO_DIR_REVERSE,
        .name = "Pinky"
    },
    {
        .gpio_pin = TWIST_SERVO_GPIO,
        .ledc_channel = LEDC_CHANNEL_5,
        // FIXME: Real???
        .min_angle = 0,
        .max_angle = 45,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .direction = SERVO_DIR_NORMAL,
        .name = "Wrist"
    },
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

    // Set initial wrist rotation at center position.
    servo_write_deg_channel(WRIST_SERVO_CONFIG_INDEX, (servos[WRIST_SERVO_CONFIG_INDEX].max_angle - servos[WRIST_SERVO_CONFIG_INDEX].min_angle) / 2 + 2);

    // Set initial positions after all channels are configured
    for (int i = 0; i < NUM_FINGER_SERVOS; i++) {
        servo_write_deg_channel(i, 180);  // Start at center position
        vTaskDelay(pdMS_TO_TICKS(50));   // Small delay between servo movements
    }
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    for (int i = 0; i < NUM_FINGER_SERVOS; i++) {
        servo_write_deg_channel(i, 0);  // Start at center position
        vTaskDelay(pdMS_TO_TICKS(50));   // Small delay between servo movements
    }

    for (int i = 0; i < NUM_FINGER_SERVOS; i++) {
        servo_write_deg_channel(i, 180);  // Start at center position
        vTaskDelay(pdMS_TO_TICKS(50));   // Small delay between servo movements
    }

    ESP_LOGI(TAG, "All servos initialized at neutral position");
    // vTaskDelay(pdMS_TO_TICKS(2000));
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
    uint32_t us = servo->min_pulse_us + ((deg - servo->min_angle) * (servo->max_pulse_us - servo->min_pulse_us)) / (servo->max_angle - servo->min_angle);
    
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


// ============================================================================
// GESTURE IMPLEMENTATION
// ============================================================================


void make_fist_gesture(void)
{
    ESP_LOGI(TAG, "Executing 'Make Fist' gesture");
    for (int i = 0; i < NUM_SERVOS; i++) {
        if(servos[i].direction == SERVO_DIR_REVERSE) {
            servo_write_deg_channel(i, servos[i].max_angle);
        } else {
            servo_write_deg_channel(i, servos[i].min_angle);
        }
    }
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}

void open_hand_gesture(void)
{
    ESP_LOGI(TAG, "Executing 'Open Hand' gesture");
    for (int i = 0; i < NUM_SERVOS; i++) {
        if(servos[i].direction == SERVO_DIR_REVERSE) {
            servo_write_deg_channel(i, servos[i].min_angle);
        } else {
            servo_write_deg_channel(i, servos[i].max_angle);
        }
    }
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}

void make_peace_gesture(void)
{
    ESP_LOGI(TAG, "Executing 'Peace' gesture");
    for (int i = 0; i < NUM_SERVOS; i++) {
        if (i == 1 || i == 2) {  // Index and Middle fingers
            if(servos[i].direction == SERVO_DIR_REVERSE) {
                servo_write_deg_channel(i, servos[i].min_angle);
            } else {
                servo_write_deg_channel(i, servos[i].max_angle);
            }
        } else {  // Other fingers
            if(servos[i].direction == SERVO_DIR_REVERSE) {
                servo_write_deg_channel(i, servos[i].max_angle);
            } else {
                servo_write_deg_channel(i, servos[i].min_angle);
            }
        }
    }
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}

void count_to_five_gesture(void)
{
    ESP_LOGI(TAG, "Executing 'Count to Five' gesture");
    // Start with all fingers closed
    make_fist_gesture();
    
    // Open fingers one by one
    for (int i = 0; i < 5; i++) {
        if(servos[i].direction == SERVO_DIR_REVERSE) {
            servo_write_deg_channel(i, servos[i].min_angle);
        } else {
            servo_write_deg_channel(i, servos[i].max_angle);
        }
        vTaskDelay(pdMS_TO_TICKS(500));  // Wait half a second between fingers
    }
    
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}

void rock_gesture(void)
{
    ESP_LOGI(TAG, "Executing 'Rock' gesture");
    for (int i = 0; i < NUM_SERVOS; i++) {
        if (i == 0 || i == 1 || i == 4) {  // Thumb, Index, Pinky fingers
            if(servos[i].direction == SERVO_DIR_NORMAL) {
                servo_write_deg_channel(i, servos[i].max_angle);
            } else {
                servo_write_deg_channel(i, servos[i].min_angle);
            }
        } else {  // Index and Middle fingers
            if(servos[i].direction == SERVO_DIR_NORMAL) {
                servo_write_deg_channel(i, servos[i].min_angle);
            } else {
                servo_write_deg_channel(i, servos[i].max_angle);
            }
        }
    }
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}

void flip_off_gesture(void)
{
    ESP_LOGI(TAG, "Executing 'Flip Off' gesture");
    for (int i = 0; i < NUM_SERVOS; i++) {
        if (i == 2) {  // Middle finger
            if(servos[i].direction == SERVO_DIR_REVERSE) {
                servo_write_deg_channel(i, servos[i].min_angle);
            } else {
                servo_write_deg_channel(i, servos[i].max_angle);
            }
        } else {  // Other fingers
            if(servos[i].direction == SERVO_DIR_REVERSE) {
                servo_write_deg_channel(i, servos[i].max_angle);
            } else {
                servo_write_deg_channel(i, servos[i].min_angle);
            }
        }
    }
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}

void custom_grip_1(void)
{
    ESP_LOGI(TAG, "Executing 'custom grip 1' gesture");
    // for (int i = 0; i < NUM_SERVOS; i++) {
    //     if(servos[i].direction == SERVO_DIR_REVERSE) {
    //         //always in reverse direction
    //         servo_write_deg_channel(i, 90);
    //     } 
    // }
    servo_write_deg_channel(0, 120); // thumb
    servo_write_deg_channel(1, 90); // pinky
    servo_write_deg_channel(2, 90); // ring
    servo_write_deg_channel(3, 120); // mid
    servo_write_deg_channel(4, 120); //index
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}

void custom_grip_2(void)
{
    ESP_LOGI(TAG, "Executing 'custom grip 2' gesture");
    // for (int i = 0; i < NUM_SERVOS; i++) {
    //     if(servos[i].direction == SERVO_DIR_REVERSE) {
    //         //always in reverse direction
    //         servo_write_deg_channel(i, 60);
    //     } 
    // }
    servo_write_deg_channel(0, 60); // thumb
    servo_write_deg_channel(1, 60); // pinky
    servo_write_deg_channel(2, 60); // ring
    servo_write_deg_channel(3, 60); // mid
    servo_write_deg_channel(4, 60); //index
    vTaskDelay(pdMS_TO_TICKS(1000));  // Hold for 1 second
}


// ============================================================================
// ROTARY ENCODER IMPLEMENTATION
// ============================================================================

static volatile int encoder_value = 0;
static volatile bool button_pressed = false;
static volatile uint32_t last_button_time = 0;
static const uint32_t DEBOUNCE_TIME_MS = 200;

// ISR for CLK pin (rotary encoder rotation)
static void IRAM_ATTR rotary_clk_isr_handler(void* arg)
{
    int clk_state = gpio_get_level(ROTARY_ENCODER_CLK_GPIO);
    int dt_state = gpio_get_level(ROTARY_ENCODER_DT_GPIO);
    
    if (clk_state == 0) {  // Falling edge on CLK
        if (dt_state == 1) {
            encoder_value++;  // Clockwise
        } else {
            encoder_value--;  // Counter-clockwise
        }
    }
}

// ISR for button press
static void IRAM_ATTR rotary_button_isr_handler(void* arg)
{
    uint32_t current_time = xTaskGetTickCountFromISR() * portTICK_PERIOD_MS;
    
    // Debounce check
    if (current_time - last_button_time > DEBOUNCE_TIME_MS) {
        button_pressed = true;
        last_button_time = current_time;
    }
}

// Initialize rotary encoder
esp_err_t rotary_encoder_init(void)
{
    ESP_LOGI(TAG, "Initializing rotary encoder");
    
    // Configure CLK pin (rotation detection)
    gpio_config_t clk_conf = {
        .pin_bit_mask = (1ULL << ROTARY_ENCODER_CLK_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE  // Trigger on falling edge
    };
    ESP_ERROR_CHECK(gpio_config(&clk_conf));
    
    // Configure DT pin (direction detection)
    gpio_config_t dt_conf = {
        .pin_bit_mask = (1ULL << ROTARY_ENCODER_DT_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    ESP_ERROR_CHECK(gpio_config(&dt_conf));
    
    // Configure SW pin (button)
    gpio_config_t sw_conf = {
        .pin_bit_mask = (1ULL << ROTARY_ENCODER_SW_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE  // Trigger on button press
    };
    ESP_ERROR_CHECK(gpio_config(&sw_conf));
    
    // Install GPIO ISR service
    gpio_install_isr_service(0);
    
    // Attach interrupt handlers
    gpio_isr_handler_add(ROTARY_ENCODER_CLK_GPIO, rotary_clk_isr_handler, NULL);
    gpio_isr_handler_add(ROTARY_ENCODER_SW_GPIO, rotary_button_isr_handler, NULL);
    
    ESP_LOGI(TAG, "Rotary encoder initialized on CLK:%d, DT:%d, SW:%d", 
             ROTARY_ENCODER_CLK_GPIO, ROTARY_ENCODER_DT_GPIO, ROTARY_ENCODER_SW_GPIO);
    
    return ESP_OK;
}

// Get current encoder value
int get_encoder_value(void)
{
    return encoder_value;
}

// Check if button was pressed and clear the flag
bool is_encoder_button_pressed(void)
{
    if (button_pressed) {
        button_pressed = false;
        return true;
    }
    return false;
}

// Calibration mode function
void start_calibration_mode(void)
{
    ESP_LOGI(TAG, "=== STARTING CALIBRATION MODE ===");
    ESP_LOGI(TAG, "Use rotary encoder to adjust angles");
    ESP_LOGI(TAG, "Press button to confirm each setting");
    
    calibration_state_t state = CAL_STATE_SELECT_FINGER;
    int selected_finger = 0;
    int temp_min_angle = 0;
    int temp_max_angle = 180;
    int current_angle = 90;
    
    encoder_value = selected_finger;  // Start at first finger
    
    while (state != CAL_STATE_DONE) {
        vTaskDelay(pdMS_TO_TICKS(50));  // Small delay for responsiveness
        
        switch (state) {
            case CAL_STATE_SELECT_FINGER:
                // Select which finger to calibrate
                selected_finger = encoder_value;
                if (selected_finger < 0) {
                    selected_finger = 0;
                    encoder_value = 0;
                }
                if (selected_finger >= NUM_SERVOS) {
                    selected_finger = NUM_SERVOS - 1;
                    encoder_value = NUM_SERVOS - 1;
                }
                
                // Visual feedback - move the selected servo slightly
                static int last_selected = -1;
                if (last_selected != selected_finger) {
                    ESP_LOGI(TAG, "Selected finger: %s", servos[selected_finger].name);
                    servo_write_deg_channel(selected_finger, 90);
                    last_selected = selected_finger;
                }
                
                if (is_encoder_button_pressed()) {
                    ESP_LOGI(TAG, "Calibrating %s", servos[selected_finger].name);
                    state = CAL_STATE_SET_MIN;
                    temp_min_angle = servos[selected_finger].min_angle;
                    encoder_value = temp_min_angle;
                }
                break;
                
            case CAL_STATE_SET_MIN:
                // Set minimum angle
                temp_min_angle = encoder_value;
                if (temp_min_angle < 0) {
                    temp_min_angle = 0;
                    encoder_value = 0;
                }
                if (temp_min_angle > 180) {
                    temp_min_angle = 180;
                    encoder_value = 180;
                }
                
                // Move servo to current angle for visual feedback
                servo_write_deg_channel(selected_finger, temp_min_angle);
                vTaskDelay(pdMS_TO_TICKS(20));
                
                if (is_encoder_button_pressed()) {
                    ESP_LOGI(TAG, "Min angle set to %d°", temp_min_angle);
                    state = CAL_STATE_SET_MAX;
                    temp_max_angle = servos[selected_finger].max_angle;
                    encoder_value = temp_max_angle;
                }
                break;
                
            case CAL_STATE_SET_MAX:
                // Set maximum angle
                temp_max_angle = encoder_value;
                if (temp_max_angle < 0) {
                    temp_max_angle = 0;
                    encoder_value = 0;
                }
                if (temp_max_angle > 180) {
                    temp_max_angle = 180;
                    encoder_value = 180;
                }
                
                // Move servo to current angle for visual feedback
                servo_write_deg_channel(selected_finger, temp_max_angle);
                vTaskDelay(pdMS_TO_TICKS(20));
                
                if (is_encoder_button_pressed()) {
                    // Validate min < max
                    if (temp_min_angle >= temp_max_angle) {
                        ESP_LOGW(TAG, "Invalid range! Min must be less than Max. Try again.");
                        state = CAL_STATE_SET_MIN;
                        encoder_value = temp_min_angle;
                    } else {
                        // Save the calibration
                        servos[selected_finger].min_angle = temp_min_angle;
                        servos[selected_finger].max_angle = temp_max_angle;
                        ESP_LOGI(TAG, "Max angle set to %d°", temp_max_angle);
                        ESP_LOGI(TAG, "%s calibration complete: Min=%d°, Max=%d°", 
                                 servos[selected_finger].name, temp_min_angle, temp_max_angle);
                        
                        // Return to finger selection or exit
                        ESP_LOGI(TAG, "Select another finger or wait 3 seconds to exit...");
                        state = CAL_STATE_SELECT_FINGER;
                        encoder_value = selected_finger;
                        
                        // Check if user wants to exit (no action for 3 seconds)
                        int timeout_count = 0;
                        while (timeout_count < 30) {  // 3 seconds / 100ms
                            if (is_encoder_button_pressed()) {
                                break;  // Continue calibration
                            }
                            vTaskDelay(pdMS_TO_TICKS(100));
                            timeout_count++;
                        }
                        
                        if (timeout_count >= 30) {
                            state = CAL_STATE_DONE;
                        }
                    }
                }
                break;
                
            case CAL_STATE_DONE:
                // Should not reach here
                break;
        }
    }
    
    ESP_LOGI(TAG, "=== CALIBRATION COMPLETE ===");
    ESP_LOGI(TAG, "Final calibration values:");
    for (int i = 0; i < NUM_SERVOS; i++) {
        ESP_LOGI(TAG, "%s: Min=%d°, Max=%d°", 
                 servos[i].name, servos[i].min_angle, servos[i].max_angle);
    }
}