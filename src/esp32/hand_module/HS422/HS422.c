#include "HS422.h" 


static const char *TAG = "HS422_Servo";


// Servo configurations
static servo_config_t servos[NUM_SERVOS] = {
    // Thumb servo - smaller range of motion
    {
        .gpio_pin = THUMB_SERVO_GPIO,
        .min_angle = 150,
        .max_angle = SERVO_MAX_DEGREE,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .current_angle = 0,
        .current_force = 0.0f,
        .direction = SERVO_DIR_REVERSE,
        .name = "Thumb",
        .operator_index = 0
    },
    // Index finger - full range
    {
        .gpio_pin = INDEX_SERVO_GPIO,
        .min_angle = 120,
        .max_angle = SERVO_MAX_DEGREE,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .current_angle = 0,
        .current_force = 0.0f,
        .direction = SERVO_DIR_REVERSE,
        .name = "Index",
        .operator_index = 0
    },
    // Middle finger - full range
    {
        .gpio_pin = MID_SERVO_GPIO,
        .min_angle = 110,
        .max_angle = SERVO_MAX_DEGREE,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .current_angle = 0,
        .current_force = 0.0f,
        .direction = SERVO_DIR_REVERSE,
        .name = "Middle",
        .operator_index = 1
    },
    // Ring finger - slightly limited range
    {
        .gpio_pin = RING_SERVO_GPIO,
        .min_angle = SERVO_MIN_DEGREE,
        .max_angle = 70,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .current_angle = 0,
        .current_force = 0.0f,
        .direction = SERVO_DIR_NORMAL,
        .name = "Ring",
        .operator_index = 1
    },
    // Pinky - limited range of motion
    {
        .gpio_pin = PINKY_SERVO_GPIO,
        .min_angle = SERVO_MIN_DEGREE,
        .max_angle = 54,
        .min_pulse_us = 1000,
        .max_pulse_us = 2000,
        .current_angle = 0,
        .current_force = 0.0f,
        .direction = SERVO_DIR_NORMAL,
        .name = "Pinky",
        .operator_index = 2
    }
};

mcpwm_timer_handle_t timer = NULL;
mcpwm_oper_handle_t operators[3] = {NULL};
mcpwm_cmpr_handle_t comparator[NUM_SERVOS] = {NULL};
mcpwm_gen_handle_t generator[NUM_SERVOS] = {NULL};


static uint32_t servo_angle_to_us(int channel, int angle)
{
    if (channel < 0 || channel >= NUM_SERVOS) return 1500;
    
    servo_config_t *servo = &servos[channel];
    
    // direction correction
    if (servo->direction == SERVO_DIR_REVERSE) {
        // Reverse direction: flip the angle
        if (angle < servo->min_angle) angle = servo->max_angle;
        if (angle > servo->max_angle) angle = servo->min_angle;
    } else {
        if (angle < servo->min_angle) angle = servo->min_angle;
        if (angle > servo->max_angle) angle = servo->max_angle;
    }
    
    // Convert to pulse width
    return servo->min_pulse_us + 
           (angle - servo->min_angle) * (servo->max_pulse_us - servo->min_pulse_us) / 
           (servo->max_angle - servo->min_angle);
}


void servo_write_finger_position(int channel, int finger_percent)
{
    if (channel < 0 || channel >= NUM_SERVOS) {
        ESP_LOGW(TAG, "Invalid servo channel: %d", channel);
        return;
    }
    
    // Clamp percentage
    if (finger_percent < 0) finger_percent = 0;
    if (finger_percent > 100) finger_percent = 100;
    
    servo_config_t *servo = &servos[channel];
    
    // Convert percentage to logical angle
    int logical_angle = servo->min_angle + 
                       (finger_percent * (servo->max_angle - servo->min_angle)) / 100;
    
    // Use existing function (handles direction automatically)
    servo_write_deg_channel(channel, logical_angle);
    
    ESP_LOGI(TAG, "%s finger set to %d%% closed (logical angle: %d°)", 
             servo->name, finger_percent, logical_angle);
}

esp_err_t servo_init(void)
{
    ESP_LOGI(TAG, "Initializing MCPWM for %d servos", NUM_SERVOS);
    
    mcpwm_timer_config_t timer_config = {
        .group_id = 0,
        .clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT,
        .resolution_hz = SERVO_TIMEBASE_RESOLUTION_HZ,
        .period_ticks = SERVO_TIMEBASE_PERIOD,
        .count_mode = MCPWM_TIMER_COUNT_MODE_UP,
    };
    ESP_ERROR_CHECK(mcpwm_new_timer(&timer_config, &timer));
    
    ESP_LOGI(TAG, "Create 3 operators for 5 servos");

    for(int op = 0; op < 3; op++) {
        mcpwm_operator_config_t operator_config = {
            .group_id = 0, // operator must be in the same group to the timer
        };
        ESP_ERROR_CHECK(mcpwm_new_operator(&operator_config, &operators[op]));
        ESP_ERROR_CHECK(mcpwm_operator_connect_timer(operators[op], timer));
        ESP_LOGI(TAG, "Created operator %d", op);
    }
    
    for(int i = 0; i < NUM_SERVOS; i++) {
        servo_config_t *servo = &servos[i];
        ESP_LOGI(TAG, "Setting up %s servo (ch%d) on GPIO %d, range: %d° to %d°", 
                 servo->name, i, servo->gpio_pin, servo->min_angle, servo->max_angle);
        
        // Create comparator
        mcpwm_comparator_config_t comparator_config = {
            .flags.update_cmp_on_tez = true,
        };
        ESP_ERROR_CHECK(mcpwm_new_comparator(operators[servo->operator_index], &comparator_config, &servo->comparator));

        // Create generator
        mcpwm_generator_config_t generator_config = {
            .gen_gpio_num = servo->gpio_pin,
        };
        ESP_ERROR_CHECK(mcpwm_new_generator(operators[servo->operator_index], &generator_config, &servo->generator));

        // Set initial position to middle of servo's range
        uint32_t initial_us = servo_angle_to_us(i, 0);
        ESP_ERROR_CHECK(mcpwm_comparator_set_compare_value(servo->comparator, initial_us));
        servo->current_angle = 0;

        // Configure PWM actions
        ESP_ERROR_CHECK(mcpwm_generator_set_action_on_timer_event(servo->generator,
                        MCPWM_GEN_TIMER_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, MCPWM_TIMER_EVENT_EMPTY, MCPWM_GEN_ACTION_HIGH)));
        ESP_ERROR_CHECK(mcpwm_generator_set_action_on_compare_event(servo->generator,
                        MCPWM_GEN_COMPARE_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, servo->comparator, MCPWM_GEN_ACTION_LOW)));
    }
    ESP_LOGI(TAG, "Enable and start timer");
    ESP_ERROR_CHECK(mcpwm_timer_enable(timer));
    ESP_ERROR_CHECK(mcpwm_timer_start_stop(timer, MCPWM_TIMER_START_NO_STOP));

    ESP_LOGI(TAG, "All %d servos initialized successfully", NUM_SERVOS);
    return ESP_OK;
}




//Set servo position in degrees
void servo_write_deg_channel(int channel, int deg)
{
    if (channel < 0 || channel >= NUM_SERVOS) {
        ESP_LOGW(TAG, "Invalid servo channel: %d", channel);
        return;
    }
    
    servo_config_t *servo = &servos[channel];

    uint32_t pulse_us = servo_angle_to_us(channel, deg);
    ESP_ERROR_CHECK(mcpwm_comparator_set_compare_value(servo->comparator, pulse_us));
    
    // Update current position
    servo->current_angle = deg;
    if (deg < servo->min_angle) servo->current_angle = servo->min_angle;
    if (deg > servo->max_angle) servo->current_angle = servo->max_angle;
    
    ESP_LOGI(TAG, "%s servo set to %d° (%lu μs)", servo->name, servo->current_angle, pulse_us);
}

// Set servo position in microseconds
void servo_write_us_channel(int channel, uint32_t pulse_us)
{
    if (channel < 0 || channel >= NUM_SERVOS) {
        ESP_LOGW(TAG, "Invalid servo channel: %d", channel);
        return;
    }
    
    servo_config_t *servo = &servos[channel];

    // Clamp to servo's pulse range
    if (pulse_us < servo->min_pulse_us) pulse_us = servo->min_pulse_us;
    if (pulse_us > servo->max_pulse_us) pulse_us = servo->max_pulse_us;

    ESP_ERROR_CHECK(mcpwm_comparator_set_compare_value(servo->comparator, pulse_us));
    ESP_LOGD(TAG, "%s servo set to %lu μs", servo->name, pulse_us);
}


// Set all servos to same angle (respecting individual limits)
void servo_write_all_deg(int deg)
{
    ESP_LOGI(TAG, "Setting all servos to %d° (within individual limits)", deg);
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_write_deg_channel(i, deg);
    }
}

int servo_get_current_angle(int channel)
{
    if (channel < 0 || channel >= NUM_SERVOS) {
        ESP_LOGW(TAG, "Invalid servo channel: %d", channel);
        return 0;
    }
    
    return servos[channel].current_angle;
}


void servo_print_info(int channel)
{
    if (channel < 0 || channel >= NUM_SERVOS) {
        ESP_LOGW(TAG, "Invalid servo channel: %d", channel);
        return;
    }
    
    servo_config_t *servo = &servos[channel];
    ESP_LOGI(TAG, "=== %s Servo (Channel %d) ===", servo->name, channel);
    ESP_LOGI(TAG, "GPIO: %d", servo->gpio_pin);
    ESP_LOGI(TAG, "Angle Range: %d° to %d°", servo->min_angle, servo->max_angle);
    ESP_LOGI(TAG, "Pulse Range: %lu - %lu μs", servo->min_pulse_us, servo->max_pulse_us);
    ESP_LOGI(TAG, "Current Angle: %d°", servo->current_angle);
}

void close_all_fingers(void)
{
    ESP_LOGI(TAG, "Closing all fingers");
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_write_finger_position(i, 100);  // 100% closed
    }
}

void open_all_fingers(void)
{
    ESP_LOGI(TAG, "Opening all fingers");
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_write_finger_position(i, 0);    // 0% closed (fully open)
    }
}

void set_finger_curl(int channel, int curl_percent)
{
    ESP_LOGI(TAG, "Setting finger on channel %d to %d%% curl", channel, curl_percent);
    servo_write_finger_position(channel, curl_percent);
}

void servo_main(void)
{
    ESP_LOGI(TAG, "Starting servo demo with direction handling");
        
    // Initialize all servos
    ESP_ERROR_CHECK(servo_init());
    vTaskDelay(pdMS_TO_TICKS(1000));

    // Print servo info
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_print_info(i);
    }

    while (1) {
        // Test percentage-based control
        /*for(int curl = 0; curl <= 100; curl += 10) {
            set_finger_curl(0, curl); // Thumb
            vTaskDelay(pdMS_TO_TICKS(1000));
        }*/
        servo_print_info(0); // Thumb
        vTaskDelay(pdMS_TO_TICKS(500));
        servo_print_info(1); // Index
        vTaskDelay(pdMS_TO_TICKS(500));
        servo_print_info(2); // Middle
        vTaskDelay(pdMS_TO_TICKS(500));
        servo_print_info(3); // Ring
        vTaskDelay(pdMS_TO_TICKS(500));
        servo_print_info(4); // Pinky

        vTaskDelay(pdMS_TO_TICKS(2000));
        
        
    }
}
