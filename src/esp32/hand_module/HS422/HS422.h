#include "driver/mcpwm_prelude.h"
#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// =================================

#define THUMB_SERVO_GPIO   21           
#define INDEX_SERVO_GPIO  22           
#define MID_SERVO_GPIO    9           
#define RING_SERVO_GPIO   12           
#define PINKY_SERVO_GPIO 15           

#define SERVO_MIN_PULSEWIDTH_US 500  // Minimum pulse width in microsecond
#define SERVO_MAX_PULSEWIDTH_US 2500  // Maximum pulse width in microsecond
#define SERVO_MIN_DEGREE        0   // Minimum angle
#define SERVO_MAX_DEGREE        180    // Maximum angle

#define SERVO_TIMEBASE_RESOLUTION_HZ 1000000  // 1MHz, 1us per tick
#define SERVO_TIMEBASE_PERIOD        20000    // 20000 ticks, 20ms

#define NUM_SERVOS        5
// =================================

typedef enum {
    SERVO_DIR_NORMAL = 1,     // Clockwise increases angle (0° = open, 180° = closed)
    SERVO_DIR_REVERSE = -1    // Counter-clockwise increases angle (0° = closed, 180° = open)
} servo_direction_t;

typedef struct {
    int gpio_pin;                   // GPIO pin for this servo
    int min_angle;                  // Minimum angle in degrees
    int max_angle;                  // Maximum angle in degrees
    uint32_t min_pulse_us;         // 
    uint32_t max_pulse_us;         // Maximum pulse width in microseconds
    int current_angle;             // Current servo position
    float current_force;           // Current force applied, measured by FSR
    servo_direction_t direction;   // Servo rotation direction
    const char* name;              // Name for debugging
    
    // MCPWM handles for this servo
    mcpwm_cmpr_handle_t comparator;
    mcpwm_gen_handle_t generator;
    int operator_index;            // Which operator this servo uses
} servo_config_t;


// Function declarations
esp_err_t servo_init(void);
esp_err_t servo_set_limits(int channel, int min_angle, int max_angle);
esp_err_t servo_set_pulse_range(int channel, uint32_t min_us, uint32_t max_us);
void servo_write_deg_channel(int channel, int deg);
void servo_write_us_channel(int channel, uint32_t pulse_us);
void servo_write_all_deg(int deg);
void servo_enable_channel(int channel, bool enable);
int servo_get_current_angle(int channel);
void servo_print_info(int channel);
void servo_main(void);