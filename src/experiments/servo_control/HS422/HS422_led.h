#include "driver/ledc.h"
#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// =================================
#define BUTTON_Next_GPIO  19    // GPIO for button to move to next position (active low)
#define BUTTON_Prev_GPIO  20    // GPIO for button to move to previous position (

// Rotary Encoder GPIO pins
#define ROTARY_ENCODER_CLK_GPIO  8    // CLK pin (A phase)
#define ROTARY_ENCODER_DT_GPIO   7    // DT pin (B phase)
#define ROTARY_ENCODER_SW_GPIO   6    // SW pin (button/switch)

#define THUMB_SERVO_GPIO    0          
#define INDEX_SERVO_GPIO    1           
#define MID_SERVO_GPIO      2           
#define RING_SERVO_GPIO     3           
#define PINKY_SERVO_GPIO    4 

#define SERVO_FREQ_HZ     50          // 50 Hz = 20 ms period
#define SERVO_RES_BITS    13          // resolution; 13 bits = 8191 ticks

#define SERVO_MIN_US      500        // 500 
#define SERVO_MAX_US      2500        // 2500 
#define SERVO_MIN_DEG     0
#define SERVO_MAX_DEG     180
// =================================

#define SERVO_PERIOD_US   (1000000UL / SERVO_FREQ_HZ)
#define SERVO_MAX_DUTY    ((1U << SERVO_RES_BITS) - 1)

#define NUM_SERVOS        5

// Direction enum
typedef enum {
    SERVO_DIR_NORMAL = 1,
    SERVO_DIR_REVERSE = -1
} servo_direction_t;

typedef struct {
    int gpio_pin;                   // GPIO pin for this servo
    ledc_channel_t ledc_channel;    // LEDC channel (0-7)
    int min_angle;                  // Minimum angle in degrees
    int max_angle;                  // Maximum angle in degrees
    uint32_t min_pulse_us;         // Minimum pulse width in microseconds
    uint32_t max_pulse_us;         // Maximum pulse width in microseconds
    int current_angle;             // Current servo position
    float current_force;           // Current force applied, measured by FSR
    servo_direction_t direction;   // Direction of servo movement
    const char* name;              // Human-readable name for debugging
} servo_config_t;

// Calibration state
typedef enum {
    CAL_STATE_SELECT_FINGER,    // Selecting which finger to calibrate
    CAL_STATE_SET_MIN,          // Setting minimum angle
    CAL_STATE_SET_MAX,          // Setting maximum angle
    CAL_STATE_DONE              // Calibration complete
} calibration_state_t;

// Function declarations
uint32_t us_to_duty(uint32_t us);
esp_err_t servo_led_init(void);
void servo_write_deg_channel(int channel, int deg);
void servo_write_all_deg(int deg);
void close_all_fingers(void);
void open_all_fingers(void);

// Gesture functions
void make_fist_gesture(void);
void open_hand_gesture(void);
void make_peace_gesture(void);
void count_to_five_gesture(void);
void rock_gesture(void);
void flip_off_gesture(void);

// Rotary encoder functions
esp_err_t rotary_encoder_init(void);
void start_calibration_mode(void);
int get_encoder_value(void);
bool is_encoder_button_pressed(void);
