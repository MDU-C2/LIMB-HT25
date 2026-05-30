#include <math.h>
#include <stdio.h>

#include "adc_service.h"
#include "can_driver.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "HAND_MAIN";

/* --- Emergency Escape Directive --- */
// Set to 1 to enable emergency fixed grip after calibration, 0 for normal PD
// control
#define SAFEWAY 1

/* --- Control State Machine --- */
typedef enum {
  E0_IDLE,
  E1_START,     // Calibration and initial setup
  E2_STABLE,    // Monitoring pressure stability
  E3_REACTION,  // PD Control burst
  E4_SAFEWAY    // EMERGENCY: Fixed firm grip forever
} state_t;

state_t current_state = E0_IDLE;

/* --- Global Control Variables --- */
float setpoint_actual = 0;
float setpoint_base = 0;
float setpoint_max = 0;
float tolerance = 0.0f;
float error_previo = 0;
float Kp = 0.5;
float Kd = 0.2;
int servo_angle = 180;
wstats_t resultss = {0.0f, 0.0f};

#define CAN_TX_GPIO 5
#define CAN_RX_GPIO 8
#define CAN_BAUDRATE 1000000

void loop_control(void);

static void reenable_can_task([[maybe_unused]] void* pvParameter) {
  while (true) {
    can_automatically_reenable_on_bus_off();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

void app_main(void) {
  ESP_LOGI(TAG, "Starting Brain Node - Safe Mode Available");

  if (init_adc_service() != ESP_OK) {
    ESP_LOGE(TAG, "ADC Service Init Failed!");
    abort();
  }

  // 2. Initialize CAN Bus
  {
    esp_err_t err = can_init(CAN_TX_GPIO, CAN_RX_GPIO, CAN_BAUDRATE, NULL);
    if (err != ESP_OK) {
      ESP_LOGE(TAG, "Failed to initialize CAN: %s", esp_err_to_name(err));
      abort();
    }
    ESP_LOGI(TAG, "CAN Bus Initialized ");
  }

  enum {
    TASK_CAN_RX_PRIORITY = 5,
  };

#if CONFIG_FORCE_REENABLE_CAN_ON_BUS_OFF
  err = xTaskCreate(reenable_can_task, "reenable_can_task", TASK_STACK_DEPTH,
                    NULL, TASK_CAN_RX_PRIORITY + 1, NULL);
  if (err != pdPASS) {
    ESP_LOGE(TAG, "Failed to create reenable_can_task, err code: %d");
    abort();
  }
#endif

  while (1) {
    loop_control();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

void loop_control(void) {
  uint8_t can_msg_data;
  uint32_t rx_id;
  uint8_t rx_data[8];
  uint8_t rx_len = 1;

  if (can_receive(&rx_id, rx_data, &rx_len, 0) == ESP_OK) {
    bool should_close = rx_data[0];

    ESP_LOGI(TAG, "CAN RECIBID: ID %X, Data: %d", rx_id, should_close);
    if (rx_id == CAN_ID_ROBOT_HAND_SET_GRIP_STATE) {
      if (should_close) {
        ESP_LOGW(TAG, ">>> REMOTE START RECEIVED!");
        current_state = E1_START;
      } else {  // COMANDO STOP
        ESP_LOGW(TAG, ">>> REMOTE STOP RECEIVED! Opening hand...");
        current_state = E0_IDLE;

        servo_angle = 180;
        uint8_t open_msg = (uint8_t)servo_angle;
        // NOTE: We're using the thumb activation as a hack. Sorry.
        can_send(CAN_ID_ROBOT_THUMB_ACTUATION, &open_msg, 1, 0);
      }
    }
  }

  switch (current_state) {
    case E0_IDLE:
      break;

    case E1_START:
      ESP_LOGI(TAG, "--- [E1] START: CALIBRATION ---");

// // Step A: Fully Open (180)
// servo_angle = 180;
// can_msg_data = (uint8_t)servo_angle;
// ESP_LOGI(TAG, "Step A: Opening hand (Angle: 180)");
// can_send(CAN_ID_ROBOT_THUMB_ACTUATION, &can_msg_data, 1);
// vTaskDelay(pdMS_TO_TICKS(3000));

// resultss = get_window_stats();
// setpoint_base = resultss.mean;
// ESP_LOGI(TAG, "Calibration Base Pressure: %.2f mV", setpoint_base);

// // Step B: Fully Closed (0)
// servo_angle = 0;
// can_msg_data = (uint8_t)servo_angle;
// ESP_LOGI(TAG, "Step B: Closing hand (Angle: 0)");
// can_send(CAN_ID_ROBOT_THUMB_ACTUATION, &can_msg_data, 1);
// vTaskDelay(pdMS_TO_TICKS(3000));

// resultss = get_window_stats();
// setpoint_max = resultss.mean;
// ESP_LOGI(TAG, "Calibration Max Pressure: %.2f mV", setpoint_max);

// // Step C: Initial Grip (35)
// servo_angle = 30;
// can_msg_data = (uint8_t)servo_angle;
// ESP_LOGI(TAG, "Step C: Setting initial grip (Angle: 35)");
// can_send(CAN_ID_ROBOT_THUMB_ACTUATION, &can_msg_data, 1);
// vTaskDelay(pdMS_TO_TICKS(3000));

// resultss = get_window_stats();
// setpoint_actual = resultss.mean;
// ESP_LOGI(TAG, "Setpoint Actual fijado en: %.2f", setpoint_actual);

// /* --- EMERGENCY PLAN CHECK --- */
#if SAFEWAY == 1
      ESP_LOGW(TAG, "!!! SAFE GRIP ACTIVE!!!");
      current_state = E4_SAFEWAY;
#else
      current_state = E2_STABLE;
#endif

      break;

    case E2_STABLE:
      wstats_t stats = get_window_stats();
      tolerance = setpoint_actual / 3.0f;

      if (fabsf(stats.mean - setpoint_actual) > tolerance ||
          stats.variance > 50.0f) {
        ESP_LOGW(TAG, "--- [E2 -> E3] INSTABILITY DETECTED! ---");
        current_state = E3_REACTION;
      }
      break;

    case E3_REACTION:
      ESP_LOGI(TAG, "--- [E3] REACTION: PD BURST ---");

      for (int i = 0; i < 5; i++) {
        float current_p = get_instant_pressure();
        float error = setpoint_actual - current_p;
        int delta = (int)(Kp * error + Kd * (error - error_previo));
        error_previo = error;

        // Delta limits for smoothness (+/- 3 deg)
        if (delta > 3) delta = 3;
        if (delta < -3) delta = -3;

        // Inverted Logic: subtract delta to close/open
        servo_angle -= delta;

        // Hardware safety constraints
        if (servo_angle > 180) servo_angle = 180;
        if (servo_angle < 5) servo_angle = 5;

        ESP_LOGI(TAG, "PD Step %d | Angle: %d", i, servo_angle);

        // can_msg_data = (uint8_t)servo_angle;
        // can_send(CAN_ID_ROBOT_THUMB_ACTUATION, &can_msg_data, 1);

        vTaskDelay(pdMS_TO_TICKS(100));
      }
      vTaskDelay(pdMS_TO_TICKS(100));

      // --- POST-CONTROL CHECK ---
      // We take a new window of data to see if the PD burst fixed the slip
      wstats_t post_stats = get_window_stats();
      tolerance = setpoint_actual / 3.0f;

      // Check if it's still unstable using your E2 condition
      if (fabsf(post_stats.mean - setpoint_actual) > tolerance ||
          post_stats.variance > 50.0f) {
        ESP_LOGW(TAG, "Post-PD check: STILL UNSTABLE! Increasing force.");

        // Increase setpoint by 20mV to force a tighter grip in the next burst
        setpoint_actual += 20.0f;

        // Keep current_state = E3_REACTION (implicitly) to loop back
        // This creates a recursive tightening until stable
      } else {
        ESP_LOGI(TAG, "Post-PD check: STABLE. Moving to E2.");
        current_state = E2_STABLE;
      }
      break;

    case E4_SAFEWAY:
      // Static variable to ensure the CAN message only sends ONCE
      servo_angle = 10;
      can_msg_data = (uint8_t)servo_angle;
      can_send(CAN_ID_ROBOT_THUMB_ACTUATION, &can_msg_data, 1, 0);

      vTaskDelay(pdMS_TO_TICKS(5000));
      break;
  }
}
