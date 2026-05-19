#include "HS422_led.h"
#include "esp_log.h"

static const char* TAG = "MAIN";

void app_main() {
  ESP_LOGI(TAG, "Starting servo control application");
  vTaskDelay(pdMS_TO_TICKS(2000));

  // Initialize all servos
  ESP_LOGI(TAG, "Initializing servos...");
  servo_led_init();
  vTaskDelay(pdMS_TO_TICKS(1000));

  // Initialize rotary encoder
  ESP_LOGI(TAG, "Initializing rotary encoder...");
  rotary_encoder_init();
  vTaskDelay(pdMS_TO_TICKS(1000));

  // Start calibration mode
  // Uncomment the line below to enter calibration mode
  // start_calibration_mode();

  ESP_LOGI(TAG, "Starting servo test loop...");
  ESP_LOGI(TAG,
           "To enter calibration mode, uncomment start_calibration_mode() in "
           "main.c");

  // Normal operation - test servo movements
  while (1) {
    // start_calibration_mode();

    make_fist_gesture();
    vTaskDelay(pdMS_TO_TICKS(2000));
    open_hand_gesture();
    vTaskDelay(pdMS_TO_TICKS(2000));
    count_to_five_gesture();
    vTaskDelay(pdMS_TO_TICKS(2000));
    make_peace_gesture();
    vTaskDelay(pdMS_TO_TICKS(2000));
    rock_gesture();
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}