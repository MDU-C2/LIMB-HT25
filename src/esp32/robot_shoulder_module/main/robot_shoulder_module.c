#include "esp_log.h"

const char* const TAG = "Shoulder module";

void app_main(void) {
  const char* world = "world";
  ESP_LOGI(TAG, "Hello %s!", world);
}
