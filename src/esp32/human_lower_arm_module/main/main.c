#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

// --- Project Component Includes ---
#include "adc_service.h"
#include "ble_service.h"
#include "imu_service.h"

static const char* TAG_MAIN = "APP_MAIN";

void app_main(void) {
  ESP_LOGI(TAG_MAIN, "Initializing Micro-Streaming System...");

  /**
   * 1. Central Synchronization: Create the Event Group
   * This group coordinates ADC (EMG/Piezo), IMU, and BLE tasks.
   * It allows low-latency signaling between producers and the BLE consumer.
   */
  EventGroupHandle_t sync_group = xEventGroupCreate();
  if (sync_group == NULL) {
    ESP_LOGE(TAG_MAIN, "Critical Error: Could not create Event Group!");
    abort();
  }

  /**
   * 3. Define the Streaming Bitmask
   * This mask tells the BLE service which synchronization bits to monitor.
   * We listen to three independent data streams: EMG, PIEZO, and IMU.
   */
  EventBits_t streaming_mask =
      (ADC_EMG_STREAM_BIT | ADC_PIEZO_STREAM_BIT | IMU_STREAM_BIT);

  /**
   * 4. Start BLE Dispatcher Service
   * Initializing BLE first ensures the stack and GATT characteristics are ready
   * before sensor data starts flowing.
   */
  if (ble_service_start(sync_group, streaming_mask) != ESP_OK) {
    ESP_LOGE(TAG_MAIN, "Failed to start BLE Service!");
    abort();
  } else {
    ESP_LOGI(TAG_MAIN,
             "BLE Service Running: Listening for Micro-Streaming events.");
  }

  // Small delay to ensure BLE stack stability before sensor interrupts fire
  vTaskDelay(pdMS_TO_TICKS(100));

#if CONFIG_IMU_ENABLED
  // 5. Start IMU Service (I2C Scanning & 100Hz Task)
  if (imu_service_start(sync_group) != ESP_OK) {
    ESP_LOGE(TAG_MAIN, "Failed to start IMU Service!");
    abort();
  } else {
    ESP_LOGI(TAG_MAIN, "IMU Service Running.");
  }
#endif

  // 6. Start ADC Service (EMG & Piezo via DMA)
  // This starts the high-speed continuous sampling engine
  if (adc_service_init(sync_group) != ESP_OK) {
    ESP_LOGE(TAG_MAIN, "Failed to start ADC Service!");
    abort();
  } else {
    ESP_LOGI(TAG_MAIN, "ADC Service Running.");
  }

  ESP_LOGI(TAG_MAIN, "System fully operational.");

  /**
   * Main Monitoring Loop
   * Keep the main task alive to monitor system health and memory leaks.
   */
  while (1) {
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}
