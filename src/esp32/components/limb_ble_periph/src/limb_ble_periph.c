#include "limb_ble_periph.h"

#include "esp_err.h"
#include "esp_log.h"
#include "gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "nimble/nimble_port.h"
#include "nvs_flash.h"
#include "sensors_service.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char* const kDeviceName = "LIMBServer";

static const char* const kTag = "LIMB BLE periph";

// Performs necessary bluetooth LE setup.
static bool BleInit(void) {
  // NimBLE stores a bunch of stuff in Non-Volatile Storage.
  {
    esp_err_t err = nvs_flash_init();
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "Error calling nvs_flash_init: %s", esp_err_to_name(err));
      return false;
    }
  }

  {
    esp_err_t err = nimble_port_init();
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "Error calling nimble_port_init: %s",
               esp_err_to_name(err));
      return false;
    }
  }

  // We need to set up some GAP and GATT stuff and set callbacks for when NimBLE
  // is started.
  ble_svc_gap_init();
  {
    int err = ble_svc_gap_device_name_set(kDeviceName);
    if (err != 0) {
      ESP_LOGE("LIMB-GAP", "failed to set device name to %s, error code: %d",
               kDeviceName, err);
      return false;
    }
  }

  ble_svc_gatt_init();
  {
    int err = ble_gatts_count_cfg(get_sensor_services());
    if (err != 0) {
      ESP_LOGE(kTag, "Error calling ble_gatts_count_cfg.");
      return false;
    }
  }
  {
    int err = ble_gatts_add_svcs(get_sensor_services());
    if (err != 0) {
      ESP_LOGE(kTag, "Error calling ble_gatts_add_svcs.");
      return false;
    }
  }

  // Configure NimBLE host callbacks.

  ble_hs_cfg.reset_cb = BleStackResetCallback;
  // This callback is called when NimBLE is started.
  ble_hs_cfg.sync_cb = BleStackSyncCallback;
  ble_hs_cfg.store_status_cb = ble_store_util_status_rr;

  // For some reason the function isn't in any header file, so the official
  // way to call it is to first do a forward declaration.
  void ble_store_config_init(void);
  ble_store_config_init();

  return true;
}

void BleTask([[maybe_unused]] void* arg) {
  ESP_LOGI(kTag, "Initializing Bluetooth...");
  bool success = BleInit();
  if (success) {
    ESP_LOGI(kTag, "Starting Bluetooth...");
    nimble_port_run();
  }

  vTaskDelete(NULL);
}
