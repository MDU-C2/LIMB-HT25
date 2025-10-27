#include "sensors_service.h"

#include "esp_log.h"
#include "host/ble_att.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"

static const char* const kLimbTag = "LIMB BLE Periph";

// UUID corresponds to 24011525-1212-efde-1523-785feabcd122.
static const ble_uuid128_t kEmgCharUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x24);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static uint16_t gEmgSubscriptionHandle;
static bool gEmgPeerNotifyEnabled;
static uint16_t gEmgValHandle;
static uint8_t gEmgVal[kEmgBufSize] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

CharacteristicBuffer get_emg_buf(void) {
  return (CharacteristicBuffer){.data = gEmgVal, .size = kEmgBufSize};
}

bool TryNotifyEmgSubscribers(void) {
  if (gEmgPeerNotifyEnabled) {
    int err = ble_gatts_notify(gEmgSubscriptionHandle, gEmgValHandle);
    if (err) {
      ESP_LOGW(kLimbTag, "Failed to send EMG notification, err(%d).", err);
    } else {
      ESP_LOGI(kLimbTag, "EMG notification sent.");
    }
    return !err;
  }
  return false;
}

// IMU characteristic.

// UUID corresponds to 25011525-1212-efde-1523-785feabcd122.
static const ble_uuid128_t kImuCharUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x25);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static uint16_t gImuSubscriptionHandle;
static bool gImuPeerNotifyEnabled;
static uint16_t gImuValHandle;
static uint8_t gImuVal[kImuBufSize] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

CharacteristicBuffer get_imu_buf(void) {
  return (CharacteristicBuffer){.data = gImuVal, .size = kImuBufSize};
}

bool TryNotifyImuSubscribers(void) {
  if (gImuPeerNotifyEnabled) {
    int err = ble_gatts_notify(gImuSubscriptionHandle, gImuValHandle);
    if (err) {
      ESP_LOGW(kLimbTag, "Failed to send IMU notification, err(%d).", err);
    } else {
      ESP_LOGI(kLimbTag, "IMU notification sent.");
    }
    return !err;
  }
  return false;
}

// Piezo characteristic.

// UUID corresponds to 26011525-1212-efde-1523-785feabcd122.
static const ble_uuid128_t kPiezoCharUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x26);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static uint16_t gPiezoSubscriptionHandle;
static bool gPiezoPeerNotifyEnabled;
static uint16_t gPiezoValHandle;
static uint8_t gPiezoVal[kPiezoBufSize] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

CharacteristicBuffer get_piezo_buf(void) {
  return (CharacteristicBuffer){.data = gPiezoVal, .size = kPiezoBufSize};
}

bool TryNotifyPiezoSubscribers(void) {
  if (gPiezoPeerNotifyEnabled) {
    // TODO(johan): Figure out if this function is blocking. If not, we might
    // need use a mutex for the buffer.
    int err = ble_gatts_notify(gPiezoSubscriptionHandle, gPiezoValHandle);
    if (err) {
      ESP_LOGW(kLimbTag, "Failed to send piezo notification, err(%d).", err);
    } else {
      ESP_LOGI(kLimbTag, "Piezo notification sent.");
    }
    return !err;
  }
  return false;
}

// Callback function for when a characteristic is accessed
static int CharAccess(uint16_t connection_handle, uint16_t attribute_handle,
                      struct ble_gatt_access_ctxt* context,
                      [[maybe_unused]] void* args) {
  if (context->op != BLE_GATT_ACCESS_OP_READ_CHR) {
    char uuid_buf[BLE_UUID_STR_LEN] = {0};
    switch (context->op) {
      case BLE_GATT_ACCESS_OP_WRITE_DSC:
        ESP_LOGW(kLimbTag,
                 "Invalid write access on characteristic descriptor, UUID: ",
                 ble_uuid_to_str(context->dsc->uuid, uuid_buf));
        return BLE_ATT_ERR_WRITE_NOT_PERMITTED;
      case BLE_GATT_ACCESS_OP_WRITE_CHR: {
        ESP_LOGW(kLimbTag, "Invalid read access on characteristic, UUID: ",
                 ble_uuid_to_str(context->chr->uuid, uuid_buf));
        return BLE_ATT_ERR_WRITE_NOT_PERMITTED;
      }
      case BLE_GATT_ACCESS_OP_READ_DSC: {
        // We shouldn't have any descriptors for the characteristics.
        ESP_LOGW(kLimbTag,
                 "Invalid read access on characteristic descriptor, UUID: ",
                 ble_uuid_to_str(context->dsc->uuid, uuid_buf));
        return BLE_ATT_ERR_READ_NOT_PERMITTED;
      }
      default: {
        // Unreachable.
      };
    }
    assert(false &&
           "GATT operation that isn't r/w on characteristic or descriptor.");
    return BLE_ATT_ERR_UNLIKELY;
  }

  if (connection_handle != BLE_HS_CONN_HANDLE_NONE) {
    ESP_LOGI(kLimbTag, "characteristic read; conn_handle=%d attr_handle=%d",
             connection_handle, attribute_handle);
  } else {
    ESP_LOGI(kLimbTag, "characteristic read by nimble stack; attr_handle=%d",
             attribute_handle);
  }

  uint8_t* buffer = NULL;
  size_t buffer_size = 0;

  // Determine which characteristic we should send.
  if (attribute_handle == gEmgValHandle) {
    ESP_LOGI(kLimbTag, "EMG read request.", connection_handle,
             attribute_handle);
    buffer = gEmgVal;
    buffer_size = sizeof(gEmgVal);
  } else if (attribute_handle == gImuValHandle) {
    ESP_LOGI(kLimbTag, "IMU read request.", connection_handle,
             attribute_handle);
    buffer = gImuVal;
    buffer_size = sizeof(gImuVal);
  } else if (attribute_handle == gPiezoValHandle) {
    ESP_LOGI(kLimbTag, "Piezo read request.", connection_handle,
             attribute_handle);
    buffer = gPiezoVal;
    buffer_size = sizeof(gPiezoVal);
  } else {
    ESP_LOGW(kLimbTag,
             "Characteristic access with an invalid attribute_handle [%d].",
             attribute_handle);
    assert(false && "Char access with an invalid attribute_handle.");
    return BLE_ATT_ERR_INVALID_HANDLE;
  }

  int err = os_mbuf_append(context->om, buffer, buffer_size);
  if (err != 0) {
    ESP_LOGE(kLimbTag,
             "Error appending characterictic value to os memory buffer. [%d]",
             err);
    assert(false &&
           "Error appending characterictic value to os memory buffer.");
    return BLE_ATT_ERR_INSUFFICIENT_RES;
  }

  return 0;
}

// UUID corresponds to 23011525-1212-efde-1523-785feabcd122
static const ble_uuid128_t kServiceUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x23);

// The definition for the BLE characteristics.
static const struct ble_gatt_chr_def kServiceChars[] = {
    {
        .uuid = &kEmgCharUuid.u,
        .access_cb = &CharAccess,
        .val_handle = &gEmgValHandle,
        .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
    },
    {
        .uuid = &kImuCharUuid.u,
        .access_cb = &CharAccess,
        .val_handle = &gImuValHandle,
        .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
    },
    {
        .uuid = &kPiezoCharUuid.u,
        .access_cb = &CharAccess,
        .val_handle = &gPiezoValHandle,
        .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
    },
    {
        .uuid = NULL,  // Array is null-terminated.
    }

};

// The definition for the BLE service.
static const struct ble_gatt_svc_def kServices[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &kServiceUuid.u,
        .characteristics = kServiceChars,
    },
    {
        .type = BLE_GATT_SVC_TYPE_END,
    },
};

void SensorSubscribe(struct ble_gap_event* event) {
  const uint16_t attr_handle = event->subscribe.attr_handle;
  const uint16_t conn_handle = event->subscribe.conn_handle;

  if (conn_handle != BLE_HS_CONN_HANDLE_NONE) {
    ESP_LOGI(kLimbTag, "subscribe event; conn_handle=%d attr_handle=%d",
             conn_handle, attr_handle);
  } else {
    ESP_LOGI(kLimbTag, "subscribe by nimble stack; attr_handle=%d",
             attr_handle);
  }

  // NOTE(johan): I don't know if it's guaranteed to always have the same value.
  // It's not used for anything critical, so it doesn't really matter.
  enum { kServiceChangedAttributeHandle = 8 };

  const bool cur_notify = event->subscribe.cur_notify;
  const char* notify_status = cur_notify ? "subscribed" : "unsubscribed";

  if (attr_handle == gEmgValHandle) {
    gEmgSubscriptionHandle = conn_handle;
    gEmgPeerNotifyEnabled = cur_notify;
    ESP_LOGI(kLimbTag, "EMG characteristic %s!", notify_status);
  } else if (attr_handle == gImuValHandle) {
    gImuSubscriptionHandle = conn_handle;
    gImuPeerNotifyEnabled = cur_notify;
    ESP_LOGI(kLimbTag, "IMU characteristic %s!", notify_status);
  } else if (attr_handle == gPiezoValHandle) {
    gPiezoSubscriptionHandle = conn_handle;
    gPiezoPeerNotifyEnabled = cur_notify;
    ESP_LOGI(kLimbTag, "Piezo characteristic %s!", notify_status);
  } else if (attr_handle == kServiceChangedAttributeHandle) {
    // This indication is used to tell bonded clients if the service has
    // changed between connections.
    const char* sub_status =
        event->subscribe.cur_indicate ? "subscribed" : "unsubscribed";
    ESP_LOGI(kLimbTag, "Service Changed characteristic %s.", sub_status,
             attr_handle);
  } else {
    ESP_LOGW(kLimbTag, "Unknown subscription attribute handle [%d]",
             attr_handle);
  }
}

const struct ble_gatt_svc_def* get_sensor_services(void) { return kServices; }
