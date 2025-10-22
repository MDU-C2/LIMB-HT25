#include "sensors_service.h"

#include "esp_log.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "services/gatt/ble_svc_gatt.h"

// It seems like the GATT definitions have to live at least until
// ble_gatts_start, which seems to get started in nimble_port_run, effectively
// meaning they might as well have static storage duration, so globals it is!

// Constants to determine the characteristic buffer sizes.
enum {
  // The amount of the new samples in a window that should be buffered before
  // being sent. E.g. 30 means one 30th of the new samples are buffered before
  // being sent.
  kPartOfWindowPerSend = 30,

  kEmgSamplesPerWindow = kEmgMsPerWindow * kEmgFrequency / 1000,
  kEmgSamplesPerOverlap = kEmgMsPerOverlap * kEmgFrequency / 1000,
  kEmgNewSamplesPerWindow = kEmgSamplesPerWindow - kEmgSamplesPerOverlap,
  kEmgSamplesToSend = kEmgNewSamplesPerWindow / kPartOfWindowPerSend,
  kEmgBufSize = kEmgSamplesToSend * kEmgBytesPerSample,
  kEmgBufInMs = 1000 * kEmgSamplesToSend / kEmgFrequency,

  kImuSamplesPerWindow = kImuMsPerWindow * kImuFrequency / 1000,
  kImuSamplesPerOverlap = kImuMsPerOverlap * kImuFrequency / 1000,
  kImuNewSamplesPerWindow = kImuSamplesPerWindow - kImuSamplesPerOverlap,
  kImuSamplesToSend = kImuNewSamplesPerWindow / kPartOfWindowPerSend,
  kImuBufSize = kImuSamplesToSend * kImuBytesPerSample,
  kImuBufInMs = 1000 * kImuSamplesToSend / kImuFrequency,

  kPiezoSamplesPerWindow = kPiezoMsPerWindow * kPiezoFrequency / 1000,
  kPiezoSamplesPerOverlap = kPiezoMsPerOverlap * kPiezoFrequency / 1000,
  kPiezoNewSamplesPerWindow = kPiezoSamplesPerWindow - kPiezoSamplesPerOverlap,
  kPiezoSamplesToSend = kPiezoNewSamplesPerWindow / kPartOfWindowPerSend,
  kPiezoBufSize = kPiezoSamplesToSend * kPiezoBytesPerSample,
  kPiezoBufInMs = 1000 * kPiezoSamplesToSend / kPiezoFrequency,
};

static const char* const kCharacteristicTag = "Char";

static const char* const kEmgLogTag = "EmgChar";

static const ble_uuid128_t kEmgCharUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x24);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static uint16_t gEmgSubscriptionHandle;
static bool gEmgPeerNotifyEnabled;
static uint16_t gEmgValHandle;
static uint8_t gEmgVal[kEmgBufSize] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

uint8_t* get_emg_buf(void) { return gEmgVal; }

bool TryNotifyEmgSubscribers(void) {
  if (gEmgPeerNotifyEnabled) {
    ble_gatts_notify(gEmgSubscriptionHandle, gEmgValHandle);
    ESP_LOGI(kEmgLogTag, "sensor notification sent!");
    return true;
  }
  return false;
}

// IMU characteristic.
static const char* const kImuLogTag = "ImuChar";

static const ble_uuid128_t kImuCharUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x25);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static uint16_t gImuSubscriptionHandle;
static bool gImuPeerNotifyEnabled;
static uint16_t gImuValHandle;
static uint8_t gImuVal[kImuBufSize] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

uint8_t* get_imu_buf(void) { return gImuVal; }

bool TryNotifyImuSubscribers(void) {
  if (gImuPeerNotifyEnabled) {
    ble_gatts_notify(gImuSubscriptionHandle, gImuValHandle);
    ESP_LOGI(kImuLogTag, "sensor notification sent!");
    return true;
  }
  return false;
}

// Piezo characteristic.
static const char* const kPiezoLogTag = "PiezoChar";

static const ble_uuid128_t kPiezoCharUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x26);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static uint16_t gPiezoSubscriptionHandle;
static bool gPiezoPeerNotifyEnabled;
static uint16_t gPiezoValHandle;
static uint8_t gPiezoVal[kPiezoBufSize] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

uint8_t* get_piezo_buf(void) { return gPiezoVal; }

bool TryNotifyPiezoSubscribers(void) {
  if (gPiezoPeerNotifyEnabled) {
    // TODO(johan): Figure out if this function is blocking. If it is, we can
    // probably get by without any synchronization.
    ble_gatts_notify(gPiezoSubscriptionHandle, gPiezoValHandle);
    ESP_LOGI(kPiezoLogTag, "sensor notification sent!");
    return true;
  }
  return false;
}

// Callback function for when a characteristic is accessed
static int CharAccess(uint16_t connection_handle, uint16_t attribute_handle,
                      struct ble_gatt_access_ctxt* context,
                      [[maybe_unused]] void* args) {
  if (context->op != BLE_GATT_ACCESS_OP_READ_CHR) {
    ESP_LOGW(kCharacteristicTag,
             "Unsupported access operation (non-read): [%d]", context->op);
    assert(false && "Unsupported characteristic access.");
    // FIXME: Use the proper return code.
    return BLE_ATT_ERR_UNLIKELY;
  }

  if (connection_handle != BLE_HS_CONN_HANDLE_NONE) {
    ESP_LOGI(kCharacteristicTag,
             "characteristic read; conn_handle=%d attr_handle=%d",
             connection_handle, attribute_handle);
  } else {
    ESP_LOGI(kCharacteristicTag,
             "characteristic read by nimble stack; attr_handle=%d",
             attribute_handle);
  }

  uint8_t* buffer = NULL;
  size_t buffer_size = 0;

  // Determine which characteristic we should send.
  if (attribute_handle == gEmgValHandle) {
    ESP_LOGI(kEmgLogTag, "EMG read", connection_handle, attribute_handle);
    buffer = gEmgVal;
    buffer_size = sizeof(gEmgVal);
  } else if (attribute_handle == gImuValHandle) {
    ESP_LOGI(kImuLogTag, "IMU read", connection_handle, attribute_handle);
    buffer = gImuVal;
    buffer_size = sizeof(gImuVal);
  } else if (attribute_handle == gPiezoValHandle) {
    ESP_LOGI(kPiezoLogTag, "Piezo read", connection_handle, attribute_handle);
    buffer = gPiezoVal;
    buffer_size = sizeof(gPiezoVal);
  } else {
    ESP_LOGW(kCharacteristicTag, "Char access with an invalid attribute_handle [%d].", attribute_handle);
    assert(false && "Char access with an invalid attribute_handle.");
    return BLE_ATT_ERR_INVALID_HANDLE;
  }

  int err = os_mbuf_append(context->om, buffer, buffer_size);
  if (err != 0) {
    ESP_LOGE(kCharacteristicTag,
             "Error appending characterictic value to os memory buffer. [%d]",
             err);
    assert(false &&
           "Error appending characterictic value to os memory buffer.");
    return BLE_ATT_ERR_INSUFFICIENT_RES;
  }

  return 0;
}

// Sensor service.
static const ble_uuid128_t kServiceUuid =
    BLE_UUID128_INIT(0x22, 0xd1, 0xbc, 0xea, 0x5f, 0x78, 0x23, 0x15, 0xde, 0xef,
                     0x12, 0x12, 0x25, 0x15, 0x01, 0x23);

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
  // FIXME: Change some of the EMG tags to some generic tag.
  if (event->subscribe.conn_handle != BLE_HS_CONN_HANDLE_NONE) {
    ESP_LOGI(kEmgLogTag, "subscribe event; conn_handle=%d attr_handle=%d",
             event->subscribe.conn_handle, event->subscribe.attr_handle);
  } else {
    ESP_LOGI(kEmgLogTag, "subscribe by nimble stack; attr_handle=%d",
             event->subscribe.attr_handle);
  }

  // NOTE(johan): I don't know if it's guaranteed to always have the same value.
  enum { kServiceChangedAttributeHandle = 8 };

  const char* notify_status =
      event->subscribe.cur_notify ? "subscribed" : "unsubscribed";
  if (event->subscribe.attr_handle == gEmgValHandle) {
    gEmgSubscriptionHandle = event->subscribe.conn_handle;
    gEmgPeerNotifyEnabled = event->subscribe.cur_notify;
    ESP_LOGI(kEmgLogTag, "Emg %s!", notify_status);
  } else if (event->subscribe.attr_handle == gImuValHandle) {
    gImuSubscriptionHandle = event->subscribe.conn_handle;
    gImuPeerNotifyEnabled = event->subscribe.cur_notify;
    ESP_LOGI(kImuLogTag, "Imu %s!", notify_status);
  } else if (event->subscribe.attr_handle == gPiezoValHandle) {
    gPiezoSubscriptionHandle = event->subscribe.conn_handle;
    gPiezoPeerNotifyEnabled = event->subscribe.cur_notify;
    ESP_LOGI(kPiezoLogTag, "Piezo %s!", notify_status);
  } else if (event->subscribe.attr_handle == kServiceChangedAttributeHandle) {
    // NOTE: This indication is used to tell bonded clients if the service has
    // changed between connections.
    // TODO(johan): We probably don't have to worry about it?
    const char* sub_status =
        event->subscribe.cur_indicate ? "subscribed" : "unsubscribed";
    ESP_LOGI(kEmgLogTag, "Service Changed characteristic %s.", sub_status,
             event->subscribe.attr_handle);
  } else {
    ESP_LOGW(kEmgLogTag, "Unknown subscription attribute handle [%d]",
             event->subscribe.attr_handle);
  }
}

const struct ble_gatt_svc_def* get_sensor_services(void) { return kServices; }
