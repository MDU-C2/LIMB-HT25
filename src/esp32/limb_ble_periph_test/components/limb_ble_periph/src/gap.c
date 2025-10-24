#include "gap.h"

#include "host/util/util.h"
#include "sensors_service.h"
#include "services/gap/ble_svc_gap.h"

static void start_advertising(void);

static const char *const kGapTag = "LIMB GAP";

// Constants.
enum {
  kBleGapLeRolePeripheral = 0x00,
  kBleGapAppearanceGenericSensor = 0x0540,
  kBleAddressStrMaxLen = 18,
  kBleAddressValMaxLen = 6,
};

// Represents a BLE address of some type (public or random).
typedef struct {
  uint8_t type;
  uint8_t val[kBleAddressValMaxLen];
} Address;

// We cache the address during BT stack sync.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static Address gOwnAddress = {0};

// Writes out the string representation of the provided address to
// [out_addr_str].
static void format_addr(char *out_addr_str, uint8_t *addr) {
  int bytes_written =
      sprintf(out_addr_str, "%02X:%02X:%02X:%02X:%02X:%02X", addr[0], addr[1],
              addr[2], addr[3], addr[4], addr[5]);  // NOLINT(*-magic-numbers)
  assert(bytes_written == 17 && "Couldn't write address as string.");
}

// Prints the provided connection description.
static void print_conn_desc(struct ble_gap_conn_desc *desc) {
  char addr_str[kBleAddressStrMaxLen] = {0};

  ESP_LOGI(kGapTag, "connection handle: %d", desc->conn_handle);

  format_addr(addr_str, desc->our_id_addr.val);
  ESP_LOGI(kGapTag, "device id address: type=%d, value=%s",
           desc->our_id_addr.type, addr_str);

  format_addr(addr_str, desc->peer_id_addr.val);
  ESP_LOGI(kGapTag, "peer id address: type=%d, value=%s",
           desc->peer_id_addr.type, addr_str);

  ESP_LOGI(kGapTag,
           "conn_itvl=%d, conn_latency=%d, supervision_timeout=%d, "
           "encrypted=%d, authenticated=%d, bonded=%d\n",
           desc->conn_itvl, desc->conn_latency, desc->supervision_timeout,
           desc->sec_state.encrypted, desc->sec_state.authenticated,
           desc->sec_state.bonded);
}

void BleStackResetCallback(int reason) {
  /* On reset, print reset reason to console */
  ESP_LOGI(kGapTag, "nimble stack reset, reset reason: %d", reason);
}

static int GapEventHandler(struct ble_gap_event *event,
                           [[maybe_unused]] void *arg) {
  switch (event->type) {
    case BLE_GAP_EVENT_CONNECT: {
      int conn_err = event->connect.status;

      ESP_LOGI(kGapTag, "connection %s; err=%d",
               conn_err == 0 ? "established" : "failed", conn_err);

      // Connection failed, so we just go back to advertising.
      if (conn_err != 0) {
        start_advertising();
        return 0;
      }

      {
        enum {
          // FIXME: Figure out what the actual effect of changing this is.
          // kLlPacketTime = 0x4290,
          kLlPacketTime = 2120,
          // LL_PACKET_TIME = 2120,
        };
        int err = ble_hs_hci_util_set_data_len(event->connect.conn_handle,
                                               kMaxLeDataLength, kLlPacketTime);
        if (err != 0) {
          ESP_LOGE(kGapTag, "Set packet length failed; rc = %d", err);
        }
      }

      {
        int err = ble_att_set_preferred_mtu(kMtuSize);
        if (err != 0) {
          ESP_LOGE(kGapTag, "Failed to set preferred MTU; rc = %d", err);
        }
      }

      {
        // int err =
        //     ble_gattc_exchange_mtu(event->connect.conn_handle, NULL, NULL);
        // if (err != 0) {
        //   ESP_LOGE(kGapTag, "Failed to negotiate MTU; rc = %d", err);
        // }
      }

      // Get the description of the connection.
      uint16_t conn_handle = event->connect.conn_handle;
      struct ble_gap_conn_desc desc = {0};
      {
        int err = ble_gap_conn_find(conn_handle, &desc);
        if (err != 0) {
          ESP_LOGE(kGapTag,
                   "failed to find connection by handle, error code: %d", err);
          return err;
        }
      }

      print_conn_desc(&desc);

      // TODO(johan): Why do we need to update the params for the connection?
      struct ble_gap_upd_params params = {
          .itvl_min = desc.conn_itvl,
          .itvl_max = desc.conn_itvl,
          .latency = 3,
          .supervision_timeout = desc.supervision_timeout,
      };
      {
        int err = ble_gap_update_params(conn_handle, &params);
        if (err != 0) {
          ESP_LOGE(kGapTag,
                   "failed to update connection parameters, error code: %d",
                   err);
          return err;
        }
      }
      return 0;
    }

    case BLE_GAP_EVENT_DISCONNECT: {
      ESP_LOGI(kGapTag, "disconnected from peer; reason=%d",
               event->disconnect.reason);

      start_advertising();
      return 0;
    };

    case BLE_GAP_EVENT_CONN_UPDATE: {
      ESP_LOGI(kGapTag, "connection updated; status=%d",
               event->conn_update.status);

      struct ble_gap_conn_desc desc = {0};
      int err = ble_gap_conn_find(event->conn_update.conn_handle, &desc);
      if (err != 0) {
        ESP_LOGE(kGapTag, "failed to find connection by handle, error code: %d",
                 err);
        return err;
      }
      print_conn_desc(&desc);
      return 0;
    };

    case BLE_GAP_EVENT_ADV_COMPLETE: {
      ESP_LOGI(kGapTag, "advertise complete; reason=%d",
               event->adv_complete.reason);
      start_advertising();
      return 0;
    };

    case BLE_GAP_EVENT_NOTIFY_TX: {
      ESP_LOGI(kGapTag,
               "notify event; conn_handle=%d attr_handle=%d "
               "status=%d is_indication=%d",
               event->notify_tx.conn_handle, event->notify_tx.attr_handle,
               event->notify_tx.status, event->notify_tx.indication);
      return 0;
    };

    case BLE_GAP_EVENT_SUBSCRIBE: {
      ESP_LOGI(kGapTag,
               "subscribe event; conn_handle=%d attr_handle=%d "
               "reason=%d prevn=%d curn=%d previ=%d curi=%d",
               event->subscribe.conn_handle, event->subscribe.attr_handle,
               event->subscribe.reason, event->subscribe.prev_notify,
               event->subscribe.cur_notify, event->subscribe.prev_indicate,
               event->subscribe.cur_indicate);

      SensorSubscribe(event);
      return 0;
    };

    case BLE_GAP_EVENT_MTU: {
      ESP_LOGI(kGapTag, "mtu update event; conn_handle=%d cid=%d mtu=%d",
               event->mtu.conn_handle, event->mtu.channel_id, event->mtu.value);
      return 0;
    };

    default: {
      ESP_LOGW(kGapTag, "Ignoring unknown GAP event: [%d]", event->type);
      return 0;
    };
  }

  return 0;
}

// Sets up advertising parameters and starts the GAP advertising process.
static void start_advertising(void) {
  const char *device_name = ble_svc_gap_device_name();
  struct ble_hs_adv_fields adv_fields = {
      // LE General discoverable mode (advertise forever)
      // and Bluetooth basic rate/enhanced data rate ("classic" BT) not
      // supported.
      .flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP,

      .name = (uint8_t *)device_name,
      .name_len = strlen(device_name),
      .name_is_complete = 1,

      .appearance = kBleGapAppearanceGenericSensor,
      .appearance_is_present = 1,

      // We only support being a peripheral.
      .le_role = kBleGapLeRolePeripheral,
      .le_role_is_present = 1,
  };

  {
    int err = ble_gap_adv_set_fields(&adv_fields);
    if (err != 0) {
      ESP_LOGE(kGapTag, "failed to set advertising data, error code: %d", err);
      return;
    }
  }

  // Scan response packet.
  struct ble_hs_adv_fields rsp_fields = {
      .device_addr = gOwnAddress.val,
      .device_addr_type = gOwnAddress.type,
      .device_addr_is_present = 1,

      // scan response advertising interval.
      .adv_itvl = BLE_GAP_ADV_ITVL_MS(500),
      .adv_itvl_is_present = 1,
  };

  {
    int err = ble_gap_adv_rsp_set_fields(&rsp_fields);
    if (err != 0) {
      ESP_LOGE(kGapTag, "failed to set scan response data, error code: %d",
               err);
      return;
    }
  }

  // Set type of advertising we want.
  struct ble_gap_adv_params adv_params = {
      // Undirected connectable.
      .conn_mode = BLE_GAP_CONN_MODE_UND,
      // General discoverable.
      .disc_mode = BLE_GAP_DISC_MODE_GEN,

      // Actual advertising interval
      .itvl_min = BLE_GAP_ADV_ITVL_MS(500),
      .itvl_max = BLE_GAP_ADV_ITVL_MS(510),
  };

  {
    // When a GAP event is triggered, the provided event handler gets called.
    int err = ble_gap_adv_start(gOwnAddress.type, NULL, BLE_HS_FOREVER,
                                &adv_params, GapEventHandler, NULL);
    if (err != 0) {
      ESP_LOGE(kGapTag, "failed to start advertising, error code: %d", err);
      return;
    }
  }

  ESP_LOGI(kGapTag, "advertising started!");
}

void BleStackSyncCallback(void) {
  {
    // Make sure we have a BT address.
    int err = ble_hs_util_ensure_addr(0);
    if (err != 0) {
      ESP_LOGE(kGapTag, "device does not have any available bt address!");
      return;
    }
  }

  {
    // Figure out the type of BT address to use.
    int err = ble_hs_id_infer_auto(false, &gOwnAddress.type);
    if (err != 0) {
      ESP_LOGE(kGapTag, "failed to infer address type, error code: %d", err);
      return;
    }
  }

  {
    // Get the BT address we're using.
    int err = ble_hs_id_copy_addr(gOwnAddress.type, gOwnAddress.val, NULL);
    if (err != 0) {
      ESP_LOGE(kGapTag, "failed to copy device address, error code: %d", err);
      return;
    }
    char addr_str[kBleAddressStrMaxLen] = {0};
    format_addr(addr_str, gOwnAddress.val);
    ESP_LOGI(kGapTag, "device address: %s", addr_str);
  }

  start_advertising();
}
