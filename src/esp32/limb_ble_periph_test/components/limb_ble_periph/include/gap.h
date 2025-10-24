#pragma once

#include "host/ble_hs.h"

enum {
  kMaxLeDataLength = 251,
  kL2CapHeaderSize = 4,
  kMtuSize = kMaxLeDataLength - kL2CapHeaderSize,
  kAttHeaderSize = 3,
  // This is the max amount of sensor data we can send in one packet.
  kMaxAttDataSize = kMtuSize - kAttHeaderSize,
};

// Callback used when NimBLE is started.
ble_hs_sync_fn BleStackSyncCallback;
// Callback used when NimBLE is reset.
ble_hs_reset_fn BleStackResetCallback;
