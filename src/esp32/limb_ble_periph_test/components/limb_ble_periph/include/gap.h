#pragma once

#include "host/ble_hs.h"

enum {
  // Using DLE and 1M PHY, the max values for the length of the payload in an LL
  // Data PDU and time taken to transmit the packet are 251 octets and 2120 us
  // (Bluetooth Core Specification 5.3 Vol 4 Part E Table 4.6)
  kMaxLeDataTimeUs = 2120,
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
