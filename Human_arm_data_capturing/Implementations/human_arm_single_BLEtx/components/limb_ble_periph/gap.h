#pragma once

#include "host/ble_hs.h"

// Callback used when NimBLE is started.
ble_hs_sync_fn BleStackSyncCallback;
// Callback used when NimBLE is reset.
ble_hs_reset_fn BleStackResetCallback;
