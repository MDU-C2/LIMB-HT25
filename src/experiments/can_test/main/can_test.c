// CAN loopback test program on a single ESP32-C3-Zero, shorting the RX and TX
// pins while doing a loopback. The setup follows the general directions in:
// https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-reference/peripherals/twai.html

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_err.h"
#include "esp_system.h"
#include "esp_twai.h"
#include "esp_twai_onchip.h"
#include "esp_twai_types.h"
// FreeRTOS.h must be included before task.h.
// NOLINTNEXTLINE(misc-header-include-cycle,misc-include-cleaner)
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"
#include "freertos/task.h"
#include "hal/twai_types.h"

// General constants.
enum {
  kMaxCanMsgLen = 8,
  kSecondsInMs = 1000,
};

// Callback context for getting information back from the callbacks.
// Since they are called during interrupts, things like IO can't be performed
// directly in the callbacks.
typedef struct {
  struct {
    enum { kMaxCallbackMsgLen = 256 };
    char msg[kMaxCallbackMsgLen];
    size_t msg_len;
    esp_err_t error;
    int invocation_count;
  } rx;
  struct {
    enum { kMaxErrFlagsLen = 128 };
    twai_error_flags_t flags[kMaxErrFlagsLen];
    int invocation_count;
  } err;
  struct {
    int invocation_count;
  } tx;
} CallbackContext;

static bool TwaiRxCallback(
    twai_node_handle_t handle,
    [[maybe_unused]] const twai_rx_done_event_data_t *edata, void *user_ctx) {
  CallbackContext *ctx = user_ctx;
  ++ctx->rx.invocation_count;

  uint8_t recv_buff[kMaxCanMsgLen];
  twai_frame_t rx_frame = {
      .buffer = recv_buff,
      .buffer_len = sizeof(recv_buff),
  };

  esp_err_t ret = twai_node_receive_from_isr(handle, &rx_frame);
  ctx->rx.error = ret;
  if (ret == ESP_OK) {
    strncpy(ctx->rx.msg, (char *)rx_frame.buffer, rx_frame.buffer_len);
    ctx->rx.msg_len = rx_frame.buffer_len;
  }

  return false;
}

static bool TwaiErrCallback([[maybe_unused]] twai_node_handle_t handle,
                            const twai_error_event_data_t *edata,
                            void *user_ctx) {
  CallbackContext *ctx = user_ctx;
  if (ctx->err.invocation_count < kMaxErrFlagsLen) {
    ctx->err.flags[ctx->err.invocation_count] = edata->err_flags;
  }
  ++ctx->err.invocation_count;

  return false;
}

static bool TwaiTxCallback(
    [[maybe_unused]] twai_node_handle_t handle,
    [[maybe_unused]] const twai_tx_done_event_data_t *edata, void *user_ctx) {
  CallbackContext *ctx = user_ctx;
  ++ctx->tx.invocation_count;
  return false;
}
void print_callback_errors(CallbackContext *user_ctx) {
#define min(x, y) ((x) < (y) ? (x) : (y))
  for (int i = 0; i < min(user_ctx->err.invocation_count, kMaxErrFlagsLen);
       ++i) {
    printf("Err %d\n", i);
    if (user_ctx->err.flags[i].ack_err) {
      printf("Err: Missing ACK\n");
    }
    if (user_ctx->err.flags[i].arb_lost) {
      printf("Err: Lost arbitration\n");
    }
    if (user_ctx->err.flags[i].bit_err) {
      printf("Err: Bit mismatch\n");
    }
    if (user_ctx->err.flags[i].form_err) {
      printf("Err: Form violation\n");
    }
    if (user_ctx->err.flags[i].stuff_err) {
      printf("Err: Stuff error\n");
    }
  }

  if (user_ctx->rx.error != 0) {
    printf("Error during RX, errcode: %d (%x)\n", user_ctx->rx.error,
           user_ctx->rx.error);
  }
}

void app_main(void) {
  twai_node_handle_t node_hdl = NULL;

  enum {
    kRxGpioPin = 4,
    kTxGpioPin = 5,
    kCanBaud = 200000,
    kTxQueueDepth = 5,
  };
  twai_onchip_node_config_t node_config = {
      .io_cfg.tx = kTxGpioPin,
      .io_cfg.rx = kRxGpioPin,
      .bit_timing.bitrate = kCanBaud,
      .tx_queue_depth = kTxQueueDepth,
      // This test is currently assuming we're doing a loopback on a single
      // device while shorting the RX and TX pins. That means we need to enable
      // both of the following flags. If we switch to using this program between
      // two different controllers, the two flags need to be turned off.

      // During the test we're shorting the TX and RX pins while doing a
      // loopback.
      .flags.enable_loopback = true,
      // Since we're doing a loopback test, we need to assume we're not getting
      // ACKs.
      .flags.enable_self_test = true,
  };
  // Create a new TWAI controller driver instance.
  ESP_ERROR_CHECK(twai_new_node_onchip(&node_config, &node_hdl));

  twai_event_callbacks_t user_cbs = {
      .on_rx_done = &TwaiRxCallback,
      .on_error = &TwaiErrCallback,
      .on_tx_done = &TwaiTxCallback,
  };
  CallbackContext user_ctx = {0};
  ESP_ERROR_CHECK(twai_node_register_event_callbacks(node_hdl, &user_cbs,
                                                     (void *)&user_ctx));

  // Start the TWAI controller.
  ESP_ERROR_CHECK(twai_node_enable(node_hdl));

  printf("Ready to do CAN IO!\n");

  enum { kSecondsBeforeRestart = 10 };
  for (int i = kSecondsBeforeRestart; i >= 0; i--) {
    uint8_t send_buf[kMaxCanMsgLen] = {0};
    int chars_written = sprintf((char *)send_buf, "Hello%d", i);
    if (chars_written < 0 || chars_written >= sizeof(send_buf)) {
      (void)fprintf(
          stderr,
          "Error writing to string, wrote [%d] characters to buffer of "
          "size %zu",
          chars_written, sizeof(send_buf));
    }

    twai_frame_t tx_msg = {
        .header.id = 0x1,
        .buffer = send_buf,
        .buffer_len = sizeof(send_buf),
    };

    // Timeout = 0: returns immediately if queue is full.
    enum { kTimeoutMs = 0 };
    esp_err_t ret = twai_node_transmit(node_hdl, &tx_msg, kTimeoutMs);
    if (ret == ESP_OK) {
      printf("Successful TX!\n");
    } else {
      printf("Error during TX, errcode: %d (%x)\n", ret, ret);
    }

    print_callback_errors(&user_ctx);

    printf("Received message: \"%.*s\"\n", user_ctx.rx.msg_len,
           user_ctx.rx.msg);
    printf("Times entered callbacks: RX(%d) TX(%d) ERR(%d)\n",
           user_ctx.rx.invocation_count, user_ctx.tx.invocation_count,
           user_ctx.err.invocation_count);
    printf("Restarting in %d seconds...\n", i);
    (void)fflush(stdout);
    vTaskDelay(kSecondsInMs / portTICK_PERIOD_MS);
  }

  // Cleanup.
  ESP_ERROR_CHECK(twai_node_disable(node_hdl));
  ESP_ERROR_CHECK(twai_node_delete(node_hdl));

  printf("Restarting now.\n");
  (void)fflush(stdout);
  esp_restart();
}
