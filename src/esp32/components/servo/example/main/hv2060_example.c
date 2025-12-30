#include "adc_manager.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"
#include "hal/adc_types.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "potentiometer.h"
#include "servo.h"

#define LIMB_ARR_LEN(arr) (sizeof(arr) / sizeof(*(arr)))

enum {
  HV2060_MIN_PULSEWIDTH_US = 850,
  HV2060_MAX_PULSEWIDTH_US = 2150,
  HV2060_MID_PULSEWIDTH_US =
      ((HV2060_MAX_PULSEWIDTH_US - HV2060_MIN_PULSEWIDTH_US) / 2) +
      HV2060_MIN_PULSEWIDTH_US,

  // NOTE: These should be measured at the maximum and minimum extents of
  // the motor actuations you want to support. These current values aren't
  // accurate.
  HV2060_MIN_POTENTIOMETER_ANGLE = 20,
  HV2060_MAX_POTENTIOMETER_ANGLE = 70,
  HV2060_MID_POTENTIOMETER_ANGLE =
      ((HV2060_MAX_POTENTIOMETER_ANGLE - HV2060_MIN_POTENTIOMETER_ANGLE) / 2) +
      HV2060_MIN_POTENTIOMETER_ANGLE,
  HV2060_POTENTIOMETER_ANGLE_RANGE =
      HV2060_MAX_POTENTIOMETER_ANGLE - HV2060_MIN_POTENTIOMETER_ANGLE,

  SERVO_POT_ADC_CHANNEL = ADC_CHANNEL_0,
};

const ServoConfig servo_config = {
    .gpio_pin = GPIO_NUM_1,
    .direction = SERVO_DIR_NORMAL,
    .ledc_channel = LEDC_CHANNEL_0,
    .ledc_timer = LEDC_TIMER_0,
    // FIXME: This value assumes 7.4V, but it was measured with 7V.
    .max_capable_angular_velocity = {400.F},
    // FIXME: The measured value I got here was 150. We cheat a bit to make sure
    // we don't get stuck.
    .max_capable_angular_velocity_pw_offset = 800,
    .min_capable_angular_velocity = {34.F},
    .min_capable_angular_velocity_pw_offset = 17,
    .motionless_pw = 1500,
    .max_angular_velocity = {60.F},
    .max_angular_acceleration = {200.F},
    .pot_adc_channel = SERVO_POT_ADC_CHANNEL,
    .potentiometer =
        (Potentiometer){
            .degrees_of_motion = {285.F},
            // FIXME: These need to be calibrated.
            .min_adc_value = 6,
            .max_adc_value = 3087,
            .min_potentiometer_angle = {HV2060_MIN_POTENTIOMETER_ANGLE},
            .max_potentiometer_angle = {HV2060_MAX_POTENTIOMETER_ANGLE},
            .min_potentiometer_angle_as_joint_angle = {0.F},
            .joint_angle_to_potentiometer_angle_ratio = 1.F,
        },
    .name = "hv2060",
};

const AdcMgrChannelConfig adc_channel_config = {
    .channel = SERVO_POT_ADC_CHANNEL,
    .sample_rate = 1000,
};

const AdcMgrConfig adc_cfg = {
    .channel_configs = &adc_channel_config,
    .channel_configs_len = 1,
    .ms_worth_of_buffer_size = 100,
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
uint16_t s_servo_potentiometer_underlying_buf[1024] = {0};
AdcMgrReadResults s_adc_read_results = {
    .channel_buffers =
        {
            [SERVO_POT_ADC_CHANNEL] =
                {
                    .data = s_servo_potentiometer_underlying_buf,
                    .capacity =
                        LIMB_ARR_LEN(s_servo_potentiometer_underlying_buf),
                },
        },
};

AdcMgrChannelBuffer *s_servo_potentiometer_adc_channel_buffer =
    &s_adc_read_results.channel_buffers[SERVO_POT_ADC_CHANNEL];
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

void app_main(void) {
  ESP_ERROR_CHECK(adc_mgr_init(adc_cfg));
  ESP_ERROR_CHECK(adc_mgr_read(&s_adc_read_results, 10));
  uint16_t latest_potentiometer_adc_value =
      limb_average16(s_servo_potentiometer_adc_channel_buffer->data,
                     s_servo_potentiometer_adc_channel_buffer->length);
  s_servo_potentiometer_adc_channel_buffer->length = 0;

  ServoHandle servo = 0;
  ESP_ERROR_CHECK(
      servo_init(&servo_config, latest_potentiometer_adc_value, &servo));

  JointAngle stops[] = {
      {0},
      {50},
  };

  int i = 0;
  servo_set_target_angle(servo, stops[i]);

  TickType_t current_tick = xTaskGetTickCount();
  ESP_LOGI("test", "tick: %d", current_tick);

  // {
  //   servo_move_to_pulse_width(servo, HV2060_MIN_PULSEWIDTH_US);
  //   adc_mgr_read(&s_adc_read_results, 0);
  //   uint16_t latest =
  //       s_servo_potentiometer_adc_channel_buffer
  //           ->data[s_servo_potentiometer_adc_channel_buffer->length - 1];
  //   s_servo_potentiometer_adc_channel_buffer->length = 0;
  //   PotentiometerAngle angle =
  //       potentiometer_adc_to_angle(&servo_config.potentiometer, latest);
  //   ESP_LOGI("mid servo angle", "adc: %u, angle: %.2f", latest,
  //   angle.degree);
  // }

  while (true) {
    // vTaskDelay(pdMS_TO_TICKS(100));
    // adc_mgr_read(&s_adc_read_results, 0);
    // uint32_t average = 0;
    // for (int i = 0; i < s_servo_potentiometer_adc_channel_buffer->length;
    // ++i) {
    //   average += s_servo_potentiometer_adc_channel_buffer->data[i];
    // }
    // average /= s_servo_potentiometer_adc_channel_buffer->length;
    // s_servo_potentiometer_adc_channel_buffer->length = 0;
    // PotentiometerAngle angle =
    //     potentiometer_adc_to_angle(&servo_config.potentiometer, average);
    // ESP_LOGI("current servo angle", "adc: %u, angle: %.2f", average,
    //          angle.degree);

    const uint16_t period_in_ms = 100;
    xTaskDelayUntil(&current_tick, pdMS_TO_TICKS(period_in_ms));
    ESP_ERROR_CHECK(adc_mgr_read(&s_adc_read_results, 0));

    latest_potentiometer_adc_value =
        moving_average16(latest_potentiometer_adc_value,
                         s_servo_potentiometer_adc_channel_buffer->data,
                         s_servo_potentiometer_adc_channel_buffer->length);
    s_servo_potentiometer_adc_channel_buffer->length = 0;
    bool done =
        servo_update(servo, period_in_ms, latest_potentiometer_adc_value);
    // ESP_LOGI("test", "ADC value %u",
    //          average);
    if (done) {
      i = (i + 1) % LIMB_ARR_LEN(stops);
      ESP_LOGI("test", "switching to joint angle %.2f", stops[i].degree);
      vTaskDelay(pdMS_TO_TICKS(1000));
      servo_set_target_angle(servo, stops[i]);
    }
    vTaskDelay(pdMS_TO_TICKS(100));

    // {
    //   servo_move_to_pulse_width(servo, HV2060_MID_PULSEWIDTH_US - 207);
    //   vTaskDelay(pdMS_TO_TICKS(5000));
    //   adc_mgr_read(&s_adc_read_results, 0);
    //   uint16_t latest =
    //       s_servo_potentiometer_adc_channel_buffer
    //           ->data[s_servo_potentiometer_adc_channel_buffer->length - 1];
    //   s_servo_potentiometer_adc_channel_buffer->length = 0;
    //   PotentiometerAngle angle =
    //       potentiometer_adc_to_angle(&servo_config.potentiometer, latest);
    //   ESP_LOGI("min servo angle", "adc: %u, angle: %.2f", latest,
    //   angle.degree);
    // }

    // {
    //   servo_move_to_pulse_width(servo, HV2060_MAX_PULSEWIDTH_US);
    //   vTaskDelay(pdMS_TO_TICKS(5000));
    //   adc_mgr_read(&s_adc_read_results, 0);
    //   uint16_t latest =
    //       s_servo_potentiometer_adc_channel_buffer
    //           ->data[s_servo_potentiometer_adc_channel_buffer->length - 1];
    //   s_servo_potentiometer_adc_channel_buffer->length = 0;
    //   PotentiometerAngle angle =
    //       potentiometer_adc_to_angle(&servo_config.potentiometer, latest);
    //   ESP_LOGI("max servo angle", "adc: %u, angle: %.2f", latest,
    //   angle.degree);
    // }

    // {
    //   servo_move_to_pulse_width(servo, HV2060_MID_PULSEWIDTH_US);
    //   vTaskDelay(pdMS_TO_TICKS(5000));
    //   adc_mgr_read(&s_adc_read_results, 0);
    //   uint16_t latest =
    //       s_servo_potentiometer_adc_channel_buffer
    //           ->data[s_servo_potentiometer_adc_channel_buffer->length - 1];
    //   s_servo_potentiometer_adc_channel_buffer->length = 0;
    //   PotentiometerAngle angle =
    //       potentiometer_adc_to_angle(&servo_config.potentiometer, latest);
    //   ESP_LOGI("mid servo angle", "adc: %u, angle: %.2f", latest,
    //   angle.degree);
    // }

    // {
    //   servo_move_to_degree(servo, (PotentiometerAngle){60});
    //   vTaskDelay(pdMS_TO_TICKS(2000));
    //   adc_mgr_read(&s_adc_read_results, 0);
    //   uint16_t latest =
    //       s_servo_potentiometer_adc_channel_buffer
    //           ->data[s_servo_potentiometer_adc_channel_buffer->length - 1];
    //   s_servo_potentiometer_adc_channel_buffer->length = 0;
    //   PotentiometerAngle angle =
    //       potentiometer_adc_to_angle(&servo_config.potentiometer, latest);
    //   ESP_LOGI("low angle", "adc: %u, angle: %.2f", latest, angle.degree);
    // }

    // {
    //   servo_move_to_degree(servo, (PotentiometerAngle){150});
    //   vTaskDelay(pdMS_TO_TICKS(2000));
    //   adc_mgr_read(&s_adc_read_results, 0);
    //   uint16_t latest =
    //       s_servo_potentiometer_adc_channel_buffer
    //           ->data[s_servo_potentiometer_adc_channel_buffer->length - 1];
    //   s_servo_potentiometer_adc_channel_buffer->length = 0;
    //   PotentiometerAngle angle =
    //       potentiometer_adc_to_angle(&servo_config.potentiometer, latest);
    //   ESP_LOGI("hi angle", "adc: %u, angle: %.2f", latest, angle.degree);
    // }
  }
}
