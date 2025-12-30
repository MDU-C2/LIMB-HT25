#include "servo.h"

#include "driver/ledc.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "motor_ramping.h"
#include "portmacro.h"
#include "potentiometer.h"

static const char *const TAG = "Servo";

typedef struct {
  portMUX_TYPE spinlock;
  ServoConfig cfg;
  AngularVelocity current_angular_velocity;
  AngularVelocity min_angular_velocity;
  PotentiometerAngle target_angle;
  PotentiometerAngle current_angle;

  bool is_moving;

  uint32_t latest_approximated_adc_value;
} ServoContext;

enum {
  SERVO_MAX_DUTY = ((1U << LEDC_TIMER_13_BIT) - 1),
  // We assume that the frequency used by the servo is 50 Hz.
  SERVO_FREQUENCY = 50,
  SERVO_PERIOD_US = 1000000UL / SERVO_FREQUENCY,

  HV2060_INTERNAL_POTENTIOMETER_DEGREES_OF_MOTION = 220,
};

#define ALPHA 0.1F
#define DEADBAND_DEG 1.5F

// We support a static amount of servo motors, so we statically allocate space
// for them.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static bool s_channels_assigned[LEDC_CHANNEL_MAX] = {false};
static ServoContext s_servo_contexts[LEDC_CHANNEL_MAX] = {0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// Convert microseconds to duty cycle
static uint32_t us_to_duty(const ServoConfig *servo, uint16_t us) {
  us = LIMB_CLAMP(us, servo->min_pulse_us, servo->max_pulse_us);

  return (uint32_t)((uint64_t)SERVO_MAX_DUTY * us / SERVO_PERIOD_US);
}

esp_err_t servo_init(const ServoConfig *servo_config,
                     const uint16_t *latest_potentiometer_values,
                     uint16_t latest_potentiometer_values_len,
                     ServoHandle *out_handle) {
  // Configure LEDC timer (shared by all servos).
  ledc_timer_config_t ledc_timer = {
      .speed_mode = LEDC_LOW_SPEED_MODE,
      .duty_resolution = LEDC_TIMER_13_BIT,
      .timer_num = LEDC_TIMER_0,
      .freq_hz = SERVO_FREQUENCY,
      .clk_cfg = LEDC_AUTO_CLK,
  };

  ESP_RETURN_ON_ERROR(ledc_timer_config(&ledc_timer), TAG,
                      "Couldn't configure ledc_timer");
  ESP_LOGI(TAG, "Timer configured");

  ESP_LOGI(TAG, "Configuring %s on GPIO%d, Channel %d", servo_config->name,
           servo_config->gpio_pin, servo_config->ledc_channel);

  if (s_channels_assigned[servo_config->ledc_channel]) {
    ESP_LOGE(
        TAG,
        "Configuring multiple servos using the same channel in servos_init!");
    return ESP_ERR_INVALID_ARG;
  }
  s_channels_assigned[servo_config->ledc_channel] = true;

  ledc_channel_config_t channel_config = {
      .gpio_num = servo_config->gpio_pin,
      .speed_mode = LEDC_LOW_SPEED_MODE,
      .channel = servo_config->ledc_channel,
      .intr_type = LEDC_INTR_DISABLE,
      .timer_sel = LEDC_TIMER_0,
      .duty = us_to_duty(servo_config, servo_config->min_pulse_us),
  };

  ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_config), TAG,
                      "Couldn't configure ledc_channel");

  uint16_t current_pot_adc_value = limb_average16(
      latest_potentiometer_values, latest_potentiometer_values_len);
  PotentiometerAngle current_angle = potentiometer_adc_to_angle(
      &servo_config->potentiometer, current_pot_adc_value);

  s_servo_contexts[servo_config->ledc_channel] = (ServoContext){
      .cfg = *servo_config,
      .current_angle = current_angle,
      .target_angle = servo_config->initial_angle,
      .latest_approximated_adc_value = current_pot_adc_value,
  };

  *out_handle = servo_config->ledc_channel;

  ESP_LOGI(TAG, "Initializing servo angle to %.2f degrees.",
           current_angle.degree);

  // TODO(johan): Remove once we have an update function.
  servo_move_to_degree(*out_handle, servo_config->initial_angle);

  ESP_LOGI(TAG, "All channels configured");

  return ESP_OK;
}

static uint32_t angle_to_pulse_width(PotentiometerAngle deg,
                                     const ServoConfig *servo) {
  if (servo->direction == SERVO_DIR_REVERSE) {
    deg.degree =
        servo->max_angle.degree - (deg.degree - servo->min_angle.degree);
  }
  return LIMB_LERP_FROM_RANGE(deg.degree, servo->min_angle.degree,
                              servo->max_angle.degree, servo->min_pulse_us,
                              servo->max_pulse_us);
}

static ServoContext *servo_get_context(ServoHandle handle) {
  return &s_servo_contexts[handle];
}

static PotentiometerAngle clamp_servo_angle(const ServoConfig *cfg,
                                            PotentiometerAngle angle) {
  return (PotentiometerAngle){
      LIMB_CLAMP(angle.degree, cfg->min_angle.degree, cfg->max_angle.degree)};
}

void servo_set_target_angle(ServoHandle handle, JointAngle target_angle) {
  ServoContext *context = servo_get_context(handle);
  PotentiometerAngle target_potentiometer_angle =
      to_potentiometer_angle(&context->cfg.potentiometer, target_angle);

  target_potentiometer_angle.degree =
      LIMB_CLAMP(target_potentiometer_angle.degree,
                 context->cfg.min_angle.degree, context->cfg.max_angle.degree);

  ESP_LOGI(TAG, "setting target pot angle: %.2f",
           target_potentiometer_angle.degree);
  portENTER_CRITICAL(&context->spinlock);
  context->target_angle = target_potentiometer_angle;
  portEXIT_CRITICAL(&context->spinlock);
}

void stop_motor(ServoHandle handle) {
  ServoContext *ctx = servo_get_context(handle);
  // TODO(johan): Check how sudden this stop is.
  // FIXME: This will move the target angle to the current target angle, which
  // might be off from the original intended target angle. This might(?) result
  // in drift?
  portENTER_CRITICAL(&ctx->spinlock);
  ctx->current_angular_velocity.dps = 0.F;
  portEXIT_CRITICAL(&ctx->spinlock);
}

static void apply_motor_velocity(ServoHandle handle, float velocity_dps,
                                 uint16_t actuation_time_in_ms) {
  const ServoContext *ctx = servo_get_context(handle);

  const float degrees_delta =
      velocity_dps * (float)actuation_time_in_ms / 1000.F;
  // FIXME: Figure out if this is even needed once we have the final version
  //
  // We need to move at least one step to prevent becoming stuck.
  float min_degree_delta =
      ctx->cfg.min_angle.degree -
      LIMB_LERP_FROM_RANGE(ctx->cfg.min_pulse_us + 1, ctx->cfg.min_pulse_us,
                           ctx->cfg.max_pulse_us, ctx->cfg.min_angle.degree,
                           ctx->cfg.max_angle.degree);

  if (degrees_delta < 0.F) {
    min_degree_delta = -min_degree_delta;
  }

  float new_angle = ctx->current_angle.degree + degrees_delta;
  ESP_LOGI(TAG,
           "curr: %.2f, delta: %.2f, mindelta: %.2f new: %.2f, target: %.2f",
           ctx->current_angle.degree, degrees_delta, min_degree_delta,
           new_angle, ctx->target_angle.degree);

  servo_move_to_degree(handle, (PotentiometerAngle){new_angle});
}

bool servo_update(ServoHandle handle, uint16_t ms_until_next_period,
                  const uint16_t *potentiometer_values,
                  uint16_t potentiometer_values_len) {
  // TODO(johan): Should this work without the latest potentiometer position?
  if (ms_until_next_period == 0 || potentiometer_values_len == 0) {
    return false;
  }

  ServoContext *ctx = servo_get_context(handle);

  ctx->latest_approximated_adc_value =
      moving_average16(ctx->latest_approximated_adc_value, potentiometer_values,
                       potentiometer_values_len);
  const PotentiometerAngle current_angle = potentiometer_adc_to_angle(
      &ctx->cfg.potentiometer, ctx->latest_approximated_adc_value);

  MotorRampingArgs args = {
      .current_angle = current_angle,
      .target_angle = ctx->target_angle,
      .deadband = (PotentiometerAngle){DEADBAND_DEG},
      .current_velocity = ctx->current_angular_velocity,
      .max_acceleration = ctx->cfg.max_angular_acceleration,
      .max_velocity = ctx->cfg.max_angular_velocity,
      .timestep_ms = ms_until_next_period,
  };
  const AngularVelocity new_velocity = motor_ramping_trapezoidal(&args);

  // Update the current state.
  portENTER_CRITICAL(&ctx->spinlock);
  ctx->current_angle = current_angle;
  ctx->current_angular_velocity = new_velocity;
  portEXIT_CRITICAL(&ctx->spinlock);

  // NOTE: Normally checking equality of floats is imprecise, but in this case
  // the literal value 0.F gets returned when within the deadband, so it should
  // be fine checking against the same literal.
  if (new_velocity.dps == 0.F) {
    ESP_LOGI(TAG, "STOPPING!");
    ESP_LOGI(TAG,
             "Update: target=%.2f°, current=%.2f°, error=%.2f°, vel=%.1f dps",
             args.target_angle.degree, ctx->current_angle.degree,
             args.target_angle.degree - ctx->current_angle.degree,
             ctx->current_angular_velocity);
    stop_motor(handle);
    return true;
  }

  // TODO(johan): Check if we want a minimum velocity.
  // Clamp to minimum velocity if moving
  // new_velocity.dps = MAX(new_velocity.dps, ctx->min_angular_velocity.dps);

  apply_motor_velocity(handle, new_velocity.dps, ms_until_next_period);

  static uint32_t log_counter = 0;
  if (++log_counter >= 10) {  // Log every 100 updates
    log_counter = 0;
    ESP_LOGI(TAG,
             "Update: target=%.2f°, current=%.2f°, error=%.2f°, vel=%.1f dps",
             args.target_angle.degree, ctx->current_angle.degree,
             args.target_angle.degree - ctx->current_angle.degree,
             ctx->current_angular_velocity);
  }

  return false;
}

void servo_move_to_pulse_width(ServoHandle handle, uint16_t pulse_width) {
  ServoContext *ctx = servo_get_context(handle);
  uint32_t duty = us_to_duty(&ctx->cfg, pulse_width);

  ESP_LOGI(TAG, "pw: %u, duty: %u", pulse_width, duty);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.ledc_channel, duty);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.ledc_channel);
}

// Write angle to specific servo channel
void servo_move_to_degree(ServoHandle handle, PotentiometerAngle deg) {
  ServoContext *ctx = servo_get_context(handle);

  deg = clamp_servo_angle(&ctx->cfg, deg);

  uint32_t us = angle_to_pulse_width(deg, &ctx->cfg);

  servo_move_to_pulse_width(handle, us);

  static int i = 0;
  if (++i > 0) {
    i = 0;
    uint32_t duty = us_to_duty(&ctx->cfg, us);
    ESP_LOGI(TAG, "%s -> %.2f° (%lu us, %u)", ctx->cfg.name, deg.degree, us,
             duty);
  }
}
