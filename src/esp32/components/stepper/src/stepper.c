#include "stepper.h"

#include <math.h>

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal/ledc_types.h"
#include "limb_utils.h"
#include "motor_ramping.h"
#include "potentiometer.h"
#include "soc/gpio_num.h"
#include "soc/soc_caps.h"
#include "sys/param.h"

static const char *TAG = "stepper";

// Control constants
#define ALPHA 0.1f  // Low-pass filter coefficient (0.0-1.0)
// NOTE: Without microstepping, our step size is 1.8 degrees, so the deadband
// should probably be at least as large.
#define DEADBAND_DEG 3.6f  // Deadband in degrees (stop if error < this)
// The LEDC library doesn't like setting a frequency lower than 5
// (calling ledc_find_suitable_duty_resolution with lower frequencies returned
// 0).
#define MIN_FREQ_HZ 5  // Minimum LEDC frequency

// Control context
typedef struct {
  stepper_control_config_t cfg;
  portMUX_TYPE spinlock;

  // LEDC config
  uint32_t duty_50_percent;

  // Motion state
  bool estop_active;
  bool is_moving;
  AngularVelocity current_velocity;
  PotentiometerAngle target_angle;
  PotentiometerAngle current_angle;
  bool use_position_feedback;

  // Calculated parameters
  float steps_per_degree;

  bool is_initialized;
} motion_control_context_t;

// We only support at most as many steppers as there are LEDC channels, since
// they require exclusive access anyway.
static motion_control_context_t s_contexts[SOC_LEDC_CHANNEL_NUM] = {0};

static void stop_motor(stepper_control_handle_t handle) {
  motion_control_context_t *ctx = &s_contexts[handle];

  ledc_set_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.pwm_channel, 0);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.pwm_channel);

  portENTER_CRITICAL(&ctx->spinlock);
  ctx->is_moving = false;
  ctx->current_velocity = (AngularVelocity){0.0F};
  portEXIT_CRITICAL(&ctx->spinlock);
}

static void apply_motor_velocity(stepper_control_handle_t handle,
                                 AngularVelocity velocity) {
  motion_control_context_t *ctx = &s_contexts[handle];

  // Enable motor
  if (ctx->cfg.enable_gpio != GPIO_NUM_NC) {
    gpio_set_level(ctx->cfg.enable_gpio, 0);  // Enable (active low)
  }

  // Direction control
  if (ctx->cfg.dir_gpio != GPIO_NUM_NC) {
    uint8_t direction = velocity.dps < 0.0F ? 1 : 0;

    // Swap the direction if we're in reverse mode.
    if (ctx->cfg.direction == STEPPER_DIR_REVERSE) {
      direction = !direction;
    }

    gpio_set_level(ctx->cfg.dir_gpio, direction);
  }

  // If we don't have microstepping enabled, we want the value to b, 1 so we do
  // full steps.
  int microstepping_factor = MAX(ctx->cfg.microstepping_mode, 1);

  float velocity_sps =
      roundf(velocity.dps * ctx->steps_per_degree * microstepping_factor);

  // Clamp frequency
  uint32_t freq_hz = MAX((uint32_t)fabsf(velocity_sps), MIN_FREQ_HZ);

  // Update frequency and duty
  ledc_set_freq(LEDC_LOW_SPEED_MODE, ctx->cfg.pwm_timer, freq_hz);
  ledc_set_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.pwm_channel,
                ctx->duty_50_percent);
  ledc_update_duty(LEDC_LOW_SPEED_MODE, ctx->cfg.pwm_channel);
}

// Initialization

esp_err_t stepper_init(const stepper_control_config_t *cfg,
                       uint16_t latest_potentiometer_adc_value,
                       stepper_control_handle_t *out_handle) {
  // Validate config
  if (!cfg) return ESP_ERR_INVALID_ARG;

  motion_control_context_t ctx = {0};

  stepper_control_handle_t handle = cfg->pwm_channel;

  // Reset and store config
  ctx.cfg = *cfg;
  ctx.spinlock = (portMUX_TYPE)portMUX_INITIALIZER_UNLOCKED;

  // We define the degrees per second and need to convert that into steps per
  // second
  ctx.steps_per_degree = (float)cfg->steps_per_rev * cfg->gear_ratio / 360.0f;

  const float min_allowed_velocity = (float)MIN_FREQ_HZ / ctx.steps_per_degree;
  if (cfg->max_velocity.dps < min_allowed_velocity) {
    ESP_LOGE(TAG, "max_velocity must be at least %f dps", min_allowed_velocity);
    return ESP_ERR_INVALID_ARG;
  }

  // Configure GPIOS for STEP, DIR and ENABLE
  uint64_t pin_mask = (1ULL << cfg->step_gpio);
  if (cfg->dir_gpio != GPIO_NUM_NC) {
    pin_mask |= (1ULL << cfg->dir_gpio);
  }
  if (cfg->enable_gpio != GPIO_NUM_NC) {
    pin_mask |= (1ULL << cfg->enable_gpio);
  }
  if (cfg->microstepping_mode != MICROSTEP_NONE) {
    if (cfg->microstep_m0_gpio == GPIO_NUM_NC) {
      ESP_LOGE(TAG,
               "Microstepping is enabled, but microstep_m0_gpio isn't enabled");
      return ESP_ERR_INVALID_ARG;
    } else {
      pin_mask |= (1ULL << cfg->microstep_m0_gpio);
    }
    if (cfg->microstep_m1_gpio == GPIO_NUM_NC) {
      ESP_LOGE(TAG,
               "Microstepping is enabled, but microstep_m1_gpio isn't enabled");
      return ESP_ERR_INVALID_ARG;
    } else {
      pin_mask |= (1ULL << cfg->microstep_m1_gpio);
    }
    if (cfg->microstep_m2_gpio == GPIO_NUM_NC) {
      ESP_LOGE(TAG,
               "Microstepping is enabled, but microstep_m2_gpio isn't enabled");
      return ESP_ERR_INVALID_ARG;
    } else {
      pin_mask |= (1ULL << cfg->microstep_m2_gpio);
    }
  }

  gpio_config_t io_conf = {
      .pin_bit_mask = pin_mask,
      .mode = GPIO_MODE_OUTPUT,
      .pull_up_en = GPIO_PULLUP_DISABLE,
      .pull_down_en = GPIO_PULLDOWN_DISABLE,
      .intr_type = GPIO_INTR_DISABLE,
  };
  ESP_RETURN_ON_ERROR(gpio_config(&io_conf), TAG, "Failed to configure GPIOs");

  // Set initial GPIO states
  gpio_set_level(cfg->step_gpio, 0);
  if (cfg->dir_gpio != GPIO_NUM_NC) {
    gpio_set_level(cfg->dir_gpio, 0);
  }
  if (cfg->enable_gpio != GPIO_NUM_NC) {
    gpio_set_level(cfg->enable_gpio, 0);
  }  // active low on DRV8825

  // Set up microstepping.
  switch (cfg->microstepping_mode) {
    case MICROSTEP_NONE: {
      // We don't need to do anything.
      break;
    }
    case MICROSTEP_1_1: {
      gpio_set_level(cfg->microstep_m0_gpio, 0);
      gpio_set_level(cfg->microstep_m1_gpio, 0);
      gpio_set_level(cfg->microstep_m2_gpio, 0);
      break;
    }
    case MICROSTEP_1_2: {
      gpio_set_level(cfg->microstep_m0_gpio, 1);
      gpio_set_level(cfg->microstep_m1_gpio, 0);
      gpio_set_level(cfg->microstep_m2_gpio, 0);
      break;
    }
    case MICROSTEP_1_4: {
      gpio_set_level(cfg->microstep_m0_gpio, 0);
      gpio_set_level(cfg->microstep_m1_gpio, 1);
      gpio_set_level(cfg->microstep_m2_gpio, 0);
      break;
    }
    case MICROSTEP_1_8: {
      gpio_set_level(cfg->microstep_m0_gpio, 1);
      gpio_set_level(cfg->microstep_m1_gpio, 1);
      gpio_set_level(cfg->microstep_m2_gpio, 0);
      break;
    }
    case MICROSTEP_1_16: {
      gpio_set_level(cfg->microstep_m0_gpio, 0);
      gpio_set_level(cfg->microstep_m1_gpio, 0);
      gpio_set_level(cfg->microstep_m2_gpio, 1);
      break;
    }
    case MICROSTEP_1_32: {
      gpio_set_level(cfg->microstep_m0_gpio, 1);
      gpio_set_level(cfg->microstep_m1_gpio, 0);
      gpio_set_level(cfg->microstep_m2_gpio, 1);
      break;
    }
    default: {
      ESP_LOGE(TAG, "You set the microstepping_mode to invalid value (%d)",
               cfg->microstepping_mode);
      return ESP_ERR_INVALID_ARG;
    }
  }

  // Configure LEDC timer
  // Start off at lowest possible speed.
  const uint32_t init_freq_hz = MIN_FREQ_HZ;

  ledc_timer_config_t timer_cfg = {
      .speed_mode = LEDC_LOW_SPEED_MODE,
      // For some reason, the LEDC library doesn't like setting a duty
      // resolution below 14 bits if the frequency is at its minimum of 5 Hz.
      .duty_resolution = LEDC_TIMER_14_BIT,
      .timer_num = cfg->pwm_timer,
      .freq_hz = init_freq_hz,
      .clk_cfg = LEDC_USE_APB_CLK,
  };
  ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_cfg), TAG,
                      "Failed to configure LEDC timer");

  // Configure LEDC channel for STEP output
  ctx.duty_50_percent = (1 << (timer_cfg.duty_resolution -
                               1));  // 50% duty for set duty resolution

  ledc_channel_config_t channel_cfg = {
      .gpio_num = cfg->step_gpio,
      .speed_mode = LEDC_LOW_SPEED_MODE,
      .channel = cfg->pwm_channel,
      .timer_sel = cfg->pwm_timer,
      .duty = 0,  // IDLE (no pulses)
      .hpoint = 0,
  };
  ESP_RETURN_ON_ERROR(ledc_channel_config(&channel_cfg), TAG,
                      "Failed to configure LEDC channel");

  PotentiometerAngle initial_angle = potentiometer_adc_to_angle(
      &ctx.cfg.potentiometer, latest_potentiometer_adc_value);
  ctx.current_angle = initial_angle;
  // Our target angle should be within the range of allowed angles, even if
  // the current angle isn't.
  ctx.target_angle =
      clamp_potentiometer_angle(&ctx.cfg.potentiometer, ctx.current_angle);
  ctx.use_position_feedback = true;

  ESP_LOGI(TAG,
           "Potentiometer initialized: ADC channel=%d, raw=%u, angle=%.2f deg",
           cfg->pot_adc_channel, latest_potentiometer_adc_value, initial_angle);

  // Initialize motion state
  ctx.estop_active = false;
  ctx.is_moving = false;
  ctx.current_velocity = (AngularVelocity){0.0F};

  ctx.is_initialized = true;

  s_contexts[handle] = ctx;

  *out_handle = handle;

  ESP_LOGI(TAG,
           "Stepper initialized: steps/deg=%.3f, max_vel=%.2f sps, "
           "max_accel=%.2f sps²",
           ctx.steps_per_degree,
           ctx.cfg.max_velocity.dps * ctx.steps_per_degree,
           ctx.cfg.max_accel.dps2 * ctx.steps_per_degree);

  return ESP_OK;
}

esp_err_t stepper_deinit(stepper_control_handle_t handle) {
  motion_control_context_t *ctx = &s_contexts[handle];

  // Stop the motor
  stop_motor(handle);

  // TODO(johan): Do we need to actually reset ledc/gpio as well?

  *ctx = (motion_control_context_t){0};

  return ESP_OK;
}

void stepper_update(stepper_control_handle_t handle, uint16_t dt_ms,
                    uint16_t latest_potentiometer_adc_value) {
  motion_control_context_t *ctx = &s_contexts[handle];
  PotentiometerAngle angle_deg = potentiometer_adc_to_angle(
      &ctx->cfg.potentiometer, latest_potentiometer_adc_value);

  // Take snapshot of shared state and update current angle from feedback
  portENTER_CRITICAL(&ctx->spinlock);
  bool estop = ctx->estop_active;
  PotentiometerAngle target_angle = ctx->target_angle;
  float current_velocity_sps =
      ctx->current_velocity.dps * ctx->steps_per_degree;
  ctx->current_angle = angle_deg;
  ctx->use_position_feedback = true;
  portEXIT_CRITICAL(&ctx->spinlock);

  if (estop) {
    stop_motor(handle);
    return;
  }

  MotorRampingArgs args = {
      .current_angle = ctx->current_angle,
      .target_angle = ctx->target_angle,
      .deadband = (PotentiometerAngle){DEADBAND_DEG},
      .current_velocity = ctx->current_velocity,
      .max_acceleration = ctx->cfg.max_accel,
      .max_velocity = ctx->cfg.max_velocity,
      .timestep_ms = dt_ms,
  };
  AngularVelocity new_velocity = motor_ramping_trapezoidal(&args);

  // NOTE: Normally checking equality of floats is imprecise, but in this case
  // the literal value 0.F gets returned when within the deadband, so it should
  // be fine checking against the same literal.
  if (new_velocity.dps == 0.F) {
    static uint32_t log_counter = 0;
    if (++log_counter >= 100) {  // Log every 100 updates
      log_counter = 0;
      ESP_LOGI(TAG, "STOPPING!");
      ESP_LOGI(
          TAG,
          "Update: target=%.2f°, current=%.2f°, error=%.2f°, vel=%.1f sps, "
          "moving=%d",
          target_angle.degree, angle_deg.degree,
          target_angle.degree - angle_deg.degree, current_velocity_sps,
          ctx->is_moving);
    }
    stop_motor(handle);
    return;
  }

  // Apply motor velocity (handles enable/disable, frequency, duty)
  apply_motor_velocity(handle, new_velocity);

  // Update the current state.
  portENTER_CRITICAL(&ctx->spinlock);
  ctx->current_velocity = new_velocity;
  ctx->is_moving = true;
  portEXIT_CRITICAL(&ctx->spinlock);

  // Logging (periodic, not every update)
  static uint32_t log_counter = 0;
  if (++log_counter >= 100) {  // Log every 100 updates
    log_counter = 0;
    ESP_LOGD(TAG,
             "Update: target=%.2f°, current=%.2f°, error=%.2f°, vel=%.1f sps, "
             "moving=%d",
             target_angle.degree, angle_deg.degree,
             target_angle.degree - angle_deg.degree, current_velocity_sps,
             ctx->is_moving);
  }
}

// ------ Setters ------

void stepper_set_target_angle(stepper_control_handle_t handle,
                              JointAngle target_angle) {
  motion_control_context_t *ctx = &s_contexts[handle];

  PotentiometerAngle target_potentiometer_angle =
      to_potentiometer_angle(&ctx->cfg.potentiometer, target_angle);

  target_potentiometer_angle = clamp_potentiometer_angle(
      &ctx->cfg.potentiometer, target_potentiometer_angle);
  portENTER_CRITICAL(&ctx->spinlock);
  ctx->target_angle = target_potentiometer_angle;
  portEXIT_CRITICAL(&ctx->spinlock);
}

void stepper_set_estop(stepper_control_handle_t handle, bool active) {
  motion_control_context_t *ctx = &s_contexts[handle];

  portENTER_CRITICAL(&ctx->spinlock);
  ctx->estop_active = active;
  portEXIT_CRITICAL(&ctx->spinlock);
  if (active) {
    stop_motor(handle);
  }
}

// ------ Getters ------

PotentiometerAngle stepper_get_current_angle(stepper_control_handle_t handle) {
  const motion_control_context_t *ctx = &s_contexts[handle];

  portENTER_CRITICAL(&ctx->spinlock);
  PotentiometerAngle angle = ctx->current_angle;
  portEXIT_CRITICAL(&ctx->spinlock);
  return angle;
}

PotentiometerAngle stepper_get_target_angle(stepper_control_handle_t handle) {
  const motion_control_context_t *ctx = &s_contexts[handle];

  portENTER_CRITICAL(&ctx->spinlock);
  PotentiometerAngle angle = ctx->target_angle;
  portEXIT_CRITICAL(&ctx->spinlock);
  return angle;
}

AngularVelocity stepper_get_current_velocity(stepper_control_handle_t handle) {
  const motion_control_context_t *ctx = &s_contexts[handle];

  portENTER_CRITICAL(&ctx->spinlock);
  AngularVelocity velocity = ctx->current_velocity;
  portEXIT_CRITICAL(&ctx->spinlock);
  return velocity;
}

bool stepper_is_moving(stepper_control_handle_t handle) {
  const motion_control_context_t *ctx = &s_contexts[handle];

  portENTER_CRITICAL(&ctx->spinlock);
  bool moving = ctx->is_moving;
  portEXIT_CRITICAL(&ctx->spinlock);
  return moving;
}

bool stepper_has_position_feedback(stepper_control_handle_t handle) {
  const motion_control_context_t *ctx = &s_contexts[handle];

  portENTER_CRITICAL(&ctx->spinlock);
  bool has_feedback = ctx->use_position_feedback;
  portEXIT_CRITICAL(&ctx->spinlock);
  return has_feedback;
}

const stepper_control_config_t *stepper_get_cfg(
    stepper_control_handle_t handle) {
  return &s_contexts[handle].cfg;
}
