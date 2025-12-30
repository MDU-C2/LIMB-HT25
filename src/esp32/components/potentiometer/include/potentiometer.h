#pragma once

#include <stdint.h>

// These are angles from the potentiometer's frame of reference.
typedef struct {
  float degree;
} PotentiometerAngle;

// These are angles from the joint's frame of reference.
typedef struct {
  float degree;
} JointAngle;

// This represents the characteristics of the potentiometer used.
typedef struct {
  // The range of motion that the potentiometer supports.
  PotentiometerAngle degrees_of_motion;

  // The range of motion that the joint supports in its own frame of reference.
  JointAngle min_potentiometer_angle_as_joint_angle;

  // The range of motion that the joint supports in values expressed by the
  // potentiometer.
  PotentiometerAngle min_potentiometer_angle;
  PotentiometerAngle max_potentiometer_angle;

  // The ratio between 1 degree in the joint angle and the corresponding degree
  // in potentiometer angle.
  float joint_angle_to_potentiometer_angle_ratio;

  // In ideal conditions this should be [0, 2^bitwidth - 1]. However, in
  // practice we might be slightly off, so this should be set to the actual
  // values measured when turning the potentiometer to its minimum and maximum
  // angles.
  uint16_t min_adc_value;
  uint16_t max_adc_value;
} Potentiometer;

// Converts the provided ADC value to the potentiometer's corresponding angle.
PotentiometerAngle potentiometer_adc_to_angle(
    const Potentiometer *potentiometer, uint16_t adc_value);

// Converts an angle from the potentiometer's frame of reference to the joint's
// frame of reference.
JointAngle to_joint_angle(const Potentiometer *potentiometer,
                          PotentiometerAngle angle);

// Converts an angle from the joint's frame of reference to the potentiometer's
// frame of reference.
PotentiometerAngle to_potentiometer_angle(const Potentiometer *potentiometer,
                                          JointAngle angle);

// Clamps a potentiometer angle to the limits determined by the potentiometer.
PotentiometerAngle clamp_potentiometer_angle(const Potentiometer *potentiometer,
                                             PotentiometerAngle angle);
