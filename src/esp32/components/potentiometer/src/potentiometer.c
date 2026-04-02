#include "potentiometer.h"

#include <stdint.h>
#include <sys/param.h>

#include "limb_utils.h"

PotentiometerAngle potentiometer_adc_to_angle(
    const Potentiometer *potentiometer, uint16_t adc_value) {
  return (PotentiometerAngle){
      LIMB_LERP_FROM_RANGE((float)adc_value, potentiometer->min_adc_value,
                           potentiometer->max_adc_value, 0,
                           potentiometer->degrees_of_motion.degree)};
}

JointAngle to_joint_angle(const Potentiometer *potentiometer,
                          PotentiometerAngle angle) {
  const float degrees_from_min_joint_angle =
      potentiometer->is_reversed
          ? potentiometer->max_potentiometer_angle.degree - angle.degree
          : angle.degree - potentiometer->min_potentiometer_angle.degree;

  return (JointAngle){
      potentiometer->min_potentiometer_angle_as_joint_angle.degree +
      (degrees_from_min_joint_angle /
       potentiometer->joint_angle_to_potentiometer_angle_ratio)};
}

PotentiometerAngle to_potentiometer_angle(const Potentiometer *potentiometer,
                                          JointAngle angle) {
  const float distance_from_min_joint_angle =
      angle.degree -
      potentiometer->min_potentiometer_angle_as_joint_angle.degree;
  const float distance_as_pot_angle =
      distance_from_min_joint_angle *
      potentiometer->joint_angle_to_potentiometer_angle_ratio;

  return (PotentiometerAngle){
      potentiometer->is_reversed
          ? potentiometer->max_potentiometer_angle.degree -
                distance_as_pot_angle
          : potentiometer->min_potentiometer_angle.degree +
                distance_as_pot_angle};
}

PotentiometerAngle clamp_potentiometer_angle(const Potentiometer *potentiometer,
                                             PotentiometerAngle angle) {
  return (PotentiometerAngle){
      LIMB_CLAMP(angle.degree, potentiometer->min_potentiometer_angle.degree,
                 potentiometer->max_potentiometer_angle.degree)};
}
