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
  float degrees_from_min_joint_angle =
      angle.degree -
      potentiometer->min_joint_angle_as_potentiometer_angle.degree;
  return (JointAngle){potentiometer->min_joint_angle.degree +
                      degrees_from_min_joint_angle};
}

PotentiometerAngle to_potentiometer_angle(const Potentiometer *potentiometer,
                                          JointAngle angle) {
  float degrees_from_min_joint_angle =
      angle.degree - potentiometer->min_joint_angle.degree;

  return (PotentiometerAngle){
      potentiometer->min_joint_angle_as_potentiometer_angle.degree +
      degrees_from_min_joint_angle};
}

PotentiometerAngle clamp_potentiometer_angle(const Potentiometer *potentiometer,
                                             PotentiometerAngle angle) {
  return (PotentiometerAngle){
      LIMB_CLAMP(angle.degree,
                 potentiometer->min_joint_angle_as_potentiometer_angle.degree,
                 potentiometer->max_joint_angle_as_potentiometer_angle.degree)};
}
