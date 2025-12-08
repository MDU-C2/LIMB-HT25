#include "potentiometer.h"

#include <stdint.h>
#include <sys/param.h>

// Performs a linear interpolation of x from the range [x0, x1] onto [y0, y1].
static float lerp_from_range(float x, float x0, float x1, float y0, float y1) {
  float x_range = x1 - x0;
  float y_range = y1 - y0;
  return y0 + ((x - x0) * y_range / x_range);
}

#define LIMB_CLAMP(x, x_min, x_max) (MAX(MIN((x), (x_max)), (x_min)))

PotentiometerAngle potentiometer_adc_to_angle(
    const Potentiometer *potentiometer, uint16_t adc_value) {
  // Don't allow values outside the acceptable range.
  adc_value = LIMB_CLAMP(adc_value, potentiometer->min_adc_value,
                         potentiometer->max_adc_value);

  return (PotentiometerAngle){
      lerp_from_range((float)adc_value, potentiometer->min_adc_value,
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
