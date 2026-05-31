# Potentiometer Component

This component provides the ability to convert between ADC values,
angles in a potentiometer's frame of reference and the corresponding
angles in a joint's frame of reference.

## Setup process

First of all, you need to figure out the relationship between the ADC
values you read from the potentiometer and the angles they represent. For
this, you need three things: the potentiometer's range of motion and the
ADC values measured when the potentiometer is turned as far as possible
in both the clockwise as well as the counterclockwise direction. If
you are using the [adc_manager component](../adc_manager/), these ADC
values should be 0 and 𝑉<sub>in</sub> expressed in millivolts. From
this, you can create a linear relationship between the range of motion
and the ADC values, with the minimum ADC value corresponding to 0&deg;
and the maximum corresponding to the largest value in the range of motion.

> [!note]
> This assumes that the ADC values increase linearly with the voltage.
> Read more about this in [the ADC documentation](../../../../docs/adc.md).

Once the ADC values have been measured, you can set the
`degrees_of_motion`, `min_adc_value`, and `max_adc_value` members
in the configuration struct.  The `min_potentiometer_angle` and
`max_potentiometer_angle` members can then be measured by calling
`potentiometer_adc_to_angle` on the ADC values at the desired min and
max angles that you want your joint to be able to move to.

## Usage

A usage example:
```c
#include "potentiometer.h"

void update_joint_angle(uint16_t adc_value) {
  // Here's an example potentiometer configuration.
  const Potentiometer elbow_pot = {
      // Turning the potentiometer to the 0 degree position gives an ADC value
      // of 3, turning it to the 285 degree position gives an ADC value of
      // 3098.
      // NOTE: Ideally you should be using the ADC manager component or
      // manually calibrating the ADC values using ESP-IDF's ADC calibration
      // driver, in which case the min and max values should be ~0 and ~Vin in
      // millivolts.
      .degrees_of_motion = {285.F},
      .min_adc_value = 3,
      .max_adc_value = 3098,
      // The desired minimum joint limit gives a potentiometer angle of 98.
      .min_potentiometer_angle = {98.F},
      // The desired maximum joint limit gives a potentiometer angle of 170.
      .max_potentiometer_angle = {170.F},
      // When the potentiometer is at min_potentiometer_angle, the joint should
      // be treated as being at the joint angle of 5 degrees (Since
      // clamp_potentiometer_angle() clamps to the min_potentiometer_angle,
      // joint angles get clamped to this value).
      .min_potentiometer_angle_as_joint_angle = {5.F},
      // There's an 18:15 gear ratio between how much the joint rotates
      // compared to the potentiometer. This means the [98, 170] potentiometer
      // angle range corresponds to the [5, 65] joint angle range (since 60 *
      // 18 / 5 = 72).
      .joint_angle_to_potentiometer_angle_ratio = 18.F / 15.F,
      // The relationship between the joint angles and potentiometer angles is
      // reversed, i.e. joint angle 5 maps to potentiometer angle 170 instead
      // of 98 and joint angle 65 maps to potentiometer angle 98 instead of
      // 170.
      .is_reversed = true,
  };

  // Convert a raw ADC reading to a potentiometer angle.
  PotentiometerAngle pot_angle =
      potentiometer_adc_to_angle(&elbow_pot, adc_value);

  // Convert a potentiometer angle to the joint reference frame.
  JointAngle joint_angle = to_joint_angle(&elbow_pot, pot_angle);

  // Convert a joint angle to a potentiometer angle. Note that it's outside the
  // [5, 65] range denoted by the config, meaning target_pot_angle will be
  // greater than max_potentiometer_angle.
  PotentiometerAngle target_pot_angle =
      to_potentiometer_angle(&elbow_pot, (JointAngle){70.F});

  // Clamp a potentiometer angle to the config's min and max potentiometer
  // angles. In this case, target_pot_angle will get clamped to
  // max_potentiometer_angle, meaning it gets set to 170 degrees.
  target_pot_angle = clamp_potentiometer_angle(&elbow_pot, target_pot_angle);
}
```
