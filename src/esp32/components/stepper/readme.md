# Stepper component

This is a component for controlling stepper motors using the
[Pololu DRV8825 Stepper M̀otor Driver](https://www.pololu.com/product/2133).

> [!caution]
> This component expects that the `stepper_update` function is called
> periodically to modify the stepper's current velocity based on the target
> angle and the current potentiometer angle. If the function isn't called
> properly, then the stepper will continue to move at its previous velocity,
> meaning that it *will not* stop even if it moves past the target angle! In
> worst case this could lead to parts of the robot arm breaking or human
> injury. As such it is *very important* that any program using the stepper
> component actually manages to call `continuous_servo_update` periodically.

## Pololu DRV8825 Stepper Motor Driver pinout

The image below shows the pinout for the Pololu DRV8825 Stepper Motor Driver:

![Pinout for the Pololu DRV8825 Stepper Motor Driver](https://a.pololu-files.com/picture/0J4232.600.png?f2f6269e0a80c41f0a5147915106aa55)

## Microstepping

The component supports microstepping either by connecting the stepper driver's
microstepping pins to a microcontroller's GPIO pins, or by connecting the
microstepping pins directly to pull-up/pull-down resistors in the hardware.
The method used is selected by setting the `microstepping_type` member in the
configuration struct to either `MICROSTEP_SOFTWARE` or `MICROSTEP_HARDWARE`.

The possible configurations for the stepper driver's microstepping pins can be
seen on its [product page](https://www.pololu.com/product/2133).

> [!warning]
> Make sure the microstepping level you set in the component is actually the same as the microstepping
> level in the stepper driver. Otherwise, if the microstepping level in the
> component is 4 times greater than the microstepping level in the stepper driver,
> then the motor will move 4 times faster than you and the component expects.
> A recommendation is to start off configuring the microstepping pins for the
> stepper driver to use your desired level of microstepping, but set the
> microstepping level in the component to a lower value and increasing it step
> by step to the correct value.
