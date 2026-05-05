# Modifying the JX Servo PDI-HV2060MG Servos to provide continuous rotation

The shoulder flexion/extension and abduction/adduction
joints of the robotic arm are moved using [JX Servo
PDI-HV2060MG](http://www.jx-servo.com/en/Product/STANDARD/SD/544.html)
servo motors. These servos have a maximum range of motion of 180&deg;,
probably corresponding to pulse widths in the range 650&ndash;2350 &mu;s,
(there's literally no official datasheet available, so it's hard to
state things confidently). This limited range of motion ends up being a
problem in this project since both joints controlled by the servos have
a gear ratio of 15:1 meaning that the effective range we get for those
joints is 12&deg;.

## Modding servo motors to be continuous

To address the conundrum of the servo motors not
having a large enough range of motion, we followed [this
guide](https://learn.adafruit.com/modifying-servos-for-continuous-rotation/overview)
to modify them so that they provide continuous rotation, allowing us
to rotate the joints as much as we want (the guide is for a different
servo motor, but they work the same way). The basic idea is:
1. Remove the internal potentiometer.
2. Remove the physical stop on the gears.
3. Add resistors between $V_{\text{in}}$ and $V_{\text{out}}$ as well
   as $V_{\text{out}}$ and $\text{gnd}$ where the internal
   potentiometer was connected.

## Crash course in how the unmodded servo works

The servo has an internal potentiometer detecting its current
position. Providing the servo with a pulse width makes the servo target
a value of the potentiometer, thereby making it rotate until it reaches
the correct value. The range of pulse widths that correspond with the
180&deg; range of motion is (probably?) 650&ndash;2350 &mu;s, with ~1500 &mu;s
being the midpoint.

## Controlling the modded servo motors

The way the HV2060 servo motors function after the modification is that
the servo thinks that its internal potentiometer is always positioned at
its midpoint corresponding to a pulse width of ~1500 &mu;s. If it
is then given a pulse width other than 1500 &mu;s it will start
rotating in an attempt to reach its corresponding target potentiometer
value. However, since we make the servo think its potentiometer value
is always at its midpoint, the servo ends up rotating continuously since
it never reaches its target potentiometer value.

The direction of the rotation is controlled by setting the pulse width
either greater than or less than the midpoint of 1500 &mu;s,
while setting the pulse width to exactly 1500 &mu;s makes it stop
rotating. The greater the absolute difference between the pulse width used
and the midpoint, the greater the resulting speed and torque ends up being.

### Issues with controlling the modified servo
> [!caution]
> One consequence of this modification is that the software is entirely
> responsible for making sure that the motors perform as desired. If
> the software sets a pulse width other than 1500 &mu;s, the
> motor *will* keep on rotating until it receives a new pulse width of
> 1500 &mu;s. That means that if the motors are rotating and the
> microcontroller's program is unable to update the pulse width once the
> joint reaches its desired angle (e.g. due to a crash, infinite loop,
> or simply a missed deadline), the joint will keep on rotating past its
> desired angle, possibly past its safety limits, and potentially breaking
> the arm.

The speed that the servo rotates at is not linear with relation to the
pulse width's distance from the midpoint. The midpoint has an effective
deadband around it where the motor won't rotate. Once the pulse width
distance from the midpoint is large enough, the motor starts rotating,
but with a decently high angular velocity right from the start, making
it difficult to perform small adjustments to the joint angles. The
velocity and torque then increases together with the pulse width until
it eventually caps out long before the maximum pulse width is reached.

Another thing that affects the speed of the servo is its load. This
means that a pulse width rotating in a direction which raises the arm
will raise the arm slower than the equivalent pulse width rotating in
a direction that lowers the arm, since the motor has to fight against-
or is helped by gravity. This also means that the effective pulse
width deadband around the midpoint is larger in the direction that
experiences greater load, since more torque is required to overcome it.
Whether the pulse width that maxes out the speed without load will still
apply maximum torque under load has not been investigated, but seems to
not be necessary for the loads that the robot arm experiences.

All of these issues result in a situation where the servo *can* move
the arm, but it is difficult to move it precisely. The actual velocity
is unlikely to be the same as the predicted velocity, meaning that
motor ramping algorithms are likely to make the joint overshoot its
target angle.

## Servo component
To make controlling the servo motor
relatively painless, an ESP-IDF component was created in
[`src/esp32/components/servo/`](../src/esp32/components/servo/). See its
[readme](../src/esp32/components/servo/readme.md) for more information.
