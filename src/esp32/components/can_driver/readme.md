# CAN driver

This component allows for sending and receiving CAN messages.

## Reenabling the driver from Bus Off state

> [!warning]
> The driver allows for automatically reenabling CAN communication if a Bus Off state is reached.
> However, this should be used with caution. If there's something wrong with the CAN bus connection
> such that the driver keeps on constantly sending error frames, then reenabling CAN when it enters a
> Bus Off state might end up congesting the bus, starving all other messages. Unfortunately, the bus on the
> arm is just noisy enough that errors don't happen constantly, but frequently enough that modules enter a
> Bus Off state regularly. As such, using this function is a bit of a hack to actually allow us to
> control the arm.
