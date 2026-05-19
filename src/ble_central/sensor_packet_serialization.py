"""Deserialize sensor packet data."""

from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt


def decode_packet(view: memoryview[int]) -> tuple[int, memoryview[int]]:
    """Extract the 32-bit sequence number and sensor data from packet data."""
    return (int.from_bytes(view[:4], "little"), view[4:])


class ValueDataType(Enum):
    """The data type of the values in a sample."""

    SIGNED_INTEGER = 0
    UNSIGNED_INTEGER = 1
    FLOATING_POINT = 2


def deserialize_packet_data(
    packet_data_view: memoryview[int],
    *,
    bytes_per_value: int,
    values_per_sample: int,
    sensor_count: int,
    value_data_type: ValueDataType,
) -> list[npt.NDArray]:
    """Turn packet data into a list of channel samples.

    The packet data is expected to be a byte array of little endian n-byte numeric values.
    After converting the byte array into an n-byte array of sensor values,
    the packet data for 2 sensors with 2 values per samples and N total samples is
    expected to be in the following format:
    [
        # Each row scales horizontally with the # of values per sample.
        sensor1sample1value1, sensor1sample1value2,
        sensor2sample1value1, sensor2sample1value2,
        # The number of rows for each sample scales with # of sensors.

        sensor1sample2value1, sensor1sample2value2,
        sensor2sample2value1, sensor2sample2value2,

        ...

        sensor1sampleNvalue1, sensor1sampleNvalue2,
        sensor2sampleNvalue1, sensor2sampleNvalue2,
    ].

    :return:
        A 3D-list containing the sample values for the different sensors.
        The structure is as follows:
            list[i] -> Sensor i.
            list[i][j] -> Sample j of sensor i.
            list[i][j][k] -> Value k of sample j of sensor i.
    """
    if value_data_type is ValueDataType.FLOATING_POINT:
        dtype_prefix = "f"
    elif value_data_type is ValueDataType.SIGNED_INTEGER:
        dtype_prefix = "i"
    else:
        dtype_prefix = "u"

    values = np.frombuffer(packet_data_view, dtype=f"<{dtype_prefix}{bytes_per_value}")

    # Group values by the amount of values present in a single sample (e.g. the first 6
    # values in IMU data belong to the same sample, the next 6 values belong to another
    # sample, etc.).
    samples = values.reshape((-1, values_per_sample))
    received_sample_count = len(samples)
    if (received_sample_count % sensor_count) != 0:
        err = (
            f"Packet sample count [{received_sample_count}] must be divisible by "
            f"sensor count [{sensor_count}]."
        )
        raise ValueError(err)

    # Each sensor's samples are staggered in the array based on the number of sensors
    # used. Assuming 2 sensors we have:
    # [sensor1sample1, sensor2sample1, sensor1sample2, sensor2sample2, ...]
    # So we want to split it into arrays for each sensor that contains all samples,
    # i.e.:
    # [
    #     [sensor1sample1, sensor1sample2, ...],
    #     [sensor2sample1, sensor2sample2, ...],
    # ]
    return [samples[i::sensor_count] for i in range(sensor_count)]
