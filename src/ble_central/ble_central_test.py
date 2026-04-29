"""Bluetooth Low Energy Central test program."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt
from bleak import (
    BleakClient,
    BleakScanner,
)

from sensor_packet_serialization import (
    ValueDataType,
    decode_packet,
    deserialize_packet_data,
)

if TYPE_CHECKING:
    from bleak.backends.characteristic import BleakGATTCharacteristic

SERVICE_UUID = "23011525-1212-efde-1523-785feabcd122"
EMG_CHARACTERISTIC_UUID = "24011525-1212-efde-1523-785feabcd122"
IMU_CHARACTERISTIC_UUID = "25011525-1212-efde-1523-785feabcd122"
PIEZO_CHARACTERISTIC_UUID = "26011525-1212-efde-1523-785feabcd122"

EMG_SENSOR_COUNT = 2
EMG_BYTES_PER_VALUE = 2
EMG_VALUES_PER_SAMPLE = 1
EMG_FREQUENCY = 4000
EMG_MS_PER_WINDOW = 200
EMG_MS_PER_OVERLAP = 50
EMG_SAMPLES_PER_WINDOW = int(EMG_FREQUENCY * EMG_MS_PER_WINDOW / 1000)
EMG_SAMPLES_PER_OVERLAP = int(EMG_FREQUENCY * EMG_MS_PER_OVERLAP / 1000)

IMU_SENSOR_COUNT = 1
IMU_BYTES_PER_VALUE = 4
IMU_VALUES_PER_SAMPLE = 6
IMU_FREQUENCY = 100
IMU_MS_PER_WINDOW = 200
IMU_MS_PER_OVERLAP = 50
IMU_SAMPLES_PER_WINDOW = int(IMU_FREQUENCY * IMU_MS_PER_WINDOW / 1000)
IMU_SAMPLES_PER_OVERLAP = int(IMU_FREQUENCY * IMU_MS_PER_OVERLAP / 1000)

PIEZO_SENSOR_COUNT = 1
PIEZO_BYTES_PER_VALUE = 2
PIEZO_VALUES_PER_SAMPLE = 1
PIEZO_FREQUENCY = 100
PIEZO_MS_PER_WINDOW = 200
PIEZO_MS_PER_OVERLAP = 50
PIEZO_SAMPLES_PER_WINDOW = int(PIEZO_FREQUENCY * PIEZO_MS_PER_WINDOW / 1000)
PIEZO_SAMPLES_PER_OVERLAP = int(PIEZO_FREQUENCY * PIEZO_MS_PER_OVERLAP / 1000)

first_emg_sequence_number = 0
first_imu_sequence_number = 0
first_piezo_sequence_number = 0
latest_emg_sequence_number = 0
latest_imu_sequence_number = 0
latest_piezo_sequence_number = 0

dropped_emg_packet_count = 0
dropped_imu_packet_count = 0
dropped_piezo_packet_count = 0


class SampleWindow:
    """A sliding window of sensor samples."""

    def __init__(
        self,
        samples_per_window: int,
        samples_per_overlap: int,
        values_per_sample: int,
    ) -> None:
        """Create a sliding window capable of storing sensor samples."""
        self._capacity = samples_per_window
        self._overlap_length = samples_per_overlap
        self._length = 0
        self._last_sequence_number_received = 0
        # Preallocate space.
        self._samples = np.array([np.array([0] * values_per_sample)] * self._capacity)

    def append_samples(
        self,
        new_samples: npt.NDArray,
        sequence_number: int,
    ) -> npt.NDArray | None:
        """Append the samples to the window.

        :return: A copy of the full window or None if it's not full.
        """
        if self._last_sequence_number_received == 0:
            self._last_sequence_number_received = sequence_number - 1

        if self._last_sequence_number_received != sequence_number - 1:
            # We've dropped packets.
            dropped_packets_count = (
                sequence_number - self._last_sequence_number_received - 1
            )
            # TODO(johan): Actually handle dropped packets.
            print(f"Dropped {dropped_packets_count} packets.")

        self._last_sequence_number_received = sequence_number

        # NOTE: This assumes that the length of the window is evenly divisible by the
        # amount of new samples received.
        if self._length + len(new_samples) > self._capacity:
            err = (
                f"Trying to add {len(new_samples)} samples to a "
                f"full({self._length}/{self._capacity}) sample window."
            )
            raise BufferError(err)

        self._samples[self._length : self._length + len(new_samples)] = new_samples
        self._length += len(new_samples)

        # If we've filled up a full window, we return a copy of it and prepare filling
        # up the next window by moving the overlap to the start.
        if self._length == self._capacity:
            full_window = self._samples[: self._capacity].copy()
            self._samples[: self._overlap_length] = self._samples[
                -self._overlap_length :
            ]
            self._length = self._overlap_length
            return full_window

        return None

    def __str__(self) -> str:
        """Return the string representation of the window."""
        return str(self._samples)

    _length: int
    _capacity: int
    _overlap_length: int
    # 2D array of samples: _samples[0] -> sample, _samples[0][0] -> value in sample.
    _samples: npt.NDArray
    _last_sequence_number_received: int


def process_emg_window(sensor_id: int, window_buf: npt.NDArray) -> None:
    """Start processing the full EMG window."""
    # This needs to in some way send the data to the part of the system processing the
    # window in a non-blocking fashion.
    print(f"Full EMG{sensor_id} window:")
    print(f"{window_buf}")


def process_imu_window(sensor_id: int, window_buf: npt.NDArray) -> None:
    """Start processing the full IMU window."""
    # This needs to in some way send the data to the part of the system processing the
    # window in a non-blocking fashion.
    print(f"Full IMU{sensor_id} window:")
    print(f"{window_buf}")


def process_piezo_window(sensor_id: int, window_buf: npt.NDArray) -> None:
    """Start processing the full piezo window."""
    # This needs to in some way send the data to the part of the system processing the
    # window in a non-blocking fashion.
    print(f"Full piezo{sensor_id} window:")
    print(f"{window_buf}")


def update_sample_windows(
    sample_windows: list[SampleWindow],
    new_sensor_samples: list[npt.NDArray],
    sequence_number: int,
    full_window_processing_cb: Callable[[int, npt.NDArray], None],
) -> None:
    """Add new samples to the windows and begin the processing of any full windows."""
    for sensor_id, (sample_window, new_samples) in enumerate(
        zip(sample_windows, new_sensor_samples),
    ):
        maybe_full_window = sample_window.append_samples(
            new_samples,
            sequence_number,
        )
        if maybe_full_window is not None:
            full_window_processing_cb(sensor_id, maybe_full_window)


def set_up_notify_handler(
    emg_sample_windows: list[SampleWindow],
    imu_sample_windows: list[SampleWindow],
    piezo_sample_windows: list[SampleWindow],
) -> tuple[
    Callable[[BleakGATTCharacteristic, bytearray], None],
    Callable[[BleakGATTCharacteristic, bytearray], None],
    Callable[[BleakGATTCharacteristic, bytearray], None],
]:
    """Return the EMG, IMU, and piezo characteristic handlers."""

    def emg_notify_handler(
        characteristic: BleakGATTCharacteristic,
        data: bytearray,
    ) -> None:
        sequence_number, sensor_data = decode_packet(memoryview(data))
        sensors = deserialize_packet_data(
            sensor_data,
            bytes_per_value=EMG_BYTES_PER_VALUE,
            values_per_sample=EMG_VALUES_PER_SAMPLE,
            sensor_count=EMG_SENSOR_COUNT,
            value_data_type=ValueDataType.UNSIGNED_INTEGER,
        )
        global first_emg_sequence_number
        global latest_emg_sequence_number
        if first_emg_sequence_number == 0:
            first_emg_sequence_number = sequence_number
            latest_emg_sequence_number = sequence_number - 1

        global dropped_emg_packet_count
        dropped_emg_packet_count += sequence_number - latest_emg_sequence_number - 1

        update_sample_windows(
            emg_sample_windows,
            sensors,
            sequence_number,
            process_emg_window,
        )

        latest_emg_sequence_number = sequence_number

    def imu_notify_handler(
        characteristic: BleakGATTCharacteristic,
        data: bytearray,
    ) -> None:
        sequence_number, sensor_data = decode_packet(memoryview(data))
        sensors = deserialize_packet_data(
            sensor_data,
            bytes_per_value=IMU_BYTES_PER_VALUE,
            values_per_sample=IMU_VALUES_PER_SAMPLE,
            sensor_count=IMU_SENSOR_COUNT,
            value_data_type=ValueDataType.FLOATING_POINT,
        )
        global first_imu_sequence_number
        global latest_imu_sequence_number
        if first_imu_sequence_number == 0:
            first_imu_sequence_number = sequence_number
            latest_imu_sequence_number = sequence_number - 1

        global dropped_imu_packet_count
        dropped_imu_packet_count += sequence_number - latest_imu_sequence_number - 1

        update_sample_windows(
            imu_sample_windows,
            sensors,
            sequence_number,
            process_imu_window,
        )

        latest_imu_sequence_number = sequence_number

    def piezo_notify_handler(
        characteristic: BleakGATTCharacteristic,
        data: bytearray,
    ) -> None:
        sequence_number, sensor_data = decode_packet(memoryview(data))
        sensors = deserialize_packet_data(
            sensor_data,
            bytes_per_value=PIEZO_BYTES_PER_VALUE,
            values_per_sample=PIEZO_VALUES_PER_SAMPLE,
            sensor_count=PIEZO_SENSOR_COUNT,
            value_data_type=ValueDataType.UNSIGNED_INTEGER,
        )
        global first_piezo_sequence_number
        global latest_piezo_sequence_number
        if first_piezo_sequence_number == 0:
            first_piezo_sequence_number = sequence_number
            latest_piezo_sequence_number = sequence_number - 1

        global dropped_piezo_packet_count
        dropped_piezo_packet_count += sequence_number - latest_piezo_sequence_number - 1

        update_sample_windows(
            piezo_sample_windows,
            sensors,
            sequence_number,
            process_piezo_window,
        )

        latest_piezo_sequence_number = sequence_number

    return (emg_notify_handler, imu_notify_handler, piezo_notify_handler)


async def main() -> None:
    """Subscribe to sensor data from peripheral and count dropped packets."""
    timeout = 10
    print(f"scanning for {timeout} seconds, please wait...")

    device = await BleakScanner.find_device_by_name(
        "LIMBServer",
        timeout,
    )
    if not device:
        print("Couldn't find device.")
        return

    print(f"Found device {device}!")
    print(f"Details: {device.details}")

    def disconnect_handler(bc: BleakClient) -> None:
        print(f"Disconnected from {bc.address}.")

    emg_sample_windows: list[SampleWindow] = [
        SampleWindow(
            EMG_SAMPLES_PER_WINDOW,
            EMG_SAMPLES_PER_OVERLAP,
            EMG_VALUES_PER_SAMPLE,
        )
        for _ in range(EMG_SENSOR_COUNT)
    ]
    imu_sample_windows: list[SampleWindow] = [
        SampleWindow(
            IMU_SAMPLES_PER_WINDOW,
            IMU_SAMPLES_PER_OVERLAP,
            IMU_VALUES_PER_SAMPLE,
        )
        for _ in range(IMU_SENSOR_COUNT)
    ]
    piezo_sample_windows: list[SampleWindow] = [
        SampleWindow(
            PIEZO_SAMPLES_PER_WINDOW,
            PIEZO_SAMPLES_PER_OVERLAP,
            PIEZO_VALUES_PER_SAMPLE,
        )
        for _ in range(PIEZO_SENSOR_COUNT)
    ]

    (emg_notify_handler, imu_notify_handler, piezo_notify_handler) = (
        set_up_notify_handler(
            emg_sample_windows,
            imu_sample_windows,
            piezo_sample_windows,
        )
    )

    async with BleakClient(device, disconnect_handler) as client:
        print(f"Connected: {client.is_connected}")
        print(f"Service handles: {client.services.services.keys()}")

        service = client.services.get_service(SERVICE_UUID)
        if service is None:
            print("Error: Couldn't get services.")
            return
        print(f"Sensor service: {service}")
        print(f"Service UUID is valid: {service.uuid == SERVICE_UUID}")
        print(f"Characteristic count: {len(service.characteristics)}")

        emg_chr = service.get_characteristic(EMG_CHARACTERISTIC_UUID)
        if emg_chr is None:
            print("Error: Couldn't get EMG characteristic.")
            return
        imu_chr = service.get_characteristic(IMU_CHARACTERISTIC_UUID)
        if imu_chr is None:
            print("Error: Couldn't get IMU characteristic.")
            return
        piezo_chr = service.get_characteristic(PIEZO_CHARACTERISTIC_UUID)
        if piezo_chr is None:
            print("Error: Couldn't get piezo characteristic.")
            return

        # Single read from characteristic.
        print()
        emg_value, imu_value, piezo_value = await asyncio.gather(
            client.read_gatt_char(emg_chr),
            client.read_gatt_char(imu_chr),
            client.read_gatt_char(piezo_chr),
        )

        print(f"EMG char (including sequence number): {emg_value}")
        print()
        print(f"IMU char (including sequence number): {imu_value}")
        print()
        print(f"Piezo char (including sequence number): {piezo_value}")
        print()

        subscribe_period = 1
        print(f"Starting subscription for {subscribe_period} seconds...")

        # Subscribe to characteristics.
        await asyncio.gather(
            client.start_notify(emg_chr, emg_notify_handler),
            client.start_notify(imu_chr, imu_notify_handler),
            client.start_notify(piezo_chr, piezo_notify_handler),
        )

        # Sleeping for 1 second means the expected amount of packets received is
        # frequency / sample count per packet.
        await asyncio.sleep(subscribe_period)

        await asyncio.gather(
            client.stop_notify(emg_chr),
            client.stop_notify(imu_chr),
            client.stop_notify(piezo_chr),
        )

        print("Stopped subscription!")

        print(
            f"EMG seq nr span: "
            f"{first_emg_sequence_number}-{latest_emg_sequence_number} "
            f"({latest_emg_sequence_number - first_emg_sequence_number + 1} total)",
        )
        print(f"Dropped EMG packets: {dropped_emg_packet_count}.")

        print(
            f"IMU seq nr span: "
            f"{first_imu_sequence_number}-{latest_imu_sequence_number} "
            f"({latest_imu_sequence_number - first_imu_sequence_number + 1} total)",
        )
        print(f"Dropped IMU packets: {dropped_imu_packet_count}.")

        print(
            f"piezo seq nr span: "
            f"{first_piezo_sequence_number}-{latest_piezo_sequence_number} "
            f"({latest_piezo_sequence_number - first_piezo_sequence_number + 1} total)",
        )
        print(f"Dropped piezo packets: {dropped_piezo_packet_count}.")


if __name__ == "__main__":
    asyncio.run(main())
