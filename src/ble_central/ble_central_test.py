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

from sensor_packet_serialization import decode_packet

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

IMU_SENSOR_COUNT = 2
IMU_BYTES_PER_VALUE = 2
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


def print_received_packets_stats(packets: list[bytearray], sensor: str) -> None:
    """Print details about the packets received."""
    print(f"{sensor}")
    notification_count = len(packets)
    print(f"notification count: {notification_count}")
    seq_nrs = [decode_packet(memoryview(x))[0] for x in packets]
    print(
        f"seq nrs: {seq_nrs}",
    )
    starting_sequence_number, _ = decode_packet(memoryview(packets[0]))
    i = starting_sequence_number
    for arr in packets:
        seqnr, _ = decode_packet(memoryview(arr))
        if seqnr > i:
            print(f"{i} to {seqnr - 1} ({seqnr - i} packets) are missing.")
            i = seqnr
        i += 1
    print(f"last seq nr received: {i - 1}")
    seq_nr_range = i - starting_sequence_number
    print(f"total range of seq nr: {seq_nr_range}")
    missing = seq_nr_range - notification_count

    print(f"missed packets: {missing} ({missing / i * 100:.2f}%)")


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
    emg_buf: list[bytearray | None],
    imu_buf: list[bytearray | None],
    piezo_buf: list[bytearray | None],
) -> tuple[
    Callable[[BleakGATTCharacteristic, bytearray], None],
    Callable[[BleakGATTCharacteristic, bytearray], None],
    Callable[[BleakGATTCharacteristic, bytearray], None],
]:
    """Return the EMG, IMU, and piezo characteristic handlers."""
    first_emg_sequence_number = 0
    first_imu_sequence_number = 0
    first_piezo_sequence_number = 0

    def emg_notify_handler(
        characteristic: BleakGATTCharacteristic,
        data: bytearray,
    ) -> None:
        i, sensor_data = decode_packet(memoryview(data))
        nonlocal first_emg_sequence_number
        if first_emg_sequence_number == 0:
            first_emg_sequence_number = i
        emg_buf[i - first_emg_sequence_number] = data

    def imu_notify_handler(
        characteristic: BleakGATTCharacteristic,
        data: bytearray,
    ) -> None:
        i, sensor_data = decode_packet(memoryview(data))
        nonlocal first_imu_sequence_number
        if first_imu_sequence_number == 0:
            first_imu_sequence_number = i
        imu_buf[i - first_imu_sequence_number] = data

    def piezo_notify_handler(
        characteristic: BleakGATTCharacteristic,
        data: bytearray,
    ) -> None:
        i, sensor_data = decode_packet(memoryview(data))
        nonlocal first_piezo_sequence_number
        if first_piezo_sequence_number == 0:
            first_piezo_sequence_number = i
        piezo_buf[i - first_piezo_sequence_number] = data

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

    emg_data_received: list[bytearray | None] = [None] * 10000
    imu_data_received: list[bytearray | None] = [None] * 10000
    piezo_data_received: list[bytearray | None] = [None] * 10000

    (emg_notify_handler, imu_notify_handler, piezo_notify_handler) = (
        set_up_notify_handler(emg_data_received, imu_data_received, piezo_data_received)
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

        filtered_emg_data_received = [x for x in emg_data_received if x is not None]
        filtered_imu_data_received = [x for x in imu_data_received if x is not None]
        filtered_piezo_data_received = [x for x in piezo_data_received if x is not None]

        print()
        print_received_packets_stats(filtered_emg_data_received, "EMG")
        print()
        print_received_packets_stats(filtered_imu_data_received, "IMU")
        print()
        print_received_packets_stats(filtered_piezo_data_received, "Piezo")
        print()


if __name__ == "__main__":
    asyncio.run(main())
