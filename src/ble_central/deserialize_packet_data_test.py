"""Test deserialize_packet_data with mock EMG, IMU, and piezo packet data."""

import struct
import unittest

from sensor_packet_serialization import ValueDataType, deserialize_packet_data


class TestDeserializePacketData(unittest.TestCase):
    """Test deserialize_packet_data with mock EMG, IMU, and piezo packet data."""

    emg_bytes_per_value = 2
    emg_values_per_sample = 1
    emg_sensor_count = 2
    imu_bytes_per_value = 4
    imu_values_per_sample = 6
    imu_sensor_count = 2
    piezo_bytes_per_value = 2
    piezo_values_per_sample = 1
    piezo_sensor_count = 1

    emg_le_data = bytes([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0])
    emg_be_data = bytes([0, 1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128])
    imu_le_data = struct.pack("<24f", *[float(value) for value in range(1, 25)])
    imu_scaled_le_data = struct.pack(
        "<24f",
        *[float(value * 256) for value in range(1, 25)],
    )
    imu_negative_le_data = struct.pack(
        "<24f",
        *[float(-value) for value in range(1, 25)],
    )
    imu_be_data = struct.pack(
        ">24f",
        *[float(value) for value in range(1, 25)],
    )
    imu_negative_be_data = struct.pack(
        ">24f",
        *[float(-value) for value in range(1, 25)],
    )
    piezo_le_data = bytes([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0])
    piezo_be_data = bytes([0, 1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128])

    def test_little_endian_emg(self) -> None:
        """Test deserialize_packet_data with mock EMG data in little endian format."""
        result = deserialize_packet_data(
            memoryview(self.emg_le_data),
            bytes_per_value=self.emg_bytes_per_value,
            values_per_sample=self.emg_values_per_sample,
            sensor_count=self.emg_sensor_count,
            value_data_type=ValueDataType.UNSIGNED_INTEGER,
        )
        target = [
            [[1], [3], [5], [7]],
            [[2], [4], [6], [8]],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_big_endian_emg(self) -> None:
        """Test deserialize_packet_data with mock EMG data in big endian format."""
        result = deserialize_packet_data(
            memoryview(self.emg_be_data),
            bytes_per_value=self.emg_bytes_per_value,
            values_per_sample=self.emg_values_per_sample,
            sensor_count=self.emg_sensor_count,
            value_data_type=ValueDataType.UNSIGNED_INTEGER,
        )
        target = [
            [[256], [1024], [4096], [16384]],
            [[512], [2048], [8192], [32768]],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_ill_formed_emg(self) -> None:
        """Test deserialize_packet_data with ill-formed EMG data."""
        with self.assertRaises(ValueError):
            # Add an extra byte.
            deserialize_packet_data(
                memoryview(self.emg_le_data + b"0"),
                bytes_per_value=self.emg_bytes_per_value,
                values_per_sample=self.emg_values_per_sample,
                sensor_count=self.emg_sensor_count,
                value_data_type=ValueDataType.UNSIGNED_INTEGER,
            )

    def test_little_endian_imu(self) -> None:
        """Test deserialize_packet_data with mock IMU data in little endian format."""
        result = deserialize_packet_data(
            memoryview(self.imu_le_data),
            bytes_per_value=self.imu_bytes_per_value,
            values_per_sample=self.imu_values_per_sample,
            sensor_count=self.imu_sensor_count,
            value_data_type=ValueDataType.FLOATING_POINT,
        )
        target = [
            [[1, 2, 3, 4, 5, 6], [13, 14, 15, 16, 17, 18]],
            [[7, 8, 9, 10, 11, 12], [19, 20, 21, 22, 23, 24]],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_scaled_little_endian_imu(self) -> None:
        """Test deserialize_packet_data with scaled float IMU data in little endian format."""
        result = deserialize_packet_data(
            memoryview(self.imu_scaled_le_data),
            bytes_per_value=self.imu_bytes_per_value,
            values_per_sample=self.imu_values_per_sample,
            sensor_count=self.imu_sensor_count,
            value_data_type=ValueDataType.FLOATING_POINT,
        )

        # Each value is the sequence 1..24 scaled by 256.
        target = [
            [
                [256, 512, 768, 1024, 1280, 1536],
                [3328, 3584, 3840, 4096, 4352, 4608],
            ],
            [
                [1792, 2048, 2304, 2560, 2816, 3072],
                [4864, 5120, 5376, 5632, 5888, 6144],
            ],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_negative_little_endian_imu(self) -> None:
        """Test deserialize_packet_data with mock negative IMU data in little endian format."""
        result = deserialize_packet_data(
            memoryview(self.imu_negative_le_data),
            bytes_per_value=self.imu_bytes_per_value,
            values_per_sample=self.imu_values_per_sample,
            sensor_count=self.imu_sensor_count,
            value_data_type=ValueDataType.FLOATING_POINT,
        )

        target = [
            [[-1, -2, -3, -4, -5, -6], [-13, -14, -15, -16, -17, -18]],
            [[-7, -8, -9, -10, -11, -12], [-19, -20, -21, -22, -23, -24]],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_big_endian_imu(self) -> None:
        """Test deserialize_packet_data with big endian float IMU byte order."""
        result = deserialize_packet_data(
            memoryview(self.imu_be_data),
            bytes_per_value=self.imu_bytes_per_value,
            values_per_sample=self.imu_values_per_sample,
            sensor_count=self.imu_sensor_count,
            value_data_type=ValueDataType.FLOATING_POINT,
        )

        expected_flat = list(struct.unpack("<24f", self.imu_be_data))
        target = [
            [expected_flat[i : i + self.imu_values_per_sample] for i in (0, 12)],
            [expected_flat[i : i + self.imu_values_per_sample] for i in (6, 18)],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_negative_big_endian_imu(self) -> None:
        """Test deserialize_packet_data with negative big endian float IMU byte order."""
        result = deserialize_packet_data(
            memoryview(self.imu_negative_be_data),
            bytes_per_value=self.imu_bytes_per_value,
            values_per_sample=self.imu_values_per_sample,
            sensor_count=self.imu_sensor_count,
            value_data_type=ValueDataType.FLOATING_POINT,
        )

        expected_flat = list(struct.unpack("<24f", self.imu_negative_be_data))
        target = [
            [expected_flat[i : i + self.imu_values_per_sample] for i in (0, 12)],
            [expected_flat[i : i + self.imu_values_per_sample] for i in (6, 18)],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_ill_formed_imu(self) -> None:
        """Test deserialize_packet_data with ill-formed IMU data."""
        with self.assertRaises(ValueError):
            # Add an extra byte.
            deserialize_packet_data(
                memoryview(self.imu_le_data + b"0"),
                bytes_per_value=self.imu_bytes_per_value,
                values_per_sample=self.imu_values_per_sample,
                sensor_count=self.imu_sensor_count,
                value_data_type=ValueDataType.FLOATING_POINT,
            )

    def test_imu_packet_sample_count_not_divisible_by_sensor_count(self) -> None:
        """Test deserialize_packet_data with ill-formed IMU data."""
        with self.assertRaisesRegex(ValueError, "must be divisible by sensor count"):
            # Add an extra byte.
            deserialize_packet_data(
                memoryview(self.imu_le_data),
                bytes_per_value=self.imu_bytes_per_value,
                values_per_sample=self.imu_values_per_sample,
                sensor_count=self.imu_sensor_count + 1,
                value_data_type=ValueDataType.FLOATING_POINT,
            )

    def test_little_endian_piezo(self) -> None:
        """Test deserialize_packet_data with mock piezo data in little endian format."""
        result = deserialize_packet_data(
            memoryview(self.piezo_le_data),
            bytes_per_value=self.piezo_bytes_per_value,
            values_per_sample=self.piezo_values_per_sample,
            sensor_count=self.piezo_sensor_count,
            value_data_type=ValueDataType.UNSIGNED_INTEGER,
        )
        target = [
            [[1], [2], [3], [4], [5], [6], [7], [8]],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_big_endian_piezo(self) -> None:
        """Test deserialize_packet_data with mock piezo data in big endian format."""
        result = deserialize_packet_data(
            memoryview(self.piezo_be_data),
            bytes_per_value=self.piezo_bytes_per_value,
            values_per_sample=self.piezo_values_per_sample,
            sensor_count=self.piezo_sensor_count,
            value_data_type=ValueDataType.UNSIGNED_INTEGER,
        )
        target = [
            [[256], [512], [1024], [2048], [4096], [8192], [16384], [32768]],
        ]
        self.assertEqual([x.tolist() for x in result], target)

    def test_ill_formed_piezo(self) -> None:
        """Test deserialize_packet_data with ill-formed piezo data."""
        with self.assertRaises(ValueError):
            # Add an extra byte.
            deserialize_packet_data(
                memoryview(self.piezo_le_data + b"0"),
                bytes_per_value=self.piezo_bytes_per_value,
                values_per_sample=self.piezo_values_per_sample,
                sensor_count=self.piezo_sensor_count,
                value_data_type=ValueDataType.UNSIGNED_INTEGER,
            )


if __name__ == "__main__":
    unittest.main()
