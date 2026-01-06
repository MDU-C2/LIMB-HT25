from re import M
from .can_interface import CANMessage
import struct
from typing import Dict, Optional

class CANMessageParser:
    """
    Parser for CAN messages.
    """

    # CAN IDs matching can_driver.h CanMessageId enum exactly
    CAN_IDS = {
        # Stop messages
        0x120: {"type": "robot_shoulder_up_down_stop", "format": None},
        0x121: {"type": "robot_shoulder_left_right_stop", "format": None},
        0x122: {"type": "robot_upper_arm_rotation_stop", "format": None},
        0x140: {"type": "robot_elbow_up_down_stop", "format": None},
        0x160: {"type": "robot_lower_arm_rotation_stop", "format": None},
        0x161: {"type": "robot_fingers_stop", "format": None},
        0x162: {"type": "robot_thumb_stop", "format": None},
        0x163: {"type": "robot_index_stop", "format": None},
        0x164: {"type": "robot_middle_stop", "format": None},
        0x165: {"type": "robot_ring_stop", "format": None},
        0x166: {"type": "robot_pinky_stop", "format": None},

        # Actuation messages
        0x220: {"type": "robot_shoulder_up_down_actuation", "format": "<f"},  # float32: position/velocity
        0x221: {"type": "robot_shoulder_left_right_actuation", "format": "<f"},
        0x222: {"type": "robot_upper_arm_rotation_actuation", "format": "<f"},
        0x240: {"type": "robot_elbow_up_down_actuation", "format": "<f"},
        0x260: {"type": "robot_lower_arm_rotation_actuation", "format": "<f"},
        0x261: {"type": "robot_thumb_actuation", "format": "<f"},
        0x262: {"type": "robot_index_actuation", "format": "<f"},
        0x263: {"type": "robot_middle_actuation", "format": "<f"},
        0x264: {"type": "robot_ring_actuation", "format": "<f"},
        0x265: {"type": "robot_pinky_actuation", "format": "<f"},
        0x266: {"type": "robot_hand_set_grip_state", "format": "<Bf"},  # [state (uint8), force (float32)]

        # Potentiometer messages
        0x4A0: {"type": "robot_elbow_up_down_potentiometer", "format": "<f"},
        0x4A1: {"type": "robot_upper_arm_rotation_potentiometer", "format": "<f"},
        0x4A2: {"type": "robot_shoulder_up_down_potentiometer", "format": "<f"},
        0x4A3: {"type": "robot_shoulder_left_right_potentiometer", "format": "<f"},

        # IMU messages (robot)
        0x5A0: {"type": "robot_shoulder_imu_gyro", "format": "<3f"},  # [gx, gy, gz]
        0x5A1: {"type": "robot_shoulder_imu_accel", "format": "<3f"},  # [ax, ay, az]
        0x5A2: {"type": "robot_elbow_imu_gyro", "format": "<3f"},
        0x5A3: {"type": "robot_elbow_imu_accel", "format": "<3f"},
        0x5A4: {"type": "robot_hand_imu_gyro", "format": "<3f"},
        0x5A5: {"type": "robot_hand_imu_accel", "format": "<3f"},

        # Pressure sensor messages
        0x7A0: {"type": "robot_thumb_pressure", "format": "<f"},
        0x7A1: {"type": "robot_index_pressure", "format": "<f"},
        0x7A2: {"type": "robot_middle_pressure", "format": "<f"},
        0x7A3: {"type": "robot_ring_pressure", "format": "<f"},
        0x7A4: {"type": "robot_pinky_pressure", "format": "<f"},

        # Human EMG message
        0x3C0: {"type": "human_upper_arm_emg", "format": "<2f"},  # 2 EMG channels

        # Human IMU messages
        0x5C0: {"type": "human_upper_arm_imu_gyro", "format": "<3f"},  # [gx, gy, gz]
        0x5C1: {"type": "human_upper_arm_imu_accel", "format": "<3f"},  # [ax, ay, az]
    }

    def parse(self, message: CANMessage) -> CANMessage:
        """Parse a CAN message based on its ID."""
        can_id = message.can_id

        # Check if we know this CAN ID
        if can_id not in self.CAN_IDS:
            return {
                "message_type": "unknown",
                "parsed_data": {"raw": message.data.hex(), "can_id": can_id}
            }

        msg_info = self.CAN_IDS[can_id]
        msg_type = msg_info["type"]
        fmt = msg_info.get("format")

        try:
            if fmt and len(message.data) > 0:
                expected_size = struct.calcsize(fmt)
                if len(message.data) != expected_size:
                    return {
                        "message_type": msg_type,
                        "parsed_data": {
                            "error": f"Data size mismatch: got {len(message.data)} bytes, expected {expected_size}",
                            "raw": message.data.hex()
                        }
                    }
                parsed_data = struct.unpack(fmt, message.data[:expected_size])
                data_dict = self._format_parsed_data(msg_type, parsed_data)
            else:
                # No format specified, (e.g. stop messages) - return raw data
                data_dict = {"raw": message.data.hex()}

            return {
                "message_type": msg_type,
                "parsed_data":data_dict
            }
        except struct.error as e:
            return {
                "message_type": msg_type,
                "parsed_data": {"error": f"Parse error {e}", "raw": message.data.hex()}
            }

    def _format_parsed_data(self, msg_type, parsed) -> Dict:
        """Format parsed data into a meaningful dictionary."""

        # Human EMG
        if msg_type == "human_upper_arm_emg":
            return {
                "channels": list(parsed), # [ch0] for single channel, [ch0, ch1] for dual channel
                "channel_count": len(parsed)
            }
        
        # Human IMU
        elif msg_type == "human_upper_arm_imu_gyro":
            return {
                "data": list(parsed), # [gx, gy, gz]
                "type": "gyro"
            }
        elif msg_type == "human_upper_arm_imu_accel":
            return {
                "data": list(parsed), # [ax, ay, az]
                "type": "accel"
            }
        
        # Robot IMU
        elif msg_type.endswith("_imu_gyro"):
            return {
                "data": list(parsed), # [gx, gy, gz]
                "type": "gyro",
                "source": msg_type.replace("_imu_gyro", "")
            }

        elif msg_type.endswith("_imu_accel"):
            return {
                "data": list(parsed), # [ax, ay, az]
                "type": "accel",
                "source": msg_type.replace("_imu_accel", "")
            }

        # Pressure sensors
        elif msg_type.endswith("_pressure"):
            finger_name = msg_type.replace("robot_", "").replace("_pressure", "")
            return {
                "value": parsed[0], # single float value
                "finger": finger_name # thumb, index, middle, ring, pinky
            }
        
        # Potentiometer messages
        elif "potentiometer" in msg_type:
            return {
                "value": parsed[0], # single float value
                "source": msg_type.replace("robot_", "").replace("_potentiometer", "")
            }

        # Actuation messages
        elif "actuation" in msg_type:
            if msg_type == "robot_hand_set_grip_state":
                return {
                    "state": parsed[0], # uint8, 0 = open, 1 = close
                    "force": parsed[1] # float32 value between 0.0 and 1.0
                }
            else:
                return {
                    "value": parsed[0], # single float value
                    "source": msg_type.replace("robot_", "").replace("_actuation", "")
                }
        
        # Stop messages (no data, just acknowledgement)
        elif "stop" in msg_type:
            return {
                "acknowledged": True,
                "actuator": msg_type.replace("robot_", "").replace("_stop", "")
            }
        
        else:
            return {"values": list(parsed)}

    def encode(self, msg_type: str, data: Dict) -> Optional[tuple]:
        """Encode data dict into CAN message format."""

        can_id = None
        for id_val, info in self.CAN_IDS.items():
            if info["type"] == msg_type:
                can_id = id_val
                break
        
        if can_id is None:
            return None

        fmt = self.CAN_IDS[can_id].get("format")
        if not fmt:
            return (can_id, b"") # Stop messages or messages without format
        
        try:
            if msg_type == "robot_hand_set_grip_state":
                values = (data["state"], data["force"])
            elif "actuation" in msg_type:
                values = (data["value"],)
            elif "pressure" in msg_type:
                values = (data["value"],)
            elif "potentiometer" in msg_type:
                values = (data["value"],)
            elif msg_type.endswith("_imu_gyro") or msg_type.endswith("_imu_accel"):
                values = tuple(data["data"])
            elif msg_type == "human_upper_arm_emg":
                values = tuple(data["channels"])
            else:
                return None

            encoded = struct.pack(fmt, *values)
            return (can_id, encoded)

        except (KeyError, struct.error) as e:
            print(f"Encode error for {msg_type}: {e}")
            return None
        
