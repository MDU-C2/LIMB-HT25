from .can_interface import CANMessage
import struct
from typing import Dict, Optional

class CANMessageParser:
    """
    Parser for CAN messages.
    """

    # CAN Message IDs for Prosthetic Arm System
    CAN_IDS = {
        # Upper ARM Transmission
        0x100: {"type": "EMG", "format": "<f"},              # Upper EMG (1 float = 4 bytes)
        
        # RESTORED LEGACY LOGIC (Will expect 24 bytes)
        0x101: {"type": "IMU", "format": "<6f"},             # IMU: [ax, ay, az, gx, gy, gz]
        
        # NEW SPLIT MESSAGES (Used for actual 6-axis data transmission)
        0x105: {"type": "IMU_AXY", "format": "<2f"},         # Upper IMU Pack 1 (accel.x, accel.y = 8 bytes)
        0x106: {"type": "IMU_AZGX", "format": "<2f"},        # Upper IMU Pack 2 (accel.z, gyro.x = 8 bytes)
        0x107: {"type": "IMU_GYGZ", "format": "<2f"},        # Upper IMU Pack 3 (gyro.y, gyro.z = 8 bytes)
        
        # Legacy/Additional sensor messages
        0x102: {"type": "pressure", "format": "<5f"},        # pressure [thumb, index, middle, ring, little]
        0x103: {"type": "piezo", "format": "<f"},            # piezo (float32)
        0x104: {"type": "potentiometer", "format": "<f"},    # potentiometer (float32)
        
        # Actuator messages (to ESP32 nodes)
        0x200: {"type": "gripper_command", "format": "<Bf"},    # [action, force]
        0x201: {"type": "arm_command", "format": "<5f"},        # [joint1, joint2, joint3, joint4, joint5]
        0x202: {"type": "motor_command", "format": "<Bf"},      # Motor control

        # Status messages 
        0x300: {"type": "motor_status", "format": "<5f"},    # [positions]
        0x301: {"type": "gripper_status", "format": "<Bf"},    # [state, force]
    }

    def parse(self, message: CANMessage) -> CANMessage:
        """Parse a CAN message based on its ID."""
        can_id = message.can_id

        # Check if we know this CAN ID:
        if can_id not in self.CAN_IDS:
            return {
                "type": "unknown",
                "data": {"raw": message.data.hex(), "can_id": can_id}
            }
        msg_info = self.CAN_IDS[can_id]
        msg_type = msg_info["type"]
        fmt = msg_info.get("format")

        try:
            if fmt and len(message.data) > 0:

                expected_size = struct.calcsize(fmt)
                # Note: For ID 0x101 sending only 8 bytes, this check will fail 
                # and return the error dict below, which is the expected legacy behavior.
                if len(message.data) != expected_size:
                    return {
                        "type": msg_type,
                        "data": {"error": f"Data too short: got {len(message.data)} bytes, expected {expected_size}", "raw": message.data.hex()}
                    }
                
                parsed_data = struct.unpack(fmt, message.data[:expected_size])
                data_dict = self._format_parsed_data(msg_type, parsed_data)

            else:
                data_dict = {"raw": message.data.hex()}

            return {
                "type": msg_type,
                "data": data_dict
            }

        except struct.error as e:
            return {
                'type': msg_type,
                'data': {'error': f'Parse error: {e}', 'raw': message.data.hex()}
            }

    def _format_parsed_data(self, msg_type, parsed) -> Dict:
        """Format parsed data into a meaningful dictionary."""

        if msg_type == 'EMG':
            return {
                'value': parsed[0],
                'unit': 'mV'
            }
        
        # RESTORED LEGACY LOGIC
        elif msg_type == 'IMU':
            return {
                 'data': list(parsed) # [ax, ay, az, wx, wy, wz]
            }
        
        # NEW SPLIT LOGIC
        elif msg_type == 'IMU_AXY':
            return {
                'accel_x': parsed[0],
                'accel_y': parsed[1],
                'package': '1/3'
            }
        
        elif msg_type == 'IMU_AZGX':
            return {
                'accel_z': parsed[0],
                'gyro_x': parsed[1],
                'package': '2/3'
            }
        
        elif msg_type == 'IMU_GYGZ':
            return {
                'gyro_y': parsed[0],
                'gyro_z': parsed[1],
                'package': '3/3 [RX-ready trigger]'
            }
        
        elif msg_type == 'pressure':
            return {
                "values": list(parsed),
                "finger_count": len(parsed)
            }
        
        elif msg_type == 'piezo':
            return {'value': parsed[0]}
        
        elif msg_type == 'potentiometer':
            return {'value': parsed[0]}
        
        elif msg_type == 'gripper_command':
            return {
                'action': parsed[0],  # 0=open, 1=close, 2=set_force
                'force': parsed[1]
            }
        
        elif msg_type == 'arm_command':
            return {
                'joint_positions': list(parsed)
            }
        
        elif msg_type == 'motor_status':
            return {
                'joint_positions': list(parsed)
            }
        
        elif msg_type == 'gripper_status':
            return {
                'state': parsed[0],
                'force': parsed[1]
            }
        
        else:
            return {'values': list(parsed)}

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
            return None
        
        try:
            if msg_type == "gripper_command":
                values = (data["action"], data["force"])
            elif msg_type == "arm_command":
                values = tuple(data["joint_positions"])
            elif msg_type == "motor_command":
                values = (data["motor_id"], data["value"])
            else:
                return None

            encoded = struct.pack(fmt, *values)
            return (can_id, encoded)

        except (KeyError, struct.error) as e:
            print(f"Encode error for {msg_type}: {e}")
            return None