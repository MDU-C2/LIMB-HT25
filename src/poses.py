"""A program to make the arm strike one of multiple different poses."""

import argparse
import struct
import time

from hardware.can.can_message_parser import CANMessageParser
from hardware.can.can_socketcan import SocketCANInterface

can_message_parser = CANMessageParser()

THUMB_ID = 0x261
INDEX_ID = 0x262
MIDDLE_ID = 0x263
RING_ID = 0x264
PINKY_ID = 0x265
WRIST_ROTATION = 0x260
ELBOW_ID = 0x240
SHOULDER_UP_DOWN = 0x220
SHOULDER_LEFT_RIGHT = 0x221
BICEPS_ROTATION = 0x222

POSE_TO_ANGLES_MAP: dict[str, dict[int, float]] = {
    "royal_wave_left": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 0,
        ELBOW_ID: 60,
        SHOULDER_UP_DOWN: 110,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
    "royal_wave_right": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 50,
        ELBOW_ID: 60,
        SHOULDER_UP_DOWN: 110,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
    "shake": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 110,
        ELBOW_ID: 50,
        SHOULDER_UP_DOWN: 50,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
    "shake_up": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 110,
        ELBOW_ID: 60,
        SHOULDER_UP_DOWN: 50,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
    "shake_down": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 110,
        ELBOW_ID: 40,
        SHOULDER_UP_DOWN: 45,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
    "grip": {
        THUMB_ID: 100,
        INDEX_ID: 100,
        MIDDLE_ID: 100,
        RING_ID: 100,
        PINKY_ID: 100,
        WRIST_ROTATION: 110,
        ELBOW_ID: 50,
        SHOULDER_UP_DOWN: 50,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
    "peace": {
        THUMB_ID: 90,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 90,
        PINKY_ID: 90,
        WRIST_ROTATION: 20,
        ELBOW_ID: 60,
        SHOULDER_UP_DOWN: 110,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: -7,
    },
    "point": {
        THUMB_ID: 90,
        INDEX_ID: 0,
        MIDDLE_ID: 90,
        RING_ID: 90,
        PINKY_ID: 90,
        WRIST_ROTATION: 110,
        ELBOW_ID: 0,
        SHOULDER_UP_DOWN: 110,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: -7,
    },
    "wave_left": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 20,
        ELBOW_ID: 60,
        SHOULDER_UP_DOWN: 110,
        SHOULDER_LEFT_RIGHT: 5,
        BICEPS_ROTATION: -10,
    },
    "wave_right": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 20,
        ELBOW_ID: 60,
        SHOULDER_UP_DOWN: 110,
        SHOULDER_LEFT_RIGHT: 10,
        BICEPS_ROTATION: 10,
    },
    "wave": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 20,
        ELBOW_ID: 60,
        SHOULDER_UP_DOWN: 110,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
    "neutral": {
        THUMB_ID: 0,
        INDEX_ID: 0,
        MIDDLE_ID: 0,
        RING_ID: 0,
        PINKY_ID: 0,
        WRIST_ROTATION: 0,
        ELBOW_ID: 0,
        SHOULDER_UP_DOWN: 0,
        SHOULDER_LEFT_RIGHT: 0,
        BICEPS_ROTATION: 0,
    },
}


def send_command(
    can_interface: SocketCANInterface,
    angle_map: dict[int, float],
) -> None:
    """Send CAN actuation messages for the provided ID:angle combinations."""
    for can_id, angle in angle_map.items():
        data = struct.pack("<ff", angle, 30)

        success = can_interface.send(can_id, data)
        if success:
            print("success")
        else:
            print("fail")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pose",
        choices=[
            "shake",
            "wave",
            "peace",
            "wave_motion",
            "royal_wave_motion",
            "shake_motion",
            "grip",
            "point",
            "neutral",
        ],
        required=True,
    )
    args = parser.parse_args()
    with SocketCANInterface(interface="can0", bitrate=1000000) as can_interface:
        if args.pose == "shake_motion":
            down_angle_map = POSE_TO_ANGLES_MAP["shake_down"]
            up_angle_map = POSE_TO_ANGLES_MAP["shake_up"]
            current_angle_map = down_angle_map
            for _ in range(1, 6):
                send_command(can_interface, current_angle_map)
                current_angle_map = (
                    up_angle_map
                    if current_angle_map == down_angle_map
                    else down_angle_map
                )
                time.sleep(1)
        elif args.pose == "wave_motion":
            left_angle_map = POSE_TO_ANGLES_MAP["wave_left"]
            right_angle_map = POSE_TO_ANGLES_MAP["wave_right"]
            current_angle_map = left_angle_map
            for _ in range(1, 6):
                send_command(can_interface, current_angle_map)
                current_angle_map = (
                    right_angle_map
                    if current_angle_map == left_angle_map
                    else left_angle_map
                )
                time.sleep(2)
        elif args.pose == "royal_wave_motion":
            left_angle_map = POSE_TO_ANGLES_MAP["royal_wave_left"]
            right_angle_map = POSE_TO_ANGLES_MAP["royal_wave_right"]
            current_angle_map = left_angle_map
            for _ in range(1, 6):
                send_command(can_interface, current_angle_map)
                current_angle_map = (
                    right_angle_map
                    if current_angle_map == left_angle_map
                    else left_angle_map
                )
                time.sleep(2)
        else:
            angle_map = POSE_TO_ANGLES_MAP[args.pose]
            send_command(can_interface, angle_map)
