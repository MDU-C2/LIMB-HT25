"""Sends provided joint angles to specified motors."""

import argparse
import sys
from pprint import pprint

from hardware.can.can_message_parser import CANMessageParser
from hardware.can.can_socketcan import SocketCANInterface

can_message_parser = CANMessageParser()


robot_shoulder_up_down_actuation = "robot_shoulder_up_down_actuation"
robot_shoulder_left_right_actuation = "robot_shoulder_left_right_actuation"
robot_upper_arm_rotation_actuation = "robot_upper_arm_rotation_actuation"
robot_elbow_up_down_actuation = "robot_elbow_up_down_actuation"
robot_lower_arm_rotation_actuation = "robot_lower_arm_rotation_actuation"
robot_thumb_actuation = "robot_thumb_actuation"
robot_index_actuation = "robot_index_actuation"
robot_middle_actuation = "robot_middle_actuation"
robot_ring_actuation = "robot_ring_actuation"
robot_pinky_actuation = "robot_pinky_actuation"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Sends the specified joint angles to the robot arm's
            microcontrollers over CAN.
        """,
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset all joints to their neutral position (0).",
    )
    parser.add_argument(
        "-e",
        "--elbow",
        type=float,
        help="The joint angle (flexion/extension) to send to the elbow motor.",
    )
    parser.add_argument(
        "-slr",
        "--shoulder-left-right",
        type=float,
        help="The joint angle (abduction/adduction) to send to the shoulder left/right motor.",
    )
    parser.add_argument(
        "-sud",
        "--shoulder-up-down",
        type=float,
        help="The joint angle (flexion/extension) to send to the shoulder up/down motor.",
    )
    parser.add_argument(
        "-uar",
        "--upper-arm-rotate",
        type=float,
        help="The joint angle (medial/lateral) to send to the upper arm rotation motor.",
    )
    parser.add_argument(
        "-lar",
        "--lower-arm-rotate",
        type=float,
        help="The joint angle (supination/pronation) to send to the lower arm rotation motor.",
    )
    parser.add_argument(
        "-t",
        "--thumb",
        type=float,
        help="The joint angle (flexion/extension) to send to the thumb motor.",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=float,
        help="The joint angle (flexion/extension) to send to the index motor.",
    )
    parser.add_argument(
        "-m",
        "--middle",
        type=float,
        help="The joint angle (flexion/extension) to send to the middle motor.",
    )
    parser.add_argument(
        "-r",
        "--ring",
        type=float,
        help="The joint angle (flexion/extension) to send to the ring motor.",
    )
    parser.add_argument(
        "-p",
        "--pinky",
        type=float,
        help="The joint angle (flexion/extension) to send to the pinky motor.",
    )
    args = parser.parse_args()

    angles_to_send: list[tuple[str, float]] = []

    if args.shoulder_up_down is not None:
        angles_to_send.append((robot_shoulder_up_down_actuation, args.shoulder_up_down))
    if args.shoulder_left_right is not None:
        angles_to_send.append(
            (robot_shoulder_left_right_actuation, args.shoulder_left_right),
        )
    if args.elbow is not None:
        angles_to_send.append((robot_elbow_up_down_actuation, args.elbow))
    if args.upper_arm_rotate is not None:
        angles_to_send.append(
            (robot_upper_arm_rotation_actuation, args.upper_arm_rotate),
        )
    if args.lower_arm_rotate is not None:
        angles_to_send.append(
            (robot_lower_arm_rotation_actuation, args.lower_arm_rotate),
        )
    if args.thumb is not None:
        angles_to_send.append((robot_thumb_actuation, args.thumb))
    if args.index is not None:
        angles_to_send.append((robot_index_actuation, args.index))
    if args.middle is not None:
        angles_to_send.append((robot_middle_actuation, args.middle))
    if args.ring is not None:
        angles_to_send.append((robot_ring_actuation, args.ring))
    if args.pinky is not None:
        angles_to_send.append((robot_pinky_actuation, args.pinky))

    if args.reset_all:
        if len(angles_to_send) != 0:
            print("--reset-all was set, but angles have also been set.")
            sys.exit()
        angles_to_send = [
            (robot_shoulder_up_down_actuation, 0),
            (robot_shoulder_left_right_actuation, 0),
            (robot_upper_arm_rotation_actuation, 0),
            (robot_elbow_up_down_actuation, 0),
            (robot_lower_arm_rotation_actuation, 0),
            (robot_thumb_actuation, 0),
            (robot_index_actuation, 0),
            (robot_middle_actuation, 0),
            (robot_ring_actuation, 0),
            (robot_pinky_actuation, 0),
        ]

    if len(angles_to_send) == 0:
        print("You need to choose an angle to send for some joint")
        sys.exit()

    maybe_can_msgs: list[tuple[int, bytes] | None] = [
        can_message_parser.encode(msg_name, {"angle": angle, "velocity": 10.0})
        for msg_name, angle in angles_to_send
    ]
    if None in maybe_can_msgs:
        print("Error: A CAN message couldn't be parsed")
        paired = list(zip(angles_to_send, maybe_can_msgs, strict=True))
        print("What follow are the names+angles along with what they are parsed as:")
        pprint(paired)
        sys.exit()

    can_msgs: list[tuple[int, bytes]] = [
        can_msg for can_msg in maybe_can_msgs if can_msg is not None
    ]

    print("Sending angles:")
    pprint(angles_to_send)

    with SocketCANInterface(interface="can0", bitrate=1000000) as can_interface:
        for can_id, data in can_msgs:
            print(f"sending id {hex(can_id)}, data {data!r}")
            success = can_interface.send(can_id, data)
            if success:
                print(
                    f"Sent {can_id} with data {data!r}, tx_count: {can_interface.tx_count}",
                )
            else:
                print(f"Error sending {can_id} with data {data!r} over can!")
