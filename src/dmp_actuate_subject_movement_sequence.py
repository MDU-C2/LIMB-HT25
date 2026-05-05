import argparse
import sys
import time
from pathlib import Path

from hardware.can.can_message_parser import CANMessageParser
from hardware.can.can_socketcan import SocketCANInterface

sys.path.insert(0, str(Path(__file__).parent / ".." / "dmp"))

from experiment import get_trajectories

can_message_parser = CANMessageParser()


def index_to_name(index: int) -> str:
    """Return CAN message name associated with the provided index."""
    if index == 0:
        return "robot_elbow_up_down_actuation"
    if index == 1:
        return "robot_shoulder_up_down_actuation"
    if index == 2:
        return "robot_shoulder_left_right_actuation"
    if index == 3:
        return "robot_upper_arm_rotation_actuation"
    errmsg = "Error converting index to name"
    raise Exception(errmsg)


def angles_to_can_message_data(angles: list[float]) -> list[tuple[int, bytes]]:
    """Convert angles to CAN data."""
    can_message_data = [
        can_message_parser.encode(index_to_name(i), {"angle": angle, "velocity": 20.0})
        for i, angle in enumerate(angles)
    ]
    if None in can_message_data:
        errmsg = f"Error encoding angle from angles: {angles}"
        raise Exception(errmsg)

    return [x for x in can_message_data if x is not None]


def send_angles_over_can(
    angles_sequence: list[list[float]],
    *,
    command_freq_s: float = 500,
    first_move_sleep_s: float = 10,
) -> None:
    """Send the provided sequence of angles over CAN."""
    actuation_command_data_sequence = [
        angles_to_can_message_data(angles) for angles in angles_sequence
    ]
    send_actuation_commands_over_can(
        actuation_command_data_sequence,
        command_freq_s=command_freq_s,
        first_move_sleep_s=first_move_sleep_s,
    )


def send_actuation_commands_over_can(
    actuation_command_data_sequence: list[list[tuple[int, bytes]]],
    *,
    command_freq_s: float = 0.5,
    first_move_sleep_s: float = 10,
) -> None:
    """Send the provided sequence of CAN data over CAN."""
    with SocketCANInterface(interface="can0", bitrate=1000000) as can_interface:
        time.sleep(1)

        first_move = True
        prev_time_ns = time.monotonic_ns()
        for msgs in actuation_command_data_sequence:
            for can_id, data in msgs:
                print(f"sending id {hex(can_id)}, data {data!r}")
                success = can_interface.send(can_id, data)
                if success:
                    print(
                        f"Sent {can_id} with data {data!r}, tx_count: {can_interface.tx_count}",
                    )
                else:
                    print(f"Error sending {can_id} with data {data!r} over can!")

            if first_move:
                first_move = False
                time.sleep(first_move_sleep_s)
                prev_time_ns = time.monotonic_ns()
            time_ns = time.monotonic_ns()
            expected_new_time = prev_time_ns + int(command_freq_s * 1_000_000_000)
            time_to_sleep_ns = expected_new_time - time_ns
            if time_to_sleep_ns > 0:
                time.sleep(time_to_sleep_ns / 1_000_000_000)
            prev_time_ns = expected_new_time

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Sends a subject's sequence of DMP joint angles to the robot arm's
            microcontrollers over CAN.
        """,
    )
    parser.add_argument(
        "-s",
        "--subject",
        type=int,
        required=True,
        help="The ID of the subject whose movement sequence should be used.",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["unpersonal", "personal", "reset"],
        required=True,
        help="""
            The type of motion sequence to send. Can be the base sequence,
            the personalized version of the sequence, or resetting the arm position.
        """,
    )
    args = parser.parse_args()

    unpersonalized_angles_sequence, personalized_angles_sequence = get_trajectories(
        args.subject,
    )

    if args.type == "unpersonal":
        joint_angles_sequence = unpersonalized_angles_sequence
    elif args.type == "personal":
        joint_angles_sequence = personalized_angles_sequence
    else:
        joint_angles_sequence = [[0, 0, 0, 0]]

    send_angles_over_can(
        joint_angles_sequence,
        command_freq_s=0.5,
        first_move_sleep_s=10,
    )
