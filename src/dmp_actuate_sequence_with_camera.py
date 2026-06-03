import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hardware.can.can_message_parser import CANMessageParser
from hardware.can.can_socketcan import SocketCANInterface

sys.path.insert(0, str(Path(__file__).parent / ".." / "dmp"))

from experiments.classical_dmp_timing_api import (
    ClassicalDMPTimingBudgetsMs,
    ClassicalDMPTimingConfig,
    run_classical_dmp_timing_experiment,
)

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


def send_can_msg(joint_angles: np.array, can_interface: Any) -> bool:
    can_messages = angles_to_can_message_data(joint_angles)

    ret = True
    for i, (can_id, data) in enumerate(can_messages):
        success = can_interface.send(can_id, data)
        if success:
            print(
                f"Sent {can_id} with angle {joint_angles[i]}, data {data!r}, tx_count: {can_interface.tx_count}",
            )
        else:
            ret = False
            print(f"Error sending {can_id} with data {data!r} over can!")
    return ret


def main() -> None:
    """Run the program."""
    parser = argparse.ArgumentParser(
        description="""
            Sends a sequence of DMP joint angles to the robot arm's microcontrollers
            over CAN based on where a subject's position is along the sequence as
            recognized by a depth camera.
        """,
    )
    parser.add_argument(
        "-s",
        "--result-file-suffix",
        default="",
        help="A file suffix to add to the result files.",
    )
    parser.add_argument(
        "-p",
        "--personalized-curvature-weights-file",
        help="Path to personalized curvature weights json file.",
    )
    args = parser.parse_args()

    trial_dir = Path("../trial_005")

    personalized_curve_weights = None
    if args.personalized_curvature_weights_file:
        with Path(args.personalized_curvature_weights_file).open() as f:
            personalized_curve_weights = json.load(f)["curvature_weights"]

    config = ClassicalDMPTimingConfig(
        phase_mode="path-progress",
        comm_mode="can",
        period_ms=200,
        n_iters=150,
    )

    # Based on measurements made on an NVIDIA Jetson AGX Orin Developer kit.
    budgets = ClassicalDMPTimingBudgetsMs(
        pose_ms=6.0,
        preprocess_ms=13.0,
        angle_ms=5.0,
        phase_ms=0.5,
        dmp_step_ms=0.3,
        comm_ms=2.0,
        e2e_ms=27.0,
    )

    with SocketCANInterface(interface="can0", bitrate=1_000_000) as can_interface:
        time.sleep(1)
        run_classical_dmp_timing_experiment(
            trial_dir,
            "./",
            config,
            budgets,
            send_can_msg,
            can_interface,
            personalized_curve_weights,
            args.result_file_suffix,
        )


if __name__ == "__main__":
    main()
