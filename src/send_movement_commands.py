import time

from hardware.can.can_message_parser import CANMessageParser
from hardware.can.can_socketcan import SocketCANInterface

can_message_parser = CANMessageParser()

# joints order: elbow, shoulder up down, shoulder left right, shoulder twist.
joint_angles_sequence = [
    [60.0, 7.08712333, 10.43174263, 7.77905341],
    [60.0, 7.22737496, 10.49350692, 8.02992856],
    [60.0, 8.0194161, 10.64461526, 8.47082603],
    [60.0, 9.6565202, 10.87830679, 9.02198238],
    [60.0, 12.09642341, 11.17732943, 9.62153845],
    [60.0, 15.15187107, 11.52883574, 10.23390999],
    [60.0, 18.66451713, 11.91044347, 10.79550486],
    [57.75552778, 22.5975942, 12.28216125, 11.18426051],
    [50.6776441, 26.73156062, 12.61400862, 11.33517802],
    [42.38320785, 30.6546675, 12.90466994, 11.25955443],
    [33.64942379, 33.97931376, 13.17136965, 11.03421838],
    [25.4229793, 36.52304268, 13.42004163, 10.79065242],
    [18.73665298, 38.35434118, 13.59658455, 10.67266909],
    [14.48014096, 39.76716804, 13.55378994, 10.81367793],
    [13.05683052, 41.16161278, 13.06154813, 11.31132112],
    [14.23566441, 42.82145176, 11.87245514, 12.21690103],
    [17.30722331, 44.75159829, 9.81128751, 13.53425615],
    [21.37371905, 46.69944236, 6.84407502, 15.20502492],
    [25.5574267, 48.24296839, 3.10602876, 17.10438602],
    [29.09304694, 48.93066837, 0.0, 19.07407085],
    [31.44018396, 48.52735424, 0.0, 20.94326552],
    [32.43760488, 47.17321484, 0.0, 22.53993321],
    [32.34502205, 45.32517459, 0.0, 23.748555],
    [31.73816056, 43.56537456, 0.0, 24.55216556],
    [31.36528292, 42.34017563, 0.0, 24.9735557],
    [31.97256366, 41.725938, 0.0, 24.97306102],
    [34.06701575, 41.36563618, 0.0, 24.40038255],
    [37.7160221, 40.60248553, 0.0, 23.04325997],
    [42.49904714, 38.74240976, 0.0, 20.74755371],
    [47.6408686, 35.3262542, 1.90217047, 17.50056786],
    [52.25602005, 30.29315458, 6.73282443, 13.46134492],
    [55.57914195, 24.02248398, 11.23215791, 8.99764251],
    [57.16968532, 17.28056119, 14.85932001, 4.64212556],
    [57.07058387, 11.03529809, 17.22218568, 0.91311412],
    [55.77878779, 6.16558369, 18.21285353, -1.86052913],
    [54.07461835, 3.21715133, 18.03442382, -3.59501813],
    [52.82917252, 2.2773176, 17.10470245, -4.42110107],
    [52.71105068, 2.95828106, 15.88053639, -4.63058233],
    [53.89680348, 4.50000063, 14.70053276, -4.59332022],
    [56.07453695, 6.0065951, 13.72349306, -4.63185162],
    [58.69379391, 6.78506705, 12.96561177, -4.9125524],
    [60.0, 6.6084234, 12.37916512, -5.43244198],
    [60.0, 5.7125264, 11.91143872, -6.09347559],
    [60.0, 4.57710286, 11.53208529, -6.76890341],
    [60.0, 3.70137526, 11.24059471, -7.32958275],
    [60.0, 3.46219009, 11.05118547, -7.68299933],
    [60.0, 4.00760441, 10.97291799, -7.79661757],
    [60.0, 5.23819307, 11.00973687, -7.65787583],
    [60.0, 6.87291078, 11.15813678, -7.26072868],
    [60.0, 8.45922081, 11.38651509, -6.66708781],
    [60.0, 9.45427987, 11.63144608, -6.0276681],
    [60.0, 9.51810435, 11.8249338, -5.51408128],
    [60.0, 8.76456998, 11.92300976, -5.24070927],
    [60.0, 7.73688264, 11.9174086, -5.21830496],
    [60.0, 7.13403779, 11.83080333, -5.3565169],
    [60.0, 7.42696267, 11.69847899, -5.53481045],
    [60.0, 8.60703796, 11.54906773, -5.68087156],
    [60.0, 10.250157, 11.39987453, -5.77792042],
    [60.0, 11.78545607, 11.26259068, -5.82622956],
    [60.0, 12.72652405, 11.14782926, -5.81404933],
    [60.0, 12.72843934, 11.06632293, -5.71714083],
    [60.0, 11.55248556, 11.02839012, -5.51781256],
    [60.0, 8.90567193, 11.03560781, -5.24521354],
    [60.0, 4.53748971, 11.08684735, -4.93911872],
    [60.0, 0.0, 11.19480328, -4.5794116],
]


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
    send_angles_over_can(
        joint_angles_sequence,
        command_freq_s=0.5,
        first_move_sleep_s=10,
    )
