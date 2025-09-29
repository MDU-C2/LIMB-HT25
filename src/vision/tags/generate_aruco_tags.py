import argparse
import os
from typing import List

import cv2
import numpy as np


def _resolve_dictionary(name: str) -> int:
    # Map common names (case-insensitive) to cv2.aruco constants
    name_upper = name.strip().upper()
    if hasattr(cv2.aruco, name_upper):
        return getattr(cv2.aruco, name_upper)
    # Friendly aliases
    aliases = {
        "DICT_4X4_50": "DICT_4X4_50",
        "4X4_50": "DICT_4X4_50",
        "DICT_5X5_50": "DICT_5X5_50",
        "5X5_50": "DICT_5X5_50",
        "DICT_6X6_50": "DICT_6X6_50",
        "6X6_50": "DICT_6X6_50",
        "DICT_ARUCO_ORIGINAL": "DICT_ARUCO_ORIGINAL",
        "ARUCO_ORIGINAL": "DICT_ARUCO_ORIGINAL",
    }
    resolved = aliases.get(name_upper)
    if resolved and hasattr(cv2.aruco, resolved):
        return getattr(cv2.aruco, resolved)
    raise ValueError(f"Unknown ArUco dictionary: {name}")


def parse_ids(ids_arg: str) -> List[int]:
    # Supports: "0,1,2" or "0-9" or mixed "0-3,7,9-12"
    result: List[int] = []
    for part in ids_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            if end < start:
                start, end = end, start
            result.extend(list(range(start, end + 1)))
        else:
            result.append(int(part))
    # Deduplicate and sort
    return sorted(list(dict.fromkeys(result)))


def save_marker_jpg(
    dictionary_id: int,
    marker_id: int,
    size_px: int,
    border_bits: int,
    output_path: str,
) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    img = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, marker_id, size_px, img, borderBits=border_bits)
    cv2.imwrite(output_path, img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ArUco marker JPG files.")
    parser.add_argument(
        "--dict",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV aruco dictionary (e.g., DICT_4X4_50, DICT_5X5_50, DICT_ARUCO_ORIGINAL)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="0-9",
        help="IDs to generate, e.g. '0-9' or '0,1,7,10-12'",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="Marker size in pixels (square image)",
    )
    parser.add_argument(
        "--border-bits",
        type=int,
        default=1,
        help="Black border width in bits around marker (OpenCV aruco parameter)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "generated_tags"),
        help="Directory to store generated JPGs (will be created if missing)",
    )
    args = parser.parse_args()

    dictionary_id = _resolve_dictionary(args.dict)
    ids = parse_ids(args.ids)

    os.makedirs(args.outdir, exist_ok=True)

    for marker_id in ids:
        filename = f"{args.dict}_id_{marker_id}_{args.size}px.jpg"
        output_path = os.path.join(args.outdir, filename)
        save_marker_jpg(dictionary_id, marker_id, args.size, args.border_bits, output_path)

    print(f"Generated {len(ids)} tag(s) in: {args.outdir}")


if __name__ == "__main__":
    main()


