"""Deserialize sensor packet data."""

from __future__ import annotations


def decode_packet(view: memoryview[int]) -> tuple[int, memoryview[int]]:
    """Extract the 32-bit sequence number and sensor data from the packet data."""
    return (int.from_bytes(view[:4], "little"), view[4:])
