"""
Robot Control System

This module provides the main control system for robotic manipulation tasks.
It implements the state machine for cup manipulation based on the flowchart.
"""

from .state_machine import StateMachine
from .main import main

__all__ = [
    'StateMachine',
    'main'
]
