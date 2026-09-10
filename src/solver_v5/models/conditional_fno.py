from __future__ import annotations

from src.solver_v3.models.fno import FNO2d


class ConditionalFNO2d(FNO2d):
    """FNO whose explicit input is condition, two states, difference and time."""

    def __init__(self, state_channels: int, condition_channels: int, width: int, modes: int, depth: int):
        self.state_channels = state_channels
        self.condition_channels = condition_channels
        super().__init__(condition_channels + 3 * state_channels + 1, state_channels, width, modes, depth)
