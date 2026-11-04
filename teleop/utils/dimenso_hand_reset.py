"""Decide when the Quest episode reset may open BrainCo hands.

The Quest loop applies this decision to controller inputs. This module never
publishes motor commands or changes robot mode.
"""


def release_brainco_at_home(*, phase: str, home_ready: bool, end_effector: str, input_mode: str) -> bool:
    """Open only after a saved/discarded take has returned the arms home."""
    return (phase in ("awaiting_reset", "complete") and home_ready
            and end_effector == "brainco" and input_mode == "controller")
