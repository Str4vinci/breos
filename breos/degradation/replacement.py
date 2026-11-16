"""Horizon guards shared by stationary and vehicle pack replacement."""


def replacement_skip_reason(
    *,
    allow_terminal_replacement: bool,
    final_period: bool,
    remaining_steps: int,
    min_remaining_steps: float,
) -> str | None:
    """Name the first rule that skips a swap, or allow it with None."""
    if remaining_steps < min_remaining_steps:
        return "min_remaining_years"
    if final_period and not allow_terminal_replacement:
        return "terminal_period"
    return None
