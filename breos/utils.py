"""
Utility functions for breos library.
"""

import multiprocessing
import os


def is_leap_year(year: int) -> bool:
    """
    Check if a year is a leap year.
    
    Args:
        year: Year to check
        
    Returns:
        True if leap year, False otherwise
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def count_leap_years(start_year: int, num_years: int) -> int:
    """
    Count the number of leap years in a range.
    
    Args:
        start_year: Starting year
        num_years: Number of years to count
        
    Returns:
        Number of leap years in the range
    """
    return sum(1 for year in range(start_year, start_year + num_years) if is_leap_year(year))


def number_of_cores() -> int:
    """
    Get the number of available CPU cores for parallel processing.
    
    Returns:
        Number of CPU cores (leaves 1 core free for system)
    """
    total_cores = multiprocessing.cpu_count()
    # Leave at least 1 core for system, use at least 1 for computation
    return max(1, total_cores - 1)


def get_hours_per_step(freq: str) -> float:
    """
    Get the number of hours per timestep based on frequency.
    
    Args:
        freq: Frequency string ('h' for hourly, '15min' for 15-minute)
        
    Returns:
        Hours per timestep (1.0 for hourly, 0.25 for 15-min)
        
    Raises:
        ValueError: If freq is not recognized
    """
    freq_map = {
        'h': 1.0,
        'H': 1.0,
        '1h': 1.0,
        '1H': 1.0,
        '15min': 0.25,
        '15T': 0.25,
        '15m': 0.25,
    }
    if freq not in freq_map:
        raise ValueError(f"Unsupported frequency: {freq}. Use 'h' or '15min'.")
    return freq_map[freq]


def get_steps_per_day(freq: str) -> int:
    """
    Get the number of timesteps per day based on frequency.
    
    Args:
        freq: Frequency string ('h' for hourly, '15min' for 15-minute)
        
    Returns:
        Steps per day (24 for hourly, 96 for 15-min)
    """
    hours_per_step = get_hours_per_step(freq)
    return int(24 / hours_per_step)


def get_steps_per_year(freq: str, leap_year: bool = False) -> int:
    """
    Get the number of timesteps per year based on frequency.
    
    Args:
        freq: Frequency string ('h' for hourly, '15min' for 15-minute)
        leap_year: Whether to account for leap year (366 days)
        
    Returns:
        Steps per year (8760/8784 for hourly, 35040/35136 for 15-min)
    """
    days = 366 if leap_year else 365
    return get_steps_per_day(freq) * days
