from __future__ import annotations

import calendar
from typing import Tuple, List

import numpy as np

from .tokenizer import (
    IDX_TO_DAY, IDX_TO_MONTH,
    decade_idx_to_token,
    denormalise_date,
    format_output_line,
    parse_condition_line,
    DATE_OFFSETS, DATE_RANGES,
)


# Date validity helpers 

def is_valid_date(day: int, month: int, year: int) -> bool:
    # Return True when (day, month, year) is a real calendar date."""
    if year < 1800 or year > 2200:
        return False
    if month < 1 or month > 12:
        return False
    max_day = calendar.monthrange(year, month)[1]
    return 1 <= day <= max_day


def day_of_week_index(day: int, month: int, year: int) -> int:
    import datetime
    return datetime.date(year, month, day).weekday()


def is_leap_year(year: int) -> bool:
    """Standard leap-year definition."""
    return calendar.isleap(year)


# Condition checks 

def check_day_condition(day: int, month: int, year: int, day_idx: int) -> bool:
    return day_of_week_index(day, month, year) == day_idx


def check_month_condition(month: int, month_idx: int) -> bool:
    return month == (month_idx + 1)          # month_idx 0-based, month 1-based


def check_leap_condition(year: int, is_leap: int) -> bool:
    return is_leap_year(year) == bool(is_leap)


def check_decade_condition(year: int, decade_idx: int) -> bool:
    decade_start = (decade_idx + 180) * 10   # e.g. idx 0 → 1800
    return decade_start <= year <= decade_start + 9


def all_conditions_met(
    day: int, month: int, year: int,
    condition: np.ndarray,
) -> bool:
    
    # Return True when every condition is satisfied.
    # condition = [day_idx, month_idx, is_leap, decade_idx]
    
    if not is_valid_date(day, month, year):
        return False
    day_idx, month_idx, is_leap, decade_idx = int(condition[0]), int(condition[1]), int(condition[2]), int(condition[3])
    return (
        check_day_condition(day, month, year, day_idx)
        and check_month_condition(month, month_idx)
        and check_leap_condition(year, is_leap)
        and check_decade_condition(year, decade_idx)
    )


# Per-condition accuracy 

def condition_scores(
    day: int, month: int, year: int,
    condition: np.ndarray,
) -> dict[str, bool]:
    valid = is_valid_date(day, month, year)
    if not valid:
        return {"valid": False, "day": False, "month": False, "leap": False, "decade": False}
    day_idx, month_idx, is_leap, decade_idx = (
        int(condition[0]), int(condition[1]), int(condition[2]), int(condition[3])
    )
    return {
        "valid":  True,
        "day":    check_day_condition(day, month, year, day_idx),
        "month":  check_month_condition(month, month_idx),
        "leap":   check_leap_condition(year, is_leap),
        "decade": check_decade_condition(year, decade_idx),
    }


# Main evaluation function 

def evaluate_predictions(
    conditions: np.ndarray,   # (N, 4)
    predictions: np.ndarray,  # (N, 3)  raw (day, month, year)
    verbose: bool = True,
) -> dict[str, float]:
 
    N = len(conditions)
    totals = {"valid": 0, "day": 0, "month": 0, "leap": 0, "decade": 0, "all": 0}

    # Detect if predictions are normalised (all values in [0,1])
    normalised = predictions.max() <= 1.01

    for i in range(N):
        if normalised:
            day, month, year = denormalise_date(predictions[i])
        else:
            day   = int(round(float(predictions[i, 0])))
            month = int(round(float(predictions[i, 1])))
            year  = int(round(float(predictions[i, 2])))

        scores = condition_scores(day, month, year, conditions[i])
        for key in totals:
            if key == "all":
                continue
            totals[key] += int(scores[key])
        if all(scores[k] for k in ["valid", "day", "month", "leap", "decade"]):
            totals["all"] += 1

    results = {k: v / N for k, v in totals.items()}
    if verbose:
        print("\n── Evaluation Results ───────────────────────")
        for key, val in results.items():
            print(f"  {key:<10}: {val * 100:.1f}%")
        print("─────────────────────────────────────────────\n")
    return results


# Best-of-N selection 

def pick_best_sample(
    samples_norm: np.ndarray,  # (num_samples, 3) normalised
    condition: np.ndarray,     # (4,)
) -> Tuple[int, int, int]:
 
    best_score = -1
    best_date  = None

    for s in samples_norm:
        day, month, year = denormalise_date(s)
        if not is_valid_date(day, month, year):
            continue
        scores = condition_scores(day, month, year, condition)
        # count how many sub-conditions are met
        score = sum(scores[k] for k in ["day", "month", "leap", "decade"])
        if score > best_score:
            best_score = score
            best_date  = (day, month, year)
        if best_score == 4:
            break  # perfect match found

    if best_date is None:
        # fallback: just round the first sample
        best_date = denormalise_date(samples_norm[0])

    return best_date