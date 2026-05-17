from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

#  Lookup tables 
DAY_TO_IDX: dict[str, int] = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
    "FRI": 4, "SAT": 5, "SUN": 6,
}
IDX_TO_DAY: dict[int, str] = {v: k for k, v in DAY_TO_IDX.items()}

MONTH_TO_IDX: dict[str, int] = {
    "JAN": 0, "FEB": 1, "MAR": 2, "APR": 3,
    "MAY": 4, "JUN": 5, "JUL": 6, "AUG": 7,
    "SEP": 8, "OCT": 9, "NOV": 10, "DEC": 11,
}
IDX_TO_MONTH: dict[int, str] = {v: k for k, v in MONTH_TO_IDX.items()}

#  Normalisation helpers 
DATE_OFFSETS = np.array([1.0, 1.0, 1800.0], dtype=np.float32)   # min values
DATE_RANGES  = np.array([30.0, 11.0, 400.0], dtype=np.float32)  # max - min


def normalise_date(day: int, month: int, year: int) -> np.ndarray:
    """Return date as a float32 array with each component scaled to [0, 1]."""
    raw = np.array([day, month, year], dtype=np.float32)
    return (raw - DATE_OFFSETS) / DATE_RANGES


def denormalise_date(normed: np.ndarray) -> Tuple[int, int, int]:
    # Convert a [0,1]-scaled array back to (day, month, year) integers
    raw = normed * DATE_RANGES + DATE_OFFSETS
    day   = int(round(float(raw[0])))
    month = int(round(float(raw[1])))
    year  = int(round(float(raw[2])))
    return day, month, year


#  Decade helpers 
def decade_token_to_idx(token: str) -> int:
    """'180' → 0, '181' → 1, … '220' → 40"""
    decade = int(token)
    return decade - 180


def decade_idx_to_token(idx: int) -> str:
    return str(idx + 180)


# Parsing 
@dataclass
class DateRecord:
    day_idx:    int   # 0-6
    month_idx:  int   # 0-11
    is_leap:    int   # 0 or 1
    decade_idx: int   # 0-40
    day:        int   # 1-31
    month:      int   # 1-12
    year:       int   # 1800-2200

    @property
    def condition_array(self) -> np.ndarray:
        return np.array(
            [self.day_idx, self.month_idx, self.is_leap, self.decade_idx],
            dtype=np.int64,
        )

    @property
    def date_array_raw(self) -> np.ndarray:
        """Return (day, month, year) as float32."""
        return np.array([self.day, self.month, self.year], dtype=np.float32)

    @property
    def date_array_norm(self) -> np.ndarray:
        # Return (day, month, year) normalised to [0, 1]
        return normalise_date(self.day, self.month, self.year)

    def to_output_line(self) -> str:
        # Reconstruct the original text format
        day_tok    = IDX_TO_DAY[self.day_idx]
        month_tok  = IDX_TO_MONTH[self.month_idx]
        leap_tok   = "True" if self.is_leap else "False"
        decade_tok = decade_idx_to_token(self.decade_idx)
        return (
            f"[{day_tok}] [{month_tok}] [{leap_tok}] [{decade_tok}] "
            f"{self.day}-{self.month}-{self.year}"
        )


def parse_line(line: str) -> Optional[DateRecord]:
   
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None

    try:
        day_tok    = parts[0].strip("[]")
        month_tok  = parts[1].strip("[]")
        leap_tok   = parts[2].strip("[]")
        decade_tok = parts[3].strip("[]")
        date_str   = parts[4]                  # dd-mm-yyyy

        day_part, month_part, year_part = date_str.split("-")
        return DateRecord(
            day_idx    = DAY_TO_IDX[day_tok],
            month_idx  = MONTH_TO_IDX[month_tok],
            is_leap    = 1 if leap_tok == "True" else 0,
            decade_idx = decade_token_to_idx(decade_tok),
            day        = int(day_part),
            month      = int(month_part),
            year       = int(year_part),
        )
    except (KeyError, ValueError):
        return None


def parse_condition_line(line: str) -> Optional[np.ndarray]:
  
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        day_tok    = parts[0].strip("[]")
        month_tok  = parts[1].strip("[]")
        leap_tok   = parts[2].strip("[]")
        decade_tok = parts[3].strip("[]")
        return np.array(
            [
                DAY_TO_IDX[day_tok],
                MONTH_TO_IDX[month_tok],
                1 if leap_tok == "True" else 0,
                decade_token_to_idx(decade_tok),
            ],
            dtype=np.int64,
        )
    except (KeyError, ValueError):
        return None


def load_dataset(filepath: str) -> List[DateRecord]:
    records: List[DateRecord] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = parse_line(line)
            if rec is not None:
                records.append(rec)
    return records


def records_to_arrays(
    records: List[DateRecord],
) -> Tuple[np.ndarray, np.ndarray]:
   
    conditions = np.stack([r.condition_array  for r in records])
    dates_norm = np.stack([r.date_array_norm  for r in records])
    return conditions, dates_norm


def format_output_line(
    condition_array: np.ndarray,
    day: int,
    month: int,
    year: int,
) -> str:
   
    day_tok    = IDX_TO_DAY[int(condition_array[0])]
    month_tok  = IDX_TO_MONTH[int(condition_array[1])]
    leap_tok   = "True" if int(condition_array[2]) == 1 else "False"
    decade_tok = decade_idx_to_token(int(condition_array[3]))
    return (
        f"[{day_tok}] [{month_tok}] [{leap_tok}] [{decade_tok}] "
        f"{day}-{month}-{year}"
    )