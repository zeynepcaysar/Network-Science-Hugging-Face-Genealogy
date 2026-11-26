"""The numbers file of a stage: one row per number the stage reports (ROADMAP Phase 2 note: every stage writes its own).

Columns are NUMBER_COLUMNS; value is text, so a count, a share, a date and an md5 live in one column and the file is stable
byte for byte (fmt fixes the number format). Rows that belong to the whole run carry an empty version and series. A reader
that wants those empty cells as text reads the file with dtype=str, keep_default_na=False, like every table of this project.

    rows = [("pilot_qwen", "full", "Qwen", "all", "events_total", 246641)]
    frame = numbers.numbers_frame(rows)       # sorted, values as text
"""
import numpy as np
import pandas as pd

NUMBER_COLUMNS = ("stage", "version", "family", "series", "name", "value")


def fmt(value):
    """Text of a value: True/False for booleans, plain digits for integers, 8 significant digits for floats (empty for NaN)."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else format(float(value), ".8g")
    return str(value)


def numbers_frame(rows):
    """DataFrame in NUMBER_COLUMNS order from row tuples, value passed through fmt, sorted by the five key columns (stable)."""
    frame = pd.DataFrame([tuple(row) for row in rows], columns=list(NUMBER_COLUMNS))
    frame["value"] = frame["value"].map(fmt)
    keys = list(NUMBER_COLUMNS[:5])
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)
