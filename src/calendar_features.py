"""Calendar and Indian holiday/festival features."""
import warnings

import numpy as np
import pandas as pd


def indian_holidays():
    """Fixed list for the data window (2023-01-01..2024-02-28). Lunar festival dates per the
    Government of India gazetted holiday lists 2023/2024."""
    h = {
        "2023-01-14": ("Makar Sankranti / Pongal", 0), "2023-01-26": ("Republic Day", 1),
        "2023-02-18": ("Maha Shivaratri", 0), "2023-03-08": ("Holi", 1),
        "2023-03-30": ("Ram Navami", 0), "2023-04-04": ("Mahavir Jayanti", 0),
        "2023-04-07": ("Good Friday", 0), "2023-04-22": ("Eid al-Fitr", 1),
        "2023-05-05": ("Buddha Purnima", 0), "2023-06-29": ("Eid al-Adha", 1),
        "2023-07-29": ("Muharram", 0), "2023-08-15": ("Independence Day", 1),
        "2023-08-30": ("Raksha Bandhan", 0), "2023-09-07": ("Janmashtami", 0),
        "2023-09-19": ("Ganesh Chaturthi", 0), "2023-09-28": ("Milad-un-Nabi", 0),
        "2023-10-02": ("Gandhi Jayanti", 0), "2023-10-24": ("Dussehra", 1),
        "2023-11-12": ("Diwali", 1), "2023-11-13": ("Govardhan Puja", 0),
        "2023-11-15": ("Bhai Dooj", 0), "2023-11-27": ("Guru Nanak Jayanti", 0),
        "2023-12-25": ("Christmas", 1), "2024-01-01": ("New Year's Day", 0),
        "2024-01-15": ("Makar Sankranti / Pongal", 0), "2024-01-26": ("Republic Day", 1),
        # beyond the data window - used only for the Phase 4 forward forecast (Mar 2024)
        "2024-03-08": ("Maha Shivaratri", 0), "2024-03-25": ("Holi", 1), "2024-03-29": ("Good Friday", 0),
    }
    return pd.DataFrame([(pd.Timestamp(k), n, m) for k, (n, m) in h.items()],
                        columns=["date", "holiday_name", "is_major_festival"]).set_index("date")


def add_calendar(q, df):
    df["day_of_week"] = df.index.dayofweek          # 0=Mon
    df["day_name"] = df.index.day_name()
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["month"] = df.index.month
    df["year"] = df.index.year
    hol = indian_holidays()
    df = df.join(hol)
    df["is_holiday"] = df.holiday_name.notna().astype(int)
    df["is_major_festival"] = df.is_major_festival.fillna(0).astype(int)
    df["holiday_name"] = df.holiday_name.fillna("")
    q.log("calendar", f"{df.is_holiday.sum()} holiday/festival days flagged in window "
        f"({df.is_major_festival.sum()} major: Republic Day, Holi, Eid x2, Independence Day, Dussehra, Diwali, Christmas)",
        "hardcoded GoI gazetted dates (lunar festivals vary by year/region - verify if regional calendar differs)",
        df.is_holiday.sum())
    return df
