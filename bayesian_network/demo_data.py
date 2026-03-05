"""Shared demo data generator for Bayesian Network examples.

Used by both ``App.py`` (batch mode) and ``webapp.py`` (web interface)
to produce synthetic data with a known causal structure::

    Weather --> Traffic --> Commute
       |                      ^
       +--> Accident ---------+
    DayOfWeek (independent noise)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMO_TARGET = "Commute"
DEMO_COLUMNS = ["Weather", "DayOfWeek", "Traffic", "Accident", "Commute"]


def generate_demo_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with a known causal graph.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.RandomState(seed)

    weather = rng.choice(["Sunny", "Rainy", "Cloudy"], size=n, p=[0.5, 0.25, 0.25])
    day_of_week = rng.choice(["Weekday", "Weekend"], size=n, p=[0.7, 0.3])

    # Traffic depends on Weather
    traffic = np.where(
        weather == "Rainy",
        rng.choice(["Heavy", "Moderate", "Light"], size=n, p=[0.6, 0.3, 0.1]),
        np.where(
            weather == "Cloudy",
            rng.choice(["Heavy", "Moderate", "Light"], size=n, p=[0.3, 0.5, 0.2]),
            rng.choice(["Heavy", "Moderate", "Light"], size=n, p=[0.1, 0.3, 0.6]),
        ),
    )

    # Accident depends on Weather
    accident_prob = np.where(weather == "Rainy", 0.3, np.where(weather == "Cloudy", 0.15, 0.05))
    accident = np.array(["Yes" if rng.random() < p else "No" for p in accident_prob])

    # Commute depends on Traffic and Accident
    commute = []
    for t, a in zip(traffic, accident):
        if t == "Heavy" and a == "Yes":
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.7, 0.25, 0.05]))
        elif t == "Heavy" or a == "Yes":
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.2, 0.6, 0.2]))
        elif t == "Moderate":
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.05, 0.3, 0.65]))
        else:
            commute.append(rng.choice(["VeryLong", "Long", "Normal"], p=[0.02, 0.08, 0.9]))

    return pd.DataFrame({
        "Weather": weather,
        "DayOfWeek": day_of_week,
        "Traffic": traffic,
        "Accident": accident,
        "Commute": commute,
    })
