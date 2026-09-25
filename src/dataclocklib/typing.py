"""Custom types module dataclocklib package.

Types:
    Aggregation: Keys representing aggregation functions.
    CmapNames: Keys representing matplotlib colour map names.
    FontStyle: Keys representing valid font styles.
    Mode: Keys representing temporal bins used in each chart.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Literal, TypeAlias

CmapNames: TypeAlias = Literal[
    "RdYlGn_r", "CMRmap_r", "inferno_r", "YlGnBu_r", "viridis"
]

Mode: TypeAlias = Literal[
    "YEAR_MONTH", "YEAR_WEEK", "WEEK_DAY", "DOW_HOUR", "DAY_HOUR"
]

Aggregation: TypeAlias = Literal[
    "count", "max", "mean", "median", "min", "sum"
]

FontStyle: TypeAlias = Literal["normal", "italic", "oblique"]
