"""Custom types module dataclocklib package.

Author: Andrew Ridyard.

License: GNU General Public License v3 or later.

Copyright (C): 2025.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Types:
    Aggregation: Keys representing aggregation functions.
    CmapNames: Keys representing matplotlib colour map names.
    FontStyle: Keys representing valid font styles.
    Mode: Keys representing temporal bins used in each chart.
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
