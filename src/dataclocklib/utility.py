"""Utility function module for chart creation.

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

Functions:
    add_text: Create annotation text on an Axes.

Types:
    Aggregation: Keys representing aggregation functions.
    Mode: Keys representing temporal bins used in each chart.
"""

from collections import defaultdict
from typing import Literal, Optional, Tuple, get_args

from matplotlib.axes import Axes
from matplotlib.text import Text
from pandas import DataFrame

FontStyle = Literal["normal", "italic", "oblique"]
VALID_STYLES: Tuple[FontStyle, ...] = get_args(FontStyle)


def add_text(
    ax: Axes, x: int, y: int, text: Optional[str] = None, **kwargs
) -> Text:
    """Annotate a position on an axis denoted by xy with text.

    Args:
        ax (Axes): Axis to annotate.
        x (int): Axis x position.
        y (int): Axis y position.
        text (str, optional): Text to annotate.

    Returns:
        Text object with annotation.
    """
    s = "" if text is None else text
    return ax.text(x, y, s, **kwargs)


def assign_ring_wedge_columns(
        data: DataFrame,
        date_column: str,
        mode: str
    ) -> DataFrame:
    """Assign ring & wedge columns to a DataFrame based on mode.

    The mode value is mapped to a predetermined division of a larger unit of
    time into rings, which are then subdivided by a smaller unit of time into
    wedges, creating a set of temporal bins. These bins are assigned as 'ring'
    and 'wedge' columns.

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.

    Returns:
        A DataFrame with 'ring' & 'wedge' columns assigned.
    """
    # dict map for ring & wedge features based on mode
    mode_map = defaultdict(dict)
    # year | January - December
    if mode == "YEAR_MONTH":
        mode_map[mode]["ring"] = data[date_column].dt.year
        mode_map[mode]["wedge"] = data[date_column].dt.month_name()
    # year | weeks 1 - 52
    if mode == "YEAR_WEEK":
        mode_map[mode]["ring"] = data[date_column].dt.year
        week = data[date_column].dt.isocalendar().week
        week[week == 53] = 52
        mode_map[mode]["wedge"] = week
    # weeks 1 - 52 | Monday - Sunday
    if mode == "WEEK_DAY":
        week = data[date_column].dt.isocalendar().week
        year = data[date_column].dt.year
        mode_map[mode]["ring"] = week + year * 100
        mode_map[mode]["wedge"] = data[date_column].dt.day_name()
    # days 1 - 7 (Monday - Sunday) | 00:00 - 23:00
    if mode == "DOW_HOUR":
        mode_map[mode]["ring"] = data[date_column].dt.day_of_week
        mode_map[mode]["wedge"] = data[date_column].dt.hour
    # days 1 - 365 | 00:00 - 23:00
    if mode == "DAY_HOUR":
        mode_map[mode]["ring"] = data[date_column].dt.strftime("%Y%j")
        mode_map[mode]["wedge"] = data[date_column].dt.hour

    return data.assign(**mode_map[mode]).astype({"ring": "int64"})
