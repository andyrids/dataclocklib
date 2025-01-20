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
    add_colorbar: Add a colorbar to a figure, using the provided axis.
    add_text: Create annotation text on an Axes.
    assign_ring_wedge_columns: Assign ring & wedge columns to a DataFrame.
    calculate_figure_dimensions: Calculate an optimal data clock figure size.

Constants:
    VALID_STYLES: Valid font styles.
"""

import math
from collections import defaultdict
from typing import Optional, Tuple, get_args

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import DTypeLike
from pandas import DataFrame

from dataclocklib.typing import CmapNames, FontStyle, Mode

VALID_STYLES: Tuple[FontStyle, ...] = get_args(FontStyle)


def add_colorbar(
    ax: Axes,
    fig: Figure,
    cmap_name: CmapNames,
    vmax: float,
    dtype: DTypeLike = np.float64,
) -> Colorbar:
    """Add a colorbar to a figure, sharing the provided axis.

    Args:
        ax (Axes): Chart Axis.
        fig (Figure): Chart Figure.
        dtype (DTypeLike): Colourbar values dtype.
        cmap_name (CmapNames): Name of matplotlib colormap.
        vmax (float): maximum value of the colorbar.
        dtype (DTypeLike): Data type for colorbar values.

    Returns:
        A Colorbar object with a cmap and normalised cmap.
    """
    colorbar_ticks = np.linspace(1, vmax, 5, dtype=dtype)

    cmap = colormaps.get_cmap(cmap_name)
    cmap.set_under("w")
    cmap_norm = Normalize(1, vmax)

    colorbar = fig.colorbar(
        ScalarMappable(norm=cmap_norm, cmap=cmap),
        ax=ax,
        orientation="vertical",
        location="right",
        ticks=colorbar_ticks,
        shrink=0.5,
        extend="min",
        use_gridspec=False,
    )

    colorbar.ax.tick_params(direction="out")
    return colorbar


def add_text(
    ax: Axes, x: float, y: float, text: Optional[str] = None, **kwargs
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
    data: DataFrame, date_column: str, mode: Mode
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


def calculate_figure_dimensions(wedges: int) -> tuple[float, float]:
    """Calculate an optimal data clock figure size based on wedge count.

    For most data clock charts, a minimum of 0.70 inches of figure space per
    wedge appears to work best. The best figure shape for this type of chart
    is square, given the circular nature of the chart.

    NOTE: The minimum figure size is capped at (10.0, 10.0).

    Example:
      # 'DOW_HOUR' mode has 24 wedges for each of the 7 rings
      >>> calculate_figure_dimensions(168)
      (11, 11)

    Args:
      wedges: Number of wedges (number of rings * wedges per ring).

    Returns:
      A tuple containing the height & width of the square figure in inches.
    """
    space_needed = wedges * 0.70
    figure_size = float(max(math.ceil(math.sqrt(space_needed)), 10))
    return figure_size, figure_size
