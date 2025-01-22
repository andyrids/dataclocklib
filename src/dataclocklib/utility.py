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
    get_figure_dimensions: Calculate an optimal data clock figure size.

Constants:
    VALID_STYLES: Valid font styles.
"""

import math
from collections import defaultdict
from typing import Optional, Tuple, get_args, Sequence

import numpy as np
from dataclocklib.typing import Aggregation
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy.typing import DTypeLike, NDArray
from pandas import DataFrame, MultiIndex
from pypalettes import load_cmap

from dataclocklib.exceptions import ModeError
from dataclocklib.typing import CmapNames, FontStyle, Mode

VALID_STYLES: Tuple[FontStyle, ...] = get_args(FontStyle)


def add_colorbar(
    ax: Axes,
    fig: Figure,
    cmap_name: str,
    cmap_reverse: bool,
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

    cmap = load_cmap(cmap_name, cmap_type="continuous", reverse=cmap_reverse)
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


def add_wedge_labels(
        ax: Axes,
        font_scale_factor: float,
        ring_scale_factor: float,
        ring_text_spacing: float,
        max_radius: int,
        theta: NDArray,
        width: float,
        wedge_labels: Sequence[str]
) -> None:
    """Add scaled and rotated labels around each data clock wedge.

    Labels are placed using Axes.text to facilitate custom rotation
    of the text, which is based on the angle of the wedge being
    annotated. The text is scaled based on the size of the chart
    Figure and padded away from the polar axis based on the number
    of rings in the chart.

    Args:
        ax (Axes): Chart Axis.
        font_scale_factor (float): Scale factor based on current figure size.
        ring_scale_factor (float): Scale factor based on number of rings.
        ring_text_spacing (float): Text label distance from polar axis.
        max_radius (int): Maximum radius (unique rings + 1).
        theta (NDArray): Angles (radians) for each data clock wedge.
        width (float): Width of each wedge (2 * Pi / number of wedges).
        wedge_labels (Sequence[str]): Label text for each wedge.

        Returns:
            None
    """
    if ring_scale_factor > 3:
        ring_text_spacing = ring_text_spacing * (ring_scale_factor**0.61)
    else:
        ring_text_spacing = ring_text_spacing * ring_scale_factor

    # place labels in the centre of each wedge
    for idx, angle in enumerate(theta + width / 2):
        # convert to degrees for text rotation
        angle_deg = np.rad2deg(angle)

        if (0 <= angle_deg < 90) or (270 <= angle_deg <= 360):
            rotation = -angle_deg
        else:
            rotation = 180 - angle_deg

        ax.text(
            angle,
            max_radius + ring_text_spacing,
            wedge_labels[idx],
            rotation=rotation,
            rotation_mode="anchor",
            transform=ax.transData,
            family="sans-serif",
            fontsize= 11 * font_scale_factor,
            weight="medium",
            style="normal",
            ha="center",
            va="center",
        )


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


def aggregate_temporal_columns(
        data: DataFrame,
        agg_column: str,
        agg: Aggregation,
        mode: Mode
) -> DataFrame:
    """Aggregate values in agg_column using pass aggregate function.

    Groups the DataFrame by the temporal 'ring' and 'wedge' columns,
    before applying the aggregate function to the chosen aggregation
    column.

    NOTE: The 'ring' & 'wedge' columns are assigned by the utility function
    assign_temporal_columns.

    Args:
        data (DataFrame): DataFrame containing data to aggregate.
        agg_column (str): DataFrame Column to aggregate.
        agg (Aggregation): Aggregation function; 'count', 'mean', 'median',
            'mode' & 'sum'.
        mode (Mode): A mode key representing the temporal bins used in the
            chart; 'YEAR_MONTH', 'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' &
            'DAY_HOUR'.

    Raises:
        ModeError: Unexpected mode value is passed.
        ValueError: Missing 'ring' & 'wedge' columns.

    Returns:
        A DataFrame with aggregate values in a new column named after the
        aggregate function.
    """
    columns = ["ring", "wedge"]
    if not set(columns).issubset(data.columns):
        raise ValueError(f"Expected DataFrame columns: {columns}")
    
    unique_rings = data["ring"].unique()
    match mode:
        case "YEAR_MONTH":
            unique_wedges = tuple(range(1, 13))
        case "YEAR_WEEK":
            unique_wedges = range(1, 53)
        case "WEEK_DAY":
            unique_wedges = range(0, 7)
        case "DOW_HOUR":
            unique_rings = range(0, 7)
            unique_wedges = range(0, 24)
        case "DAY_HOUR":
            unique_wedges = range(0, 24)
        case _:
            raise ModeError(mode, get_args(Mode))
    
    # groupby 'ring' & 'wedge' values and apply aggregate function agg
    data_agg = data.groupby(columns, as_index=False)[agg_column].agg(agg)
    data_agg = data_agg.set_axis([*columns, agg], axis="columns")

    # index with all possible combinations of ring & wedge values
    product_idx = MultiIndex.from_product(
        [unique_rings, unique_wedges], names=columns
    )

    # populate any rows for missing ring/wedge combinations
    data_agg = data_agg.set_index(columns).reindex(product_idx).reset_index()

    # replace NaN values created for missing missing ring/wedge combinations
    return data_agg.fillna(0)


def assign_temporal_columns(
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
        mode_map[mode]["wedge"] = data[date_column].dt.month
        #data[date_column].dt.month_name()
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
        mode_map[mode]["wedge"] = data[date_column].dt.day_of_week
    # days 1 - 7 (Monday - Sunday) | 00:00 - 23:00
    if mode == "DOW_HOUR":
        mode_map[mode]["ring"] = data[date_column].dt.day_of_week
        mode_map[mode]["wedge"] = data[date_column].dt.hour
    # days 1 - 365 | 00:00 - 23:00
    if mode == "DAY_HOUR":
        mode_map[mode]["ring"] = data[date_column].dt.strftime("%Y%j")
        mode_map[mode]["wedge"] = data[date_column].dt.hour

    return data.assign(**mode_map[mode]).astype({"ring": "int64"})


def get_figure_dimensions(wedges: int) -> tuple[float, float]:
    """Calculate an optimal data clock figure size based on wedge count.

    For most data clock charts, a minimum of 0.70 inches of figure space per
    wedge appears to work best. The best figure shape for this type of chart
    is square, given the circular nature of the chart.

    NOTE: The minimum figure size is capped at (10.0, 10.0).

    Example:
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
