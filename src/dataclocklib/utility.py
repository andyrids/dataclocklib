"""Utility function module for chart creation.

Functions:
    add_colorbar: Add a colorbar to a figure, using the provided axis.
    add_text: Create annotation text on an Axes.
    add_wedge_labels: Add scaled and rotated labels around each wedge.
    aggregate_temporal_columns: Aggregate values by ring & wedge columns.
    assign_temporal_columns: Assign ring & wedge columns to a DataFrame.
    get_figure_dimensions: Calculate an optimal data clock figure size.

Constants:
    VALID_STYLES: Valid font styles.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any, get_args

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy.typing import DTypeLike, NDArray
from pandas import DataFrame, MultiIndex, Series
from pypalettes import load_cmap  # type: ignore[import-untyped]

from dataclocklib.exceptions import ModeError
from dataclocklib.typing import Aggregation, FontStyle, Mode

VALID_STYLES: tuple[FontStyle, ...] = get_args(FontStyle)


def add_colorbar(
    ax: Axes,
    fig: Figure,
    cmap_name: str,
    cmap_reverse: bool,
    vmax: float,
    dtype: DTypeLike = np.float64,
    vmin: float = 1,
) -> Colorbar:
    """Add a colorbar to a figure, sharing the provided axis.

    Values below vmin and NaN values are mapped to white.

    Args:
        ax (Axes): Chart Axis.
        fig (Figure): Chart Figure.
        cmap_name (str): Name of matplotlib/PyPalettes colormap.
        cmap_reverse (bool): Reverse cmap colors flag.
        vmax (float): Maximum value of the colorbar.
        dtype (DTypeLike, optional): Data type for colorbar values.
        vmin (float, optional): Minimum value of the colorbar.

    Returns:
        A Colorbar object with a cmap and normalised cmap.
    """
    # unique, as integer ticks over a narrow range can repeat
    colorbar_ticks = np.unique(np.linspace(vmin, vmax, 5, dtype=dtype))

    cmap = load_cmap(cmap_name, cmap_type="continuous", reverse=cmap_reverse)
    # values below the colorbar minimum & NaN (empty bins) are plotted as white
    cmap = cmap.with_extremes(under="w", bad="w")
    cmap_norm = Normalize(vmin, vmax)

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
    theta: NDArray[np.float64],
    width: float,
    wedge_labels: Sequence[str],
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
        theta (NDArray[np.float64]): Angles (radians) for each wedge.
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
            fontsize=11 * font_scale_factor,
            weight="medium",
            style="normal",
            ha="center",
            va="center",
        )


def add_text(
    ax: Axes, x: float, y: float, text: str | None = None, **kwargs: Any
) -> Text:
    """Annotate a position on an axis denoted by xy with text.

    Args:
        ax (Axes): Axis to annotate.
        x (float): Axis x position.
        y (float): Axis y position.
        text (str, optional): Text to annotate; an empty string if None.
        **kwargs (Any): Text properties passed to Axes.text.

    Returns:
        Text object with annotation.
    """
    s = "" if text is None else text
    return ax.text(x, y, s, **kwargs)


def aggregate_temporal_columns(
    data: DataFrame, agg_column: str, agg: Aggregation, mode: Mode
) -> DataFrame:
    """Aggregate values in agg_column using pass aggregate function.

    Groups the DataFrame by the temporal 'ring' and 'wedge' columns,
    before applying the aggregate function to the chosen aggregation
    column. Missing ring/wedge combinations are filled with 0.

    NOTE: The 'ring' & 'wedge' columns are assigned by the utility function
    assign_temporal_columns.

    Args:
        data (DataFrame): DataFrame containing data to aggregate.
        agg_column (str): DataFrame Column to aggregate.
        agg (Aggregation): Aggregation function; 'count', 'max', 'mean',
            'median', 'min' & 'sum'.
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
    data_agg = _aggregate_temporal_columns(data, agg_column, agg, mode)

    # replace NaN values created for missing ring/wedge combinations
    return data_agg.fillna(0)


def _aggregate_temporal_columns(
    data: DataFrame, agg_column: str, agg: Aggregation, mode: Mode
) -> DataFrame:
    """Aggregate values in agg_column, leaving empty temporal bins as NaN.

    Args:
        data (DataFrame): DataFrame containing data to aggregate.
        agg_column (str): DataFrame Column to aggregate.
        agg (Aggregation): Aggregation function; 'count', 'max', 'mean',
            'median', 'min' & 'sum'.
        mode (Mode): A mode key representing the temporal bins used in the
            chart; 'YEAR_MONTH', 'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' &
            'DAY_HOUR'.

    Raises:
        ModeError: Unexpected mode value is passed.
        ValueError: Missing 'ring' & 'wedge' columns.

    Returns:
        A DataFrame with aggregate values in a new column named after the
        aggregate function, with NaN for missing ring/wedge combinations.
    """
    columns = ["ring", "wedge"]
    if not set(columns).issubset(data.columns):
        raise ValueError(f"Expected DataFrame columns: {columns}")

    # rings are drawn in chronological order
    unique_rings: Iterable[int] = np.sort(data["ring"].unique())
    unique_wedges: Iterable[int]
    match mode:
        case "YEAR_MONTH":
            unique_wedges = range(1, 13)
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
    data_agg = data.groupby(columns, as_index=False)[[agg_column]].agg(agg)
    data_agg = data_agg.set_axis([*columns, agg], axis="columns")

    # index with all possible combinations of ring & wedge values
    product_idx = MultiIndex.from_product(
        [unique_rings, unique_wedges], names=columns
    )

    # populate any rows for missing ring/wedge combinations
    return data_agg.set_index(columns).reindex(product_idx).reset_index()


def assign_temporal_columns(
    data: DataFrame, date_column: str, mode: Mode
) -> DataFrame:
    """Assign ring & wedge columns to a DataFrame based on mode.

    The mode value is mapped to a predetermined division of a larger unit of
    time into rings, which are then subdivided by a smaller unit of time into
    wedges, creating a set of temporal bins. These bins are assigned as 'ring'
    and 'wedge' columns.

    'YEAR_WEEK' rings are calendar years, with ISO week numbers clamped so
    that weeks never cross a calendar year boundary; week 1 therefore spans
    4 - 10 days and week 52 spans 5 - 12 days. 'WEEK_DAY' rings are ISO
    year-weeks (YYYYWW), so a calendar-year filter can include a partial ISO
    week from a neighbouring year (e.g. 2010-01-01 is in ring 200953).

    Args:
        data (DataFrame): DataFrame containing data to visualise.
        date_column (str): Name of DataFrame datetime64 column.
        mode (Mode, optional): A mode key representing the
            temporal bins used in the chart; 'YEAR_MONTH',
            'YEAR_WEEK', 'WEEK_DAY', 'DOW_HOUR' & 'DAY_HOUR'.

    Raises:
        ModeError: Unexpected mode value is passed.

    Returns:
        A DataFrame with 'ring' & 'wedge' columns assigned.
    """
    dates = data[date_column].dt
    ring: Series
    wedge: Series
    match mode:
        # year | January - December
        case "YEAR_MONTH":
            ring, wedge = dates.year, dates.month
        # calendar year | weeks 1 - 52 (ISO weeks clamped to the calendar
        # year; ISO week 53 is merged into week 52)
        case "YEAR_WEEK":
            week = dates.isocalendar().week
            week = week.mask(dates.month.eq(1) & week.ge(52), 1)
            week = week.mask(dates.month.eq(12) & week.eq(1), 52)
            ring, wedge = dates.year, week.where(week != 53, 52)
        # ISO year-week (YYYYWW) | Monday - Sunday
        case "WEEK_DAY":
            iso = dates.isocalendar()
            ring = iso.year * 100 + iso.week
            wedge = dates.day_of_week
        # days 1 - 7 (Monday - Sunday) | 00:00 - 23:00
        case "DOW_HOUR":
            ring, wedge = dates.day_of_week, dates.hour
        # days 1 - 366 | 00:00 - 23:00
        case "DAY_HOUR":
            ring, wedge = dates.strftime("%Y%j"), dates.hour
        case _:
            raise ModeError(mode, get_args(Mode))

    return data.assign(ring=ring, wedge=wedge).astype({"ring": "int64"})


def get_figure_dimensions(wedges: int) -> tuple[float, float]:
    """Calculate an optimal data clock figure size based on wedge count.

    For most data clock charts, a minimum of 0.70 inches of figure space per
    wedge appears to work best. The best figure shape for this type of chart
    is square, given the circular nature of the chart.

    NOTE: The minimum figure size is capped at (10.0, 10.0).

    Example:
        >>> get_figure_dimensions(168)
        (11.0, 11.0)

    Args:
        wedges (int): Number of wedges (number of rings * wedges per ring).

    Returns:
        A tuple containing the height & width of the square figure in inches.
    """
    space_needed = wedges * 0.70
    figure_size = float(max(math.ceil(math.sqrt(space_needed)), 10))
    return figure_size, figure_size
