"""Unit tests module.

Author: Andrew Ridyard.

License: GNU General Public License v3 or later.

Copyright (C): 2025.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Functions:
    test_year_month: Test YEAR_MONTH mode chart generation.
"""

import calendar
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from dataclocklib.charts import dataclock

tests_directory = pathlib.Path("__file__").parent / "tests"
data_file = tests_directory / "data" / "traffic_data.parquet.gzip"
traffic_data = pd.read_parquet(data_file.as_posix())

mpl_kwargs = {"baseline_dir": "plotting/baseline", "tolerance": 35}


@pytest.mark.mpl_image_compare(
    **mpl_kwargs, filename="test_baseline_year_month_chart.png"
)
def test_year_month_default() -> Figure:
    """Test default YEAR_MONTH mode chart generation.

    >>> pytest --mpl

    Returns:
        Figure object for comparison with reference figure in
        tests/plotting/baseline directory.
    """
    datetime_start = "Date_Time.dt.year.ge(2013)"
    datetime_stop = "Date_Time.dt.year.le(2014)"

    chart_data, fig, ax = dataclock(
        data=traffic_data.query(f"{datetime_start} & {datetime_stop}"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="YEAR_MONTH",
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
        default_text=True
    )
    return fig


@pytest.mark.mpl_image_compare(
    **mpl_kwargs, filename="test_baseline_week_day_chart.png"
)
def test_week_day_default() -> Figure:
    """Test default WEEK_DAY mode chart generation.

    >>> pytest --mpl

    Returns:
        Figure object for comparison with reference figure in
        tests/plotting/baseline directory.
    """
    datetime_start = "Date_Time.ge('2013-12-1 00:00:00')"
    datetime_stop = "Date_Time.le('2013-12-31 23:59:59')"

    chart_data, fig, ax = dataclock(
        data=traffic_data.query(f"{datetime_start} & {datetime_stop}"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="WEEK_DAY",
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
        default_text=True
    )
    return fig


@pytest.mark.mpl_image_compare(
    **mpl_kwargs, filename="test_baseline_dow_hour_chart.png"
)
def test_dow_hour_default() -> Figure:
    """Test default DOW_HOUR mode chart generation.

    >>> pytest --mpl

    Returns:
        Figure object for comparison with reference figure in
        tests/plotting/baseline directory.
    """
    chart_data, fig, ax = dataclock(
        data=traffic_data.query("Date_Time.dt.year.eq(2013)"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="DOW_HOUR",
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
        default_text=True
    )
    return fig


@pytest.mark.mpl_image_compare(
    **mpl_kwargs, filename="test_baseline_day_hour_chart.png"
)
def test_day_hour_default() -> Figure:
    """Test default DOW_HOUR mode chart generation.

    >>> pytest --mpl

    Returns:
        Figure object for comparison with reference figure in
        tests/plotting/baseline directory.
    """
    datetime_start = "Date_Time.ge('2013-12-1 00:00:00')"
    datetime_stop = "Date_Time.le('2013-12-14 23:59:59')"

    chart_data, fig, ax = dataclock(
        data=traffic_data.query(f"{datetime_start} & {datetime_stop}"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="DAY_HOUR",
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
        default_text=True
    )
    return fig


def test_chart_annotation(): 
    """Test chart annotation text.

    >>> pytest --mpl
    """
    chart_title = "**CUSTOM TITLE**"
    chart_subtitle = "**CUSTOM SUBTITLE**"
    chart_period = "**CUSTOM PERIOD**"
    chart_source = "**CUSTOM SOURCE**"

    chart_data, fig, ax = dataclock(
        data=traffic_data,
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="YEAR_MONTH",
        chart_title=chart_title,
        chart_subtitle=chart_subtitle,
        chart_period=chart_period,
        chart_source=chart_source,
        default_text=False
    )

    axis_text_children = filter(
        lambda x: isinstance(x, Text), ax.properties()["children"]
    )

    axis_text_str = " ".join(
        map(lambda x: x.properties()["text"], axis_text_children)
    )

    # test polar axis label, title, subtitle & source text
    month_names = " ".join(tuple(calendar.month_name[1:]))
    assert month_names in axis_text_str
    assert chart_title in axis_text_str
    assert chart_subtitle in axis_text_str
    assert chart_period in axis_text_str
    assert chart_source in axis_text_str


def test_chart_aggregation(): 
    """Test chart aggregation calculations.

    >>> pytest --mpl
    """
    manual_data = (
        traffic_data
        .assign(year=lambda x: x["Date_Time"].dt.year)
        .assign(month=lambda x: x["Date_Time"].dt.month)
        .assign(week=lambda x: x["Date_Time"].dt.isocalendar().week)
        .assign(woy=lambda x: x["week"] + x["year"] * 100)
        .assign(dow=lambda x: x["Date_Time"].dt.day_of_week)
        .assign(dow_name=lambda x: x["Date_Time"].dt.day_name())
        .assign(hour=lambda x: x["Date_Time"].dt.hour)
    )
    # account for leap years pushing into week 53
    manual_data.loc[manual_data["week"].eq(53), "week"] = 52

    for mode, columns in (
        ("YEAR_MONTH", ["year", "month"]),
        ("YEAR_WEEK", ["year", "week"]),
        ("WEEK_DAY", ["woy", "dow_name"]),
        ("DOW_HOUR", ["dow", "hour"]),
    ):
        print(mode)

        chart_data, fig, ax = dataclock(
            data=traffic_data,
            date_column="Date_Time",
            agg="count",
            agg_column=None,
            mode=mode,
            chart_title=None,
            chart_subtitle=None,
            chart_period=None,
            chart_source=None,
            default_text=False
        )

        manual_aggregation = (
            manual_data
            .groupby(columns, as_index=False)
            .agg(count=pd.NamedAgg("Date_Time", "count"))
        )

        assert manual_aggregation["count"].max() == chart_data["count"].max()
        assert manual_aggregation["count"].sum() == chart_data["count"].sum()
