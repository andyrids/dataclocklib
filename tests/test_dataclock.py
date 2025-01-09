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
from matplotlib.text import Text

from dataclocklib.charts import dataclock

tests_directory = pathlib.Path("__file__").parent / "tests"
data_file = tests_directory / "data" / "traffic_data.parquet.gzip"
traffic_data = pd.read_parquet(data_file.as_posix())


kwargs = {"baseline_dir": "plotting/baseline", "tolerance": 10}


@pytest.mark.mpl_image_compare(**kwargs, filename="test_year_month_chart.png")
def test_year_month():
    """Test YEAR_MONTH mode chart generation.

    >>> pytest --mpl

    Returns:
        Figure object for comparison with reference figure in
        tests/plotting/baseline directory.
    """
    datetime_start = "Date_Time.dt.year.ge(2013)"
    datetime_stop = "Date_Time.dt.year.le(2014)"

    chart_title = "UK Car Accidents 2013 - 2014"
    chart_subtitle = "Count by year & month"
    chart_source = "www.kaggle.com/datasets/silicon99/dft-accident-data"

    chart_data, fig, ax = dataclock(
        data=traffic_data.query(f"{datetime_start} & {datetime_stop}"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="YEAR_MONTH",
        chart_title=chart_title,
        chart_subtitle=chart_subtitle,
        chart_source=chart_source,
    )

    manual_aggregation = (
        traffic_data.query(f"{datetime_start} & {datetime_stop}")
        .assign(year=lambda x: x["Date_Time"].dt.year)
        .assign(month=lambda x: x["Date_Time"].dt.month)
        .groupby(["year", "month"], as_index=False)
        .agg(count=pd.NamedAgg("Date_Time", "count"))
    )

    assert manual_aggregation["count"].min() == chart_data["count"].min()
    assert manual_aggregation["count"].max() == chart_data["count"].max()
    assert manual_aggregation["count"].sum() == chart_data["count"].sum()

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
    assert chart_source in axis_text_str

    # return Figure for comparison with baseline reference
    return fig


@pytest.mark.mpl_image_compare(**kwargs, filename="test_dow_hour_chart.png")
def test_dow_hour():
    """Test DOW_HOUR mode chart generation.

    >>> pytest --mpl --mpl-generate-summary=html

    Returns:
        Figure object for comparison with reference figure in
        tests/plotting/baseline directory.
    """

    chart_title = "UK Car Accidents 2011"
    chart_subtitle = "Count by day of week & hour of day"
    chart_source = "www.kaggle.com/datasets/silicon99/dft-accident-data"

    chart_data, fig, ax = dataclock(
        data=traffic_data.query("Date_Time.dt.year.eq(2011)"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="DOW_HOUR",
        chart_title=chart_title,
        chart_subtitle=chart_subtitle,
        chart_source=chart_source,
    )

    manual_aggregation = (
        traffic_data.query("Date_Time.dt.year.eq(2011)")
        .assign(dow=lambda x: x["Date_Time"].dt.day_name())
        .assign(hour=lambda x: x["Date_Time"].dt.hour)
        .groupby(["dow", "hour"], as_index=False)
        .agg(count=pd.NamedAgg("Date_Time", "count"))
    )

    assert manual_aggregation["count"].min() == chart_data["count"].min()
    assert manual_aggregation["count"].max() == chart_data["count"].max()
    assert manual_aggregation["count"].sum() == chart_data["count"].sum()

    axis_text_children = filter(
        lambda x: isinstance(x, Text), ax.properties()["children"]
    )

    axis_text_str = " ".join(
        map(lambda x: x.properties()["text"], axis_text_children)
    )

    # test polar axis label, title, subtitle & source text
    hour_labels = " ".join(f"{str(h).zfill(2)}:00" for h in range(24))
    assert hour_labels in axis_text_str
    assert chart_title in axis_text_str
    assert chart_subtitle in axis_text_str
    assert chart_source in axis_text_str

    # return Figure for comparison with baseline reference
    return fig
