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
    test_year_month_default: Test default YEAR_MONTH mode chart generation.
    test_week_day_default: Test default WEEK_DAY mode chart generation.
    test_dow_hour_default: Test default DOW_HOUR mode chart generation.
    test_day_hour_default: Test default DAY_HOUR mode chart generation.
    test_chart_annotation: Test chart annotation text.
    test_chart_aggregation: Test chart aggregation calculations.
    test_week_boundaries: Test week mode bins around year boundaries.
    test_week_ring_order: Test week mode rings are chronological.
    test_line_chart_aggregation: Test line chart aggregation calculations.
    test_non_ns_datetime_units: Test non-nanosecond datetime64 columns.
    test_parsed_string_datetimes: Test datetimes parsed from strings.
    test_tz_aware_datetime_raises: Test tz-aware datetime rejection.
    test_validation_errors: Test chart parameter validation errors.
    test_no_deprecation_warnings: Test no deprecation warnings are raised.
    test_aggregation_dtypes: Test aggregation result dtypes.
    test_wedge_labels: Test wedge label text for each mode.
"""

import calendar
import pathlib
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from dataclocklib.charts import (
    VALID_MODES,
    _wedge_labels,
    dataclock,
    line_chart,
)
from dataclocklib.exceptions import (
    AggregationColumnError,
    AggregationFunctionError,
    EmptyDataFrameError,
    MissingDatetimeError,
    ModeError,
)
from dataclocklib.utility import (
    aggregate_temporal_columns,
    assign_temporal_columns,
)

tests_directory = pathlib.Path(__file__).parent
data_file = tests_directory / "data" / "traffic_data.parquet.gzip"
traffic_data = pd.read_parquet(data_file.as_posix())

# small two week subset for fast unit tests
subset_data = traffic_data.query(
    "Date_Time.ge('2013-12-01') & Date_Time.le('2013-12-14 23:59:59')"
)

mpl_kwargs = {"baseline_dir": "plotting/baseline", "tolerance": 35}


@pytest.mark.mpl_image_compare(**mpl_kwargs)
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
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig


@pytest.mark.mpl_image_compare(**mpl_kwargs)
def test_week_day_default() -> Figure:
    """Test default WEEK_DAY mode chart generation.

    >>> pytest --mpl

    Returns:
        Figure object for comparison with reference figure in
        tests/plotting/baseline directory.
    """
    datetime_start = "Date_Time.ge('2011-12-1 00:00:00')"
    datetime_stop = "Date_Time.le('2011-12-31 23:59:59')"

    chart_data, fig, ax = dataclock(
        data=traffic_data.query(f"{datetime_start} & {datetime_stop}"),
        date_column="Date_Time",
        agg="count",
        agg_column=None,
        mode="WEEK_DAY",
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig


@pytest.mark.mpl_image_compare(**mpl_kwargs)
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
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig


@pytest.mark.mpl_image_compare(**mpl_kwargs)
def test_day_hour_default() -> Figure:
    """Test default DAY_HOUR mode chart generation.

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
        default_text=True,
        chart_title=None,
        chart_subtitle=None,
        chart_period=None,
        chart_source=None,
    )
    return fig


def test_chart_annotation() -> None:
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
        default_text=False,
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


def test_chart_aggregation() -> None:
    """Test chart aggregation calculations.

    >>> pytest --mpl
    """
    dates = traffic_data["Date_Time"].dt
    iso = dates.isocalendar()
    # YEAR_WEEK: ISO week, clamped to the calendar year, week 53 -> 52
    week = iso["week"].astype("int64")
    week[iso["year"].lt(dates.year)] = 1
    week[iso["year"].gt(dates.year)] = 52
    week[week.eq(53)] = 52
    manual_data = traffic_data.assign(
        year=dates.year,
        month=dates.month,
        week=week,
        iso_year_week=(iso["year"] * 100 + iso["week"]).astype("int64"),
        dow=dates.day_of_week,
        hour=dates.hour,
    )

    for mode, columns in (
        ("YEAR_MONTH", ["year", "month"]),
        ("YEAR_WEEK", ["year", "week"]),
        ("WEEK_DAY", ["iso_year_week", "dow"]),
        ("DOW_HOUR", ["dow", "hour"]),
    ):
        chart_data, fig, ax = dataclock(
            data=traffic_data,
            date_column="Date_Time",
            agg="count",
            agg_column=None,
            mode=mode,
            default_text=False,
            chart_title=None,
            chart_subtitle=None,
            chart_period=None,
            chart_source=None,
        )

        plt.close(fig)

        # compare every non-empty ring/wedge cell
        manual_counts = (
            manual_data.groupby(columns).size().rename_axis(["ring", "wedge"])
        )
        chart_counts = (
            chart_data[chart_data["count"].gt(0)]
            .set_index(["ring", "wedge"])["count"]
            .astype("int64")
        )
        pd.testing.assert_series_equal(
            chart_counts.sort_index(),
            manual_counts.sort_index(),
            check_names=False,
            check_index_type=False,
        )


@pytest.mark.parametrize(
    ("date", "year_week", "week_day"),
    [
        ("2009-12-31", (2009, 52), (200953, 3)),
        ("2010-01-01", (2010, 1), (200953, 4)),
        ("2010-01-03", (2010, 1), (200953, 6)),
        ("2010-01-04", (2010, 1), (201001, 0)),
        ("2013-12-30", (2013, 52), (201401, 0)),
        ("2013-12-31", (2013, 52), (201401, 1)),
        ("2015-12-31", (2015, 52), (201553, 3)),
        ("2016-01-03", (2016, 1), (201553, 6)),
        ("2020-12-31", (2020, 52), (202053, 3)),
        ("2021-01-01", (2021, 1), (202053, 4)),
    ],
)
def test_week_boundaries(
    date: str, year_week: tuple[int, int], week_day: tuple[int, int]
) -> None:
    """Test ring & wedge values for dates around year boundaries."""
    data = pd.DataFrame({"Date_Time": pd.to_datetime([date])})
    for mode, expected in (("YEAR_WEEK", year_week), ("WEEK_DAY", week_day)):
        result = assign_temporal_columns(data, "Date_Time", mode)
        assert (result["ring"].item(), result["wedge"].item()) == expected


def test_week_ring_order() -> None:
    """Test week mode rings are chronological for a calendar-year filter."""
    data = traffic_data.query("Date_Time.dt.year.eq(2010)")

    week_day = aggregate_temporal_columns(
        assign_temporal_columns(data, "Date_Time", "WEEK_DAY"),
        "Date_Time",
        "count",
        "WEEK_DAY",
    )
    rings = week_day["ring"].unique()
    assert rings[0] == 200953
    assert list(rings) == sorted(rings)

    year_week = aggregate_temporal_columns(
        assign_temporal_columns(data, "Date_Time", "YEAR_WEEK"),
        "Date_Time",
        "count",
        "YEAR_WEEK",
    )
    assert list(year_week["ring"].unique()) == [2010]


def test_line_chart_aggregation() -> None:
    """Test line chart aggregation calculations against a manual groupby."""
    manual_data = traffic_data.assign(
        dow=lambda x: x["Date_Time"].dt.day_of_week,
        hour=lambda x: x["Date_Time"].dt.hour,
    )
    manual_aggregation = manual_data.groupby(["dow", "hour"]).size()

    chart_data, fig, ax = line_chart(
        data=traffic_data, date_column="Date_Time", mode="DOW_HOUR"
    )
    plt.close(fig)

    assert chart_data.index.name == "ring"
    assert list(chart_data.columns) == ["wedge", "count"]
    assert manual_aggregation.max() == chart_data["count"].max()
    assert manual_aggregation.sum() == chart_data["count"].sum()


@pytest.mark.parametrize("mode", VALID_MODES)
@pytest.mark.parametrize("unit", ["us", "ms", "s"])
def test_non_ns_datetime_units(unit: str, mode: str) -> None:
    """Test non-nanosecond datetime64 columns give the same results as ns.

    Args:
        unit: datetime64 resolution unit.
        mode: Chart mode.
    """
    data = subset_data.astype({"Date_Time": f"datetime64[{unit}]"})

    for chart in (dataclock, line_chart):
        expected, fig_ns, _ = chart(subset_data, "Date_Time", mode=mode)
        result, fig, _ = chart(data, "Date_Time", mode=mode)
        plt.close(fig_ns)
        plt.close(fig)
        pd.testing.assert_frame_equal(result, expected)


def test_parsed_string_datetimes() -> None:
    """Test datetimes parsed from strings (datetime64[us]) are accepted."""
    data = subset_data.assign(
        Date_Time=pd.to_datetime(subset_data["Date_Time"].astype(str))
    )
    expected, fig_ns, _ = dataclock(subset_data, "Date_Time")
    result, fig, _ = dataclock(data, "Date_Time")
    plt.close(fig_ns)
    plt.close(fig)
    pd.testing.assert_frame_equal(result, expected)


def test_tz_aware_datetime_raises() -> None:
    """Test a tz-aware datetime column raises MissingDatetimeError."""
    data = subset_data.assign(
        Date_Time=subset_data["Date_Time"].dt.tz_localize("UTC")
    )
    with pytest.raises(MissingDatetimeError):
        dataclock(data, "Date_Time")


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"data": subset_data.iloc[:0]}, EmptyDataFrameError),
        ({"date_column": "missing"}, KeyError),
        ({"agg_column": "missing"}, KeyError),
        ({"date_column": "Latitude"}, MissingDatetimeError),
        ({"mode": "BAD_MODE"}, ModeError),
        ({"agg": "bad_agg"}, AggregationFunctionError),
        ({"agg": "sum"}, AggregationColumnError),
    ],
)
def test_validation_errors(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    """Test chart parameter validation raises the expected exceptions.

    Args:
        kwargs: Chart function keyword arguments to override.
        exception: Expected exception class.
    """
    parameters: dict[str, object] = {
        "data": subset_data,
        "date_column": "Date_Time",
    } | kwargs

    for chart in (dataclock, line_chart):
        with pytest.raises(exception) as exc_info:
            chart(**parameters)  # type: ignore[arg-type]
        if exception is not KeyError:
            assert isinstance(exc_info.value, ValueError)


def test_no_deprecation_warnings() -> None:
    """Test dataclock raises no (pending) deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error", (DeprecationWarning, PendingDeprecationWarning)
        )
        _, fig, _ = dataclock(subset_data, "Date_Time")
    plt.close(fig)


@pytest.mark.parametrize(
    ("agg", "expected_dtype"),
    [("count", "int64"), ("sum", "int64"), ("mean", "float64")],
)
def test_aggregation_dtypes(agg: str, expected_dtype: str) -> None:
    """Test aggregation result dtypes.

    Args:
        agg: Aggregation function name.
        expected_dtype: Expected aggregation column dtype.
    """
    chart_data, fig, _ = dataclock(
        subset_data,
        "Date_Time",
        agg_column=None if agg == "count" else "Number_of_Casualties",
        agg=agg,  # type: ignore[arg-type]
    )
    plt.close(fig)

    assert chart_data[agg].dtype == expected_dtype
    assert chart_data["ring"].dtype == "int64"


def test_wedge_labels() -> None:
    """Test wedge label text for each mode."""
    hours = tuple(f"{x:02d}:00" for x in range(24))

    assert _wedge_labels("WEEK_DAY", range(7)) == tuple(calendar.day_name)
    assert _wedge_labels("YEAR_MONTH", range(1, 13)) == tuple(
        calendar.month_name[1:]
    )
    assert _wedge_labels("DOW_HOUR", range(24)) == hours
    assert _wedge_labels("DAY_HOUR", range(24)) == hours
    assert _wedge_labels("YEAR_WEEK", range(1, 53)) == tuple(
        str(x) for x in range(1, 53)
    )
