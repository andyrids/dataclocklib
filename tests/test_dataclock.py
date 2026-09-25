"""Unit tests module.

Functions:
    test_year_month_default: Test default YEAR_MONTH mode chart generation.
    test_week_day_default: Test default WEEK_DAY mode chart generation.
    test_dow_hour_default: Test default DOW_HOUR mode chart generation.
    test_day_hour_default: Test default DAY_HOUR mode chart generation.
    test_chart_annotation: Test chart annotation text.
    test_chart_aggregation: Test chart aggregation calculations.
    test_week_boundaries: Test week mode bins around year boundaries.
    test_ring_order: Test rings are chronological in every mode.
    test_week_ring_order: Test week mode rings are chronological.
    test_line_chart_aggregation: Test line chart aggregation calculations.
    test_non_ns_datetime_units: Test non-nanosecond datetime64 columns.
    test_parsed_string_datetimes: Test datetimes parsed from strings.
    test_tz_aware_datetime_raises: Test tz-aware datetime rejection.
    test_validation_errors: Test chart parameter validation errors.
    test_validation_error_messages: Test validation error message reasons.
    test_no_deprecation_warnings: Test no deprecation warnings are raised.
    test_aggregation_dtypes: Test aggregation result dtypes.
    test_wedge_labels: Test wedge label text for each mode.
    test_colour_scale: Test the colour scale for non-count aggregations.
    test_equal_values_colour_scale: Test a colour scale of equal values.
    test_count_colour_scale: Test the count colour scale starts from 1.
    test_float_sum_ticks: Test float sum colorbar ticks end at the maximum.
    test_title_font_scaling: Test title text scaling on a large figure.
    test_aggregate_zero_filled: Test aggregate_temporal_columns zero fills.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

import calendar
import pathlib
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
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

# subset with 3 missing (NaT) datetimes
nat_data = subset_data.copy()
nat_data.iloc[:3, nat_data.columns.get_loc("Date_Time")] = pd.NaT

# three consecutive hours on a Monday; DOW_HOUR ring 0, wedges 0 - 2
three_hours = pd.date_range("2024-01-01", periods=3, freq="h")

# empty temporal bins are drawn white, at the wedge bar alpha
white = to_rgba("w", 0.8)

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


@pytest.mark.parametrize("mode", VALID_MODES)
def test_ring_order(mode: str) -> None:
    """Test rings are chronological for shuffled input in every mode."""
    # spans two year boundaries, so every mode except DOW_HOUR has
    # multiple rings to order
    data = traffic_data.query(
        "Date_Time.ge('2012-12-20') & Date_Time.le('2014-01-10 23:59:59')"
    ).sample(frac=1, random_state=0)
    result = aggregate_temporal_columns(
        assign_temporal_columns(data, "Date_Time", mode),
        "Date_Time",
        "count",
        mode,
    )
    rings = list(result["ring"].unique())
    if mode != "DOW_HOUR":
        assert len(rings) > 1
    assert rings == sorted(rings)


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
        ({"data": nat_data}, MissingDatetimeError),
        (
            {"agg_column": "Accident_Severity_Label", "agg": "sum"},
            AggregationColumnError,
        ),
        ({"agg_column": "Date_Time", "agg": "max"}, AggregationColumnError),
        (
            {"data": subset_data.assign(ring=1), "agg_column": "ring"},
            AggregationColumnError,
        ),
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


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        ({"data": nat_data}, MissingDatetimeError, r"3 NaT .*dropna"),
        (
            {"agg_column": "Accident_Severity_Label", "agg": "mean"},
            AggregationColumnError,
            "not numeric",
        ),
        (
            {
                "data": subset_data.assign(wedge=1.0),
                "agg_column": "wedge",
                "agg": "sum",
            },
            AggregationColumnError,
            "reserved",
        ),
        ({"agg": "sum"}, AggregationColumnError, "Expected agg_column"),
    ],
)
def test_validation_error_messages(
    kwargs: dict[str, object], exception: type[Exception], match: str
) -> None:
    """Test chart parameter validation error messages give the reason.

    Args:
        kwargs: Chart function keyword arguments to override.
        exception: Expected exception class.
        match: Regular expression expected in the error message.
    """
    parameters: dict[str, object] = {
        "data": subset_data,
        "date_column": "Date_Time",
    } | kwargs

    for chart in (dataclock, line_chart):
        with pytest.raises(exception, match=match):
            chart(**parameters)  # type: ignore[arg-type]


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


@pytest.mark.parametrize(
    ("values", "agg"),
    [
        ([0.1, 0.5, 0.9], "mean"),
        ([0.0, 0.25, 0.75], "mean"),
        ([-5, 0, 5], "sum"),
        ([-3.5, -2.0, -0.5], "min"),
    ],
)
def test_colour_scale(values: list[float], agg: str) -> None:
    """Test the colour scale spans the data minimum to maximum.

    Values below 1, zero & negative values are coloured, while empty
    temporal bins are white.

    Args:
        values: Aggregation values for three consecutive hours.
        agg: Aggregation function name.
    """
    data = pd.DataFrame({"Date_Time": three_hours, "Value": values})
    chart_data, fig, ax = dataclock(
        data,
        "Date_Time",
        "Value",
        agg,  # type: ignore[arg-type]
        mode="DOW_HOUR",
    )
    colorbar_ax = fig.axes[1]
    ticks = list(colorbar_ax.get_yticks())
    colours = [bar.get_facecolor() for bar in ax.patches]
    plt.close(fig)

    assert colorbar_ax.get_ylim() == (min(values), max(values))
    assert ticks == sorted(set(ticks))
    # three distinct, non-white colours for the three populated bins
    assert len(set(colours[:3])) == 3
    assert white not in colours[:3]
    assert set(colours[3:]) == {white}
    # the returned aggregation values are still zero filled
    assert chart_data[agg].notna().all()
    assert chart_data[agg].iloc[3:].eq(0).all()


@pytest.mark.parametrize(
    ("values", "agg"), [([2, 2, 2], "mean"), ([0, 0, 0], "sum")]
)
def test_equal_values_colour_scale(values: list[float], agg: str) -> None:
    """Test a colour scale of equal values does not raise.

    Args:
        values: Equal aggregation values for three consecutive hours.
        agg: Aggregation function name.
    """
    data = pd.DataFrame({"Date_Time": three_hours, "Value": values})
    _, fig, ax = dataclock(
        data,
        "Date_Time",
        "Value",
        agg,  # type: ignore[arg-type]
        mode="DOW_HOUR",
    )
    ticks = list(fig.axes[1].get_yticks())
    colours = [bar.get_facecolor() for bar in ax.patches]
    plt.close(fig)

    assert ticks == [values[0]]
    assert len(set(colours[:3])) == 1
    assert white not in colours[:3]


@pytest.mark.parametrize("mode", VALID_MODES)
def test_count_colour_scale(mode: str) -> None:
    """Test the count colour scale spans 1 to the maximum count.

    Args:
        mode: Chart mode.
    """
    chart_data, fig, _ = dataclock(subset_data, "Date_Time", mode=mode)
    colorbar = fig.axes[1]
    ticks = list(colorbar.get_yticks())
    plt.close(fig)

    assert colorbar.get_ylim() == (1, chart_data["count"].max())
    assert ticks == sorted(set(ticks))
    assert ticks[0] == 1
    assert ticks[-1] == chart_data["count"].max()


def test_float_sum_ticks() -> None:
    """Test float sum colorbar ticks end at the maximum value."""
    data = pd.DataFrame({"Date_Time": three_hours, "Value": [1.25, 2.5, 3.3]})
    chart_data, fig, _ = dataclock(
        data, "Date_Time", "Value", "sum", mode="DOW_HOUR"
    )
    ticks = list(fig.axes[1].get_yticks())
    plt.close(fig)

    assert chart_data["sum"].dtype == "float64"
    assert ticks[0] == 1.25
    assert ticks[-1] == pytest.approx(3.3)


@pytest.mark.parametrize(
    ("dtype", "agg", "expected"),
    [
        ("bool", "sum", [1, 0, 1]),
        ("bool", "mean", [1.0, 0.0, 1.0]),
        ("bool", "max", [1, 0, 1]),
        ("boolean", "sum", [1, 0, 1]),
    ],
)
def test_bool_aggregation(dtype: str, agg: str, expected: list[float]) -> None:
    """Test a bool agg_column can be aggregated (e.g. sum counts True values).

    Args:
        dtype: Bool dtype of the aggregation column.
        agg: Aggregation function name.
        expected: Expected aggregation values for the three populated bins.
    """
    data = pd.DataFrame(
        {"Date_Time": three_hours, "Flag": [True, False, True]}
    ).astype({"Flag": dtype})
    for chart in (dataclock, line_chart):
        chart_data, fig, _ = chart(
            data,
            "Date_Time",
            "Flag",
            agg,  # type: ignore[arg-type]
            mode="DOW_HOUR",
        )
        plt.close(fig)

        assert chart_data[agg].iloc[:3].tolist() == expected


def test_title_font_scaling() -> None:
    """Test title text is scaled once & does not overlap on a large figure."""
    _, fig, ax = dataclock(
        subset_data, "Date_Time", mode="DAY_HOUR", chart_period="Period"
    )
    font_scale_factor = fig.get_figwidth() / 11
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

    title, subtitle, period, _ = (
        text for text in ax.texts if text.get_transform() == fig.transFigure
    )
    extents = [
        text.get_window_extent(renderer).transformed(
            fig.transFigure.inverted()
        )
        for text in (title, subtitle, period)
    ]
    plt.close(fig)

    assert font_scale_factor > 1
    assert title.get_fontsize() == pytest.approx(14 * font_scale_factor)
    assert subtitle.get_fontsize() == pytest.approx(12 * font_scale_factor)
    # title above subtitle above period, all inside the figure
    assert extents[0].y0 > extents[1].y1
    assert extents[1].y0 > extents[2].y1
    for extent in extents:
        assert 0 <= extent.x0 < extent.x1 <= 1
        assert 0 <= extent.y0 < extent.y1 <= 1


@pytest.mark.parametrize("mode", VALID_MODES)
def test_aggregate_zero_filled(mode: str) -> None:
    """Test aggregate_temporal_columns fills empty temporal bins with 0.

    Args:
        mode: Chart mode.
    """
    data = assign_temporal_columns(subset_data, "Date_Time", mode)
    result = aggregate_temporal_columns(data, "Date_Time", "count", mode)

    assert list(result.columns) == ["ring", "wedge", "count"]
    assert result["count"].notna().all()
    assert result["count"].sum() == len(subset_data)
    if mode in ("YEAR_MONTH", "YEAR_WEEK"):
        # a two week subset leaves most months & weeks empty
        assert result["count"].eq(0).any()
