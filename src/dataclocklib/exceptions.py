"""Exception module for chart creation errors.

Classes:
    AggregationColumnError: Raised on missing or unsuitable (non-numeric or
        reserved name) aggregation column.
    AggregationFunctionError: Raised on unexpected aggregation function.
    EmptyDataFrameError: Raised on empty DataFrame.
    ModeError: Raised on incorrect chart mode value.
    MissingDatetimeError: Raised on missing expected datetime64 dtype or
        missing (NaT) datetimes.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

# Iterable & DataFrame kept at runtime (not TYPE_CHECKING) so
# typing.get_type_hints resolves them for the public exception __init__s.
from collections.abc import Iterable  # noqa: TC003

from pandas import DataFrame  # noqa: TC002


class AggregationColumnError(ValueError):
    """Raised on a missing or unsuitable aggregation column."""

    def __init__(self, agg: str, reason: str | None = None) -> None:
        """Initialise AggregationColumnError exception.

        Args:
            agg (str): Aggregation function.
            reason (str, optional): Why the agg_column is unsuitable; the
                agg_column is reported as missing if None.
        """
        if reason is None:
            msg = f"Expected agg_column for aggregation function {agg}."
        else:
            msg = f"Unsuitable agg_column for aggregation function {agg}: "
            msg += f"{reason}."
        super().__init__(msg)


class AggregationFunctionError(ValueError):
    """Raised on unexpected aggregation function."""

    def __init__(self, agg: str, valid_functions: Iterable[str]) -> None:
        """Initialise AggregationFunctionError exception.

        Args:
            agg (str): Aggregation function.
            valid_functions (Iterable[str]): Valid aggregation functions.
        """
        msg = f"Unexpected aggregation function ({agg}): {valid_functions}."
        super().__init__(msg)


class EmptyDataFrameError(ValueError):
    """Raised on empty DataFrame."""

    def __init__(self, data: DataFrame) -> None:
        """Initialise EmptyDataFrameError exception.

        Args:
            data (DataFrame): The empty DataFrame.
        """
        msg = f"Unexpected empty DataFrame - {data.empty=}."
        super().__init__(msg)


class ModeError(ValueError):
    """Raised on incorrect chart mode value."""

    def __init__(self, mode: str, valid_modes: Iterable[str]) -> None:
        """Initialise ModeError exception.

        Args:
            mode (str): Incorrect mode value.
            valid_modes (Iterable[str]): Valid mode values.
        """
        msg = f"Unexpected mode value ({mode}): {valid_modes}."
        super().__init__(msg)


class MissingDatetimeError(ValueError):
    """Raised on missing expected datetime64 dtype or missing datetimes."""

    def __init__(self, column: str, reason: str | None = None) -> None:
        """Initialise MissingDatetimeError exception.

        Args:
            column (str): Name of the invalid date_column.
            reason (str, optional): Why the date_column is invalid; the
                column is reported as not naive datetime64 dtype if None.
        """
        if reason is None:
            msg = (
                f"Expected naive datetime64 dtype for date_column ({column})."
            )
        else:
            msg = f"Invalid date_column ({column}): {reason}."
        super().__init__(msg)
