"""Exception module for chart creation errors.

Classes:
    AggregationColumnError: Raised on missing aggregation column.
    AggregationFunctionError: Raised on unexpected aggregation function.
    EmptyDataFrameError: Raised on empty DataFrame.
    ModeError: Raised on incorrect chart mode value.
    MissingDatetimeError: Raised on missing expected datetime64 dtype.

License:
    SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

# Iterable & DataFrame kept at runtime (not TYPE_CHECKING) so
# typing.get_type_hints resolves them for the public exception __init__s.
from collections.abc import Iterable  # noqa: TC003

from pandas import DataFrame  # noqa: TC002


class AggregationColumnError(ValueError):
    """Raised on missing aggregation column."""

    def __init__(self, agg: str) -> None:
        """Initialise AggregationColumnError exception.

        Args:
            agg (str): Aggregation function.
        """
        msg = f"Expected agg_column for aggregation function {agg}."
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
    """Raised on missing expected datetime64 dtype."""

    def __init__(self, column: str) -> None:
        """Initialise MissingDatetimeError exception.

        Args:
            column (str): Name of the column without a naive datetime64 dtype.
        """
        msg = f"Expected naive datetime64 dtype for date_column ({column})."
        super().__init__(msg)
