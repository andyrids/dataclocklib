"""Exception module for chart creation errors.

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

Classes:
    AggregationError: Raised on incompatible dtype values and function.
    ModeError: Raised on incorrect chart mode value.

Types:
    Aggregation: Keys representing aggregation functions.
    Mode: Keys representing temporal bins used in each chart.
"""
from typing import Iterable


class ModeError(ValueError):
    """Raised on incorrect chart mode value."""

    def __init__(self, mode: str, valid_modes: Iterable):
        """Initialise ModeError exception.

        Args:
            mode (str): Incorrect mode value.
        """
        msg = f"Unexpected mode value ({mode}): {valid_modes}"
        super().__init__(msg)


class AggregationError(ValueError):
    """Raised on incompatible aggregation value & function combination."""

    def __init__(self, agg: str, agg_column: str):
        """Initialise AggregationError exception.

        Args:
            agg (str): Aggregation function.
            agg_column (str): Aggregation column.
        """
        msg = f"Expected numeric dtype for agg_column & agg function {agg}."
        super().__init__(msg)
