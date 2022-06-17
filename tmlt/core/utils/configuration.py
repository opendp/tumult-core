"""Configuration properties for Tumult Core."""

# SPDX-License-Identifier: Apache-2.0

import time
from uuid import uuid4


class Config:
    """Global configuration for programs using Core."""

    _temp_db_name = f'tumult_temp_{time.strftime("%Y%m%d_%H%M%S")}_{uuid4().hex}'

    @classmethod
    def temp_db_name(cls) -> str:
        """Get the name of the temporary database that Tumult Core uses."""
        return cls._temp_db_name
