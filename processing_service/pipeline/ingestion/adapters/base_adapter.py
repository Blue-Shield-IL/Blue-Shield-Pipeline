"""Base adapter interface for data ingestion."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

RawPost = dict[str, Any]

class BaseAdapter(ABC):
    """Abstract base class for all ingestion adapters."""

    @abstractmethod
    def validate_settings(self) -> None:
        """Validate that all required settings are present."""
        pass

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the data source."""
        pass

    @abstractmethod
    def fetch_recent(self, limit: int) -> list[RawPost]:
        """Fetch a specified number of recent posts from the data source."""
        pass

    @abstractmethod
    def listen_forever(
        self,
        on_message: Callable[[RawPost], None],
        *,
        max_messages: int | None = None,
    ) -> None:
        """Listen to the data source continuously and call on_message for each new post."""
        pass
