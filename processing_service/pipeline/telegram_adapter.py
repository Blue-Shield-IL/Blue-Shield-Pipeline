"""Telegram ingestion adapter.

This module is additive to the existing pipeline. It does not alter the
current ingestion entrypoints and can be wired in later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from config import Settings, settings

TelegramRawPost = dict[str, Any]

try:
    from telethon import events
    from telethon.sync import TelegramClient
except ImportError:  # pragma: no cover - exercised through runtime guard
    events = None
    TelegramClient = None


class TelegramConfigError(RuntimeError):
    """Raised when Telegram adapter runtime settings are incomplete."""


class TelegramAdapter:
    """Thin wrapper around the Telegram client for supplier-channel reads."""

    def __init__(
        self,
        cfg: Settings = settings,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = cfg
        self._client_factory = client_factory or TelegramClient
        self._client: Any | None = None
        self._supplier_entity: Any | None = None

    def validate_settings(self) -> None:
        if not self.settings.telegram_enabled:
            return
        if self._client_factory is None:
            raise TelegramConfigError(
                "Telethon is not installed. Add the 'telethon' dependency before using Telegram."
            )
        if self.settings.telegram_api_id is None:
            raise TelegramConfigError("TELEGRAM_API_ID is required when Telegram is enabled.")
        if not self.settings.telegram_api_hash:
            raise TelegramConfigError("TELEGRAM_API_HASH is required when Telegram is enabled.")
        if not self.settings.telegram_supplier_channel:
            raise TelegramConfigError(
                "TELEGRAM_SUPPLIER_CHANNEL is required when Telegram is enabled."
            )

    def connect(self) -> None:
        self.validate_settings()
        if not self.settings.telegram_enabled:
            return
        if self._client is None:
            self._client = self._client_factory(
                self.settings.telegram_session_file,
                self.settings.telegram_api_id,
                self.settings.telegram_api_hash,
                timeout=self.settings.telegram_request_timeout,
            )
        self._client.connect()
        self._supplier_entity = None

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()
        self._supplier_entity = None

    def fetch_recent(self, limit: int) -> list[TelegramRawPost]:
        if limit <= 0:
            return []
        self.validate_settings()
        if not self.settings.telegram_enabled:
            return []

        self.connect()
        entity = self._resolve_supplier_entity()
        raw_posts: list[TelegramRawPost] = []
        for message in self._client.iter_messages(entity, limit=limit):
            raw_posts.append(self.normalize_message(message))
        return raw_posts

    def listen_forever(
        self,
        on_message: Callable[[TelegramRawPost], None],
        *,
        max_messages: int | None = None,
    ) -> None:
        """Keep one Telegram connection open and stream supplier messages."""
        self.validate_settings()
        if not self.settings.telegram_enabled:
            return
        if events is None:
            raise TelegramConfigError(
                "Telethon events are unavailable. Add the 'telethon' dependency before listening."
            )

        self.connect()
        entity = self._resolve_supplier_entity()
        seen_messages = 0

        @self._client.on(events.NewMessage(chats=entity))
        async def _handle_new_message(event: Any) -> None:
            nonlocal seen_messages
            on_message(self.normalize_message(event.message))
            seen_messages += 1
            if max_messages is not None and seen_messages >= max_messages:
                await event.client.disconnect()

        self._client.run_until_disconnected()

    def normalize_message(self, message: Any) -> TelegramRawPost:
        channel_identity = self._channel_name(message)
        created_at = self._coerce_datetime(getattr(message, "date", None))

        return {
            "content": self._extract_text(message),
            "channel": channel_identity,
            "creation_time": created_at.isoformat(),
        }

    def _resolve_supplier_entity(self) -> Any:
        if self._supplier_entity is None:
            self._supplier_entity = self._client.get_entity(
                self.settings.telegram_supplier_channel
            )
        return self._supplier_entity

    def _extract_text(self, message: Any) -> str:
        text = getattr(message, "message", None) or getattr(message, "text", None)
        if text is None:
            text = getattr(message, "raw_text", None)
        if text is None:
            text = getattr(message, "caption", None)
        if text is None:
            return ""
        normalized = str(text).strip()
        return normalized

    def _channel_name(self, message: Any) -> str:
        chat = getattr(message, "chat", None)
        peer_id = getattr(message, "peer_id", None)

        username = getattr(chat, "username", None)
        if username:
            return f"@{username}"

        title = getattr(chat, "title", None)
        if title:
            return str(title)

        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            return str(chat_id)

        for attr in ("channel_id", "chat_id", "user_id"):
            value = getattr(peer_id, attr, None)
            if value is not None:
                return str(value)

        if self.settings.telegram_supplier_channel:
            return self.settings.telegram_supplier_channel

        return "unknown"

    def _coerce_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)
        return datetime.now(UTC)
