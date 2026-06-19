"""Telegram ingestion adapter.

This module provides Telegram ingestion support for the pipeline and backs
the Telegram fetch/listen ingestion entrypoints.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from telethon.tl.functions.channels import GetFullChannelRequest

from config import Settings, settings
from .base_adapter import BaseAdapter, RawPost

TelegramRawPost = RawPost
logger = logging.getLogger(__name__)

try:
    from telethon import events
    from telethon.sync import TelegramClient
except ImportError:
    events = None
    TelegramClient = None


class TelegramConfigError(RuntimeError):
    """Raised when Telegram adapter runtime settings are incomplete."""


class TelegramAdapter(BaseAdapter):
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
        self._supplier_entities: list[Any] | None = None
        self._channel_descriptions: dict[str, str] = {}

    def validate_settings(self) -> None:
        if self._client_factory is None:
            raise TelegramConfigError(
                "Telethon is not installed. Add the 'telethon' dependency before using Telegram."
            )
        if self.settings.telegram_api_id is None:
            raise TelegramConfigError("TELEGRAM_API_ID is required when Telegram is enabled.")
        if not self.settings.telegram_api_hash:
            raise TelegramConfigError("TELEGRAM_API_HASH is required when Telegram is enabled.")
        if not self.settings.telegram_supplier_channels:
            raise TelegramConfigError(
                "TELEGRAM_SUPPLIER_CHANNELS is required when Telegram is enabled."
            )

    def connect(self) -> None:
        self.validate_settings()
        if self._client is None:
            self._client = self._client_factory(
                self.settings.telegram_session_file,
                self.settings.telegram_api_id,
                self.settings.telegram_api_hash,
                timeout=self.settings.telegram_request_timeout
            )
        self._client.connect()
        self._supplier_entities = None

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()
        self._supplier_entities = None

    def _get_last_fetched_date(self) -> datetime | None:
        import os, json
        if os.path.exists("cursors/.telegram_cursor.json"):
            try:
                with open("cursors/.telegram_cursor.json", "r") as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data["last_date"])
            except Exception:
                pass
        return None

    def _set_last_fetched_date(self, dt: datetime) -> None:
        import os, json
        os.makedirs("cursors", exist_ok=True)
        with open("cursors/.telegram_cursor.json", "w") as f:
            json.dump({"last_date": dt.isoformat()}, f)

    def fetch_recent(self, limit: int) -> list[TelegramRawPost]:
        if limit <= 0:
            return []

        self.connect()
        entities = self._resolve_supplier_entities()

        last_date = self._get_last_fetched_date()
        max_date_seen = None
        raw_posts: list[TelegramRawPost] = []

        for entity in entities:
            for message in self._client.iter_messages(entity, limit=limit):
                msg_date = self._coerce_datetime(getattr(message, "date", None))

                if last_date and msg_date <= last_date:
                    break

                if max_date_seen is None or msg_date > max_date_seen:
                    max_date_seen = msg_date

                normalized = self.normalize_message(message)
                if normalized.get("text_content", ""):
                    raw_posts.append(normalized)

        if max_date_seen:
            self._set_last_fetched_date(max_date_seen)

        return raw_posts

    def listen_forever(
        self,
        on_message: Callable[[TelegramRawPost], None],
        *,
        max_messages: int | None = None,
    ) -> None:
        """Keep one Telegram connection open and stream supplier messages."""
        if events is None:
            raise TelegramConfigError(
                "Telethon events are unavailable. Add the 'telethon' dependency before listening."
            )

        self.connect()
        assert self._client is not None
        entities = self._resolve_supplier_entities()
        seen_messages = 0

        @self._client.on(events.NewMessage(chats=entities))
        async def _handle_new_message(event: Any) -> None:
            on_message(self.normalize_message(event.message))

        self._client.run_until_disconnected()

    def normalize_message(self, message: Any) -> TelegramRawPost:
        import re
        created_at = self._coerce_datetime(getattr(message, "date", None))

        text_content = self._extract_text(message)
        hashtags = list(set(re.findall(r'#\w+', text_content)))
        mentions = list(set(re.findall(r'@\w+', text_content)))

        print(message)
        sender = getattr(message, "sender", None)
        phone = getattr(sender, "phone", None)

        chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
        channel_description = self._channel_descriptions.get(chat_id)

        return {
            "post_id": self._post_id(message),
            "text_content": text_content,
            "author": self._author_name(message),
            "channel": self._channel_name(message),
            "platform": "telegram",
            "hashtags": hashtags,
            "mentions": mentions,
            "created_at": created_at.isoformat(),
            "author_phone": f"+{phone}" if phone else None,
            "channel_description": channel_description,
        }

    def _resolve_supplier_entities(self) -> list[Any]:
        if self._supplier_entities is None:
            entities = []
            for channel in self.settings.telegram_supplier_channels:
                try:
                    entity = self._client.get_entity(channel)
                    entities.append(entity)
                    logger.info("Successfully resolved Telegram channel", extra={"payload": {"channel": channel}})

                    if str(entity.id) not in self._channel_descriptions:
                        try:
                            full = self._client(GetFullChannelRequest(channel=entity))
                            about = getattr(full.full_chat, "about", None)
                            if about:
                                self._channel_descriptions[str(entity.id)] = about
                        except Exception as e:
                            logger.warning("Could not fetch full channel info",
                                           extra={"payload": {"channel": channel}, "error": str(e)})
                except Exception as e:
                    logger.error("Failed to resolve Telegram channel",
                                 extra={"payload": {"channel": channel}, "error": str(e)})
            self._supplier_entities = entities
        return self._supplier_entities

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

        return "unknown"

    def _author_name(self, message: Any) -> str:
        sender = getattr(message, "sender", None)

        username = getattr(sender, "username", None)
        if username:
            return f"@{username}"

        first_name = getattr(sender, "first_name", None)
        last_name = getattr(sender, "last_name", None)
        full_name = " ".join(part for part in (first_name, last_name) if part)
        if full_name:
            return full_name

        sender_id = getattr(sender, "id", None)
        if sender_id is not None:
            return str(sender_id)

        return self._channel_name(message)

    def _post_id(self, message: Any) -> str:
        chat_key = self._channel_key(message)
        message_id = getattr(message, "id", None)
        if message_id is not None:
            return f"telegram:{chat_key}:{message_id}"

        created_at = self._coerce_datetime(getattr(message, "date", None))
        return f"telegram:{chat_key}:{created_at.isoformat()}"

    def _channel_key(self, message: Any) -> str:
        chat = getattr(message, "chat", None)
        peer_id = getattr(message, "peer_id", None)

        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            return str(chat_id)

        for attr in ("channel_id", "chat_id", "user_id"):
            value = getattr(peer_id, attr, None)
            if value is not None:
                return str(value)

        username = getattr(chat, "username", None)
        if username:
            return str(username)

        return "unknown"

    def _coerce_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return datetime.now(timezone.utc)
