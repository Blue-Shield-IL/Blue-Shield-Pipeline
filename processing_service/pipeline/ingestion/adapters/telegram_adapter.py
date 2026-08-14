"""Telegram ingestion adapter.

This module provides Telegram ingestion support for the pipeline and backs
the Telegram fetch/listen ingestion entrypoints.
"""

from __future__ import annotations

import logging, os, json, time, re
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
    from telethon.tl.types import InputPeerChannel
except ImportError:
    events = None
    TelegramClient = None
    InputPeerChannel = None

_global_supplier_entities: list[Any] | None = None
_global_channel_descriptions: dict[str, str] = {}


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
        self._channel_descriptions = _global_channel_descriptions

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

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()

    def _get_last_fetched_date(self) -> datetime | None:
        if os.path.exists("cursors/.telegram_cursor.json"):
            try:
                with open("cursors/.telegram_cursor.json", "r") as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data["last_date"])
            except Exception:
                pass
        return None

    def _set_last_fetched_date(self, dt: datetime) -> None:
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
            time.sleep(1.5)
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
            try:
                logger.info("New message received from listener",
                            extra={"payload": {"chat_id": str(getattr(event.message, "chat_id", "unknown"))}})
                on_message(self.normalize_message(event.message))
            except Exception as e:
                logger.error("Failed to handle new message", extra={"error": str(e)})

        self._client.run_until_disconnected()

    def normalize_message(self, message: Any) -> TelegramRawPost:
        created_at = self._coerce_datetime(getattr(message, "date", None))

        text_content = self._extract_text(message)
        hashtags = list(set(re.findall(r'#\w+', text_content)))
        mentions = list(set(re.findall(r'@\w+', text_content)))

        author_info = self._extract_author_info(message)
        channel_info = self._extract_channel_info(message)

        chat_id = channel_info.get("id") or ""
        channel_description = self._channel_descriptions.get(chat_id)
        channel_info["description"] = channel_description

        views = getattr(message, "views", 0) or 0
        shares = getattr(message, "forwards", 0) or 0

        comments_count = 0
        replies_obj = getattr(message, "replies", None)
        if replies_obj and hasattr(replies_obj, "replies"):
            comments_count = getattr(replies_obj, "replies", 0) or 0

        likes = 0
        reactions_obj = getattr(message, "reactions", None)
        if reactions_obj and hasattr(reactions_obj, "results"):
            likes = sum(getattr(r, "count", 0) for r in reactions_obj.results)

        return {
            "post_id": self._post_id(message, channel_info),
            "text_content": text_content,
            "author": author_info,
            "channel": channel_info,
            "platform": "telegram",
            "hashtags": hashtags,
            "mentions": mentions,
            "created_at": created_at.isoformat(),
            "views": views,
            "shares": shares,
            "comments_count": comments_count,
            "likes": likes,
        }

    def _load_channel_cache(self) -> dict[str, Any]:
        cache_path = "cursors/.telegram_channels.json"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _resolve_supplier_entities(self) -> list[Any]:
        global _global_supplier_entities
        if _global_supplier_entities is not None:
            return _global_supplier_entities

        cache = self._load_channel_cache()
        entities = []

        for channel in self.settings.telegram_supplier_channels:
            cached = cache.get(channel)
            if cached and InputPeerChannel is not None:
                peer = InputPeerChannel(
                    channel_id=cached["id"],
                    access_hash=cached["access_hash"],
                )
                entities.append(peer)
                desc = cached.get("description")
                if desc:
                    self._channel_descriptions[str(cached["id"])] = desc
                logger.info("Loaded channel from cache", extra={"payload": {"channel": channel}})
            else:
                try:
                    time.sleep(1.5)
                    entity = self._client.get_entity(channel)
                    entities.append(entity)
                    logger.info("Resolved Telegram channel via API", extra={"payload": {"channel": channel}})
                except Exception as e:
                    logger.error("Failed to resolve Telegram channel",
                                 extra={"payload": {"channel": channel}, "error": str(e)})

        _global_supplier_entities = entities
        return _global_supplier_entities

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

    def _extract_channel_info(self, message: Any) -> dict[str, str | None]:
        chat = getattr(message, "chat", None)
        peer_id = getattr(message, "peer_id", None)

        channel_username = getattr(chat, "username", None)
        if channel_username:
            channel_username = str(channel_username)

        channel_name = getattr(chat, "title", None)
        if channel_name:
            channel_name = str(channel_name)

        channel_id = getattr(chat, "id", None)
        if channel_id is not None:
            channel_id = str(channel_id)

        if not channel_id:
            for attr in ("channel_id", "chat_id", "user_id"):
                value = getattr(peer_id, attr, None)
                if value is not None:
                    channel_id = str(value)
                    break

        return {
            "id": channel_id,
            "name": channel_name,
            "username": channel_username
        }

    def _extract_author_info(self, message: Any) -> dict[str, str | None]:
        sender = getattr(message, "sender", None)
        phone = getattr(sender, "phone", None) if sender else None
        formatted_phone = f"+{phone}" if phone else None

        author_id = None
        author_name = None
        author_username = None

        if sender:
            author_username = getattr(sender, "username", None)
            if author_username:
                author_username = str(author_username)
            author_id = getattr(sender, "id", None)
            if author_id is not None:
                author_id = str(author_id)

            first_name = getattr(sender, "first_name", None)
            last_name = getattr(sender, "last_name", None)
            full_name = " ".join(part for part in (first_name, last_name) if part)
            if full_name:
                author_name = full_name

        if not author_name and not author_username:
            post_author = getattr(message, "post_author", None)
            if post_author:
                author_name = str(post_author)

        if not author_name and not author_username and not author_id:
            fwd_from = getattr(message, "fwd_from", None)
            if fwd_from:
                from_name = getattr(fwd_from, "from_name", None)
                if from_name:
                    author_name = str(from_name)
                else:
                    from_id = getattr(fwd_from, "from_id", None)
                    if from_id:
                        user_id = getattr(from_id, "user_id", None)
                        if user_id:
                            author_id = str(user_id)

        return {
            "id": author_id,
            "name": author_name,
            "username": author_username,
            "phone": formatted_phone
        }

    def _post_id(self, message: Any, channel_info: dict[str, str | None]) -> str:
        chat_key = channel_info.get("id") or channel_info.get("username") or "unknown"
        message_id = getattr(message, "id", None)
        if message_id is not None:
            return f"telegram:{chat_key}:{message_id}"

        created_at = self._coerce_datetime(getattr(message, "date", None))
        return f"telegram:{chat_key}:{created_at.isoformat()}"

    def _coerce_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return datetime.now(timezone.utc)
