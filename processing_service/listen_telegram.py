from __future__ import annotations

import json
from typing import Any

from config import settings
from pipeline.ingest import ingest_forever
from pipeline.telegram_adapter import TelegramAdapter, TelegramConfigError


def _settings_snapshot() -> dict[str, Any]:
    return {
        "telegram_enabled": settings.telegram_enabled,
        "telegram_session_file": settings.telegram_session_file,
        "telegram_supplier_channel": settings.telegram_supplier_channel,
        "telegram_request_timeout": settings.telegram_request_timeout,
    }


def main() -> int:
    print("Telegram listener settings:")
    print(json.dumps(_settings_snapshot(), indent=2))

    if not settings.telegram_enabled:
        print("Telegram listener is disabled. Set TELEGRAM_ENABLED=true to enable it.")
        return 0

    print("Listening for new messages. Press Ctrl+C to stop.")

    adapter = TelegramAdapter()
    try:
        adapter.connect()

        client = adapter._client
        authorized = bool(client and client.is_user_authorized())
        print(f"Telegram authorized: {authorized}")
        if not authorized:
            print("Telegram session is not authorized. Run the login flow first.")
            return 1

        def _print_message(post: dict[str, Any]) -> None:
            print(json.dumps(post, ensure_ascii=False))

        ingest_forever(["telegram"], _print_message)
        return 0
    except KeyboardInterrupt:
        print("Listener stopped.")
        return 0
    except TelegramConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except Exception as exc:
        print(f"Telegram listener failed: {exc}")
        return 1
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
