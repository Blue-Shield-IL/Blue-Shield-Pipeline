import logging
import sys
from pythonjsonlogger import json


def setup_logging() -> None:
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = json.JsonFormatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
        rename_fields={"levelname": "level", "asctime": "timestamp"}
    )
    log_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[log_handler])
