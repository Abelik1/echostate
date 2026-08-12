import logging
import logging.handlers
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)
logging.Logger.trace = trace  # type: ignore

PLAIN_FMT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
DATE_FMT = "%H:%M:%S"

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            try:
                payload.update(record.extra)
            except Exception:
                payload["extra_error"] = "failed to merge extra"
        return json.dumps(payload, ensure_ascii=False)

def setup_logging(
    log_dir: str = "./logs",
    run_name: str | None = None,
    console_level: str | None = None,
    file_level: str | None = None,
    jsonl_file: bool = True,
    plain_file: bool = True,
    rotate_mb: int = 128,
    backups: int = 3,
) -> Dict[str, str]:
    """
    Configure root logging. One-shot, idempotent per process.
    Env overrides:
      ESN_LOG_LEVEL (console), ESN_FILE_LEVEL (files)
    """
    os.makedirs(log_dir, exist_ok=True)
    used = {}
    run_stamp = run_name or time.strftime("%Y%m%d_%H%M%S")

    console_level = (console_level or os.getenv("ESN_LOG_LEVEL") or "INFO").upper()
    file_level = (file_level or os.getenv("ESN_FILE_LEVEL") or "DEBUG").upper()

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(TRACE_LEVEL_NUM)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level))
    ch.setFormatter(logging.Formatter(PLAIN_FMT, datefmt=DATE_FMT))
    root.addHandler(ch)

    if plain_file:
        plain_path = os.path.join(log_dir, f"{run_stamp}.log")
        fh_plain = logging.handlers.RotatingFileHandler(
            plain_path, maxBytes=rotate_mb * 1024 * 1024, backupCount=backups, encoding="utf-8"
        )
        fh_plain.setLevel(getattr(logging, file_level))
        fh_plain.setFormatter(logging.Formatter(PLAIN_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh_plain)
        used["plain_log"] = plain_path

    if jsonl_file:
        json_path = os.path.join(log_dir, f"{run_stamp}.jsonl")
        fh_json = logging.handlers.RotatingFileHandler(
            json_path, maxBytes=rotate_mb * 1024 * 1024, backupCount=backups, encoding="utf-8"
        )
        fh_json.setLevel(getattr(logging, file_level))
        fh_json.setFormatter(JsonFormatter())
        root.addHandler(fh_json)
        used["jsonl_log"] = json_path

    return used

def write_run_summary_md(path: str, meta: Dict[str, Any], sections: Dict[str, Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    def kv(d):
        lines = []
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)):
                v = json.dumps(v, ensure_ascii=False)
            lines.append(f"- **{k}**: {v}")
        return "\n".join(lines) if lines else "_none_"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# ESN Run Summary — {meta.get('run_name', 'unnamed')}\n\n")
        f.write("## Metadata\n")
        f.write(kv(meta) + "\n\n")
        for title, content in sections.items():
            f.write(f"## {title}\n")
            f.write(kv(content) + "\n\n")
