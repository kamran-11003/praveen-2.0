"""Thread-safe JSONL conversation logger with daily rotation.

Each line is one event:
    {"ts": "2026-05-13T10:21:55.123456", "sid": "abc...", "role": "user",
     "text": "...", "lang": "en", "event": "speech"}

Files: logs/YYYY-MM-DD.jsonl
"""

import datetime as _dt
import json
import logging
import os
import threading

log = logging.getLogger("conv_log")


class ConversationLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for_today(self) -> str:
        date = _dt.date.today().isoformat()
        return os.path.join(self.log_dir, f"{date}.jsonl")

    def log(
        self,
        sid: str,
        role: str,           # "user" | "assistant" | "system"
        text: str,
        lang: str = "en",    # "en" | "ur"
        event: str = "speech",  # "speech" | "barge_in" | "service_click" | "greeting"
    ) -> None:
        record = {
            "ts":    _dt.datetime.now().isoformat(timespec="microseconds"),
            "sid":   sid,
            "role":  role,
            "text":  text,
            "lang":  lang,
            "event": event,
        }
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock:
                with open(self._path_for_today(), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as exc:
            log.warning("conversation log write failed: %s", exc)

    def get_session(self, sid: str) -> list[dict]:
        """Return all log entries for a session id from today's file."""
        path = self._path_for_today()
        if not os.path.exists(path):
            return []
        out: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("sid") == sid:
                        out.append(rec)
        except OSError as exc:
            log.warning("conversation log read failed: %s", exc)
        return out
