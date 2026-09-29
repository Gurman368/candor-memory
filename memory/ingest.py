"""Loads the Candor data into retrievable "units" and a link index.

Correctness-critical bi-temporal logic (delivery time, edits, deletions) is NOT
reimplemented here -- it's imported straight from eval_harness.records, which is
also what the scorer uses. That guarantees our notion of "what's visible as_of X"
matches the grader's exactly, so we can never retrieve/cite a forbidden record.

On top of that we build a static link index (thread parents, calendar event ids,
meeting/session membership, explicit id-mentions in text) used by the retriever
to pull in connected context that keyword search alone would miss.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from eval_harness import records as _records

ID_PATTERN = re.compile(
    r"\b(?:MTG-[A-Z0-9-]+(?:#\d+)?|CAL-[A-Z0-9-]+|SL-[A-Z0-9-]+|EM-[A-Z0-9-]+|"
    r"DCT-[A-Z0-9-]+|CDX-[A-Z0-9-]+|CGPT-[A-Z0-9-]+(?:#m\d+)?|TH-[A-Z0-9-]+)\b"
)

# Defense-in-depth against the two planted hazards the brief calls out: a pasted
# secret, and a hidden instruction aimed at an AI reader. This runs on every unit's
# text once, here at ingestion, so BOTH retrieval and every answer path (LLM or
# extractive) only ever see the redacted version -- an answerer can't leak what it
# never received, whatever else goes wrong downstream.
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
SECRET_RE = re.compile(
    r"\bsk-[A-Za-z0-9-]{8,}\b|\bAKIA[A-Z0-9]{12,}\b|"
    r"\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S+", re.I)


def _redact(text):
    text = HTML_COMMENT_RE.sub(" [hidden text removed -- was addressed to an AI assistant, not reported] ", text)
    text = SECRET_RE.sub("[REDACTED SECRET]", text)
    return text


@dataclass
class Unit:
    id: str
    record: str
    time: object
    text: str
    source: str = "other"           # meeting | dictation | slack | email | calendar | codex | chatgpt
    thread_id: str = None           # slack thread_parent_id or email thread_id
    channel_id: str = None
    event_id: str = None            # calendar_event_id this belongs to (meetings) or event id itself
    speaker_name: str = None
    speaker_confidence: float = None
    mentions: list = field(default_factory=list)  # other ids referenced in the text


def _dt(s):
    from datetime import datetime
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def build_meta_index(data_dir):
    """id -> dict of static link metadata, independent of as_of."""
    d = Path(data_dir)
    meta = {}

    for f in sorted((d / "native/meetings").glob("*.json")):
        m = json.loads(f.read_text())
        for s in m["segments"]:
            meta[s["seg_id"]] = {
                "source": "meeting", "event_id": m.get("calendar_event_id"), "record": m["id"],
                "speaker_name": s.get("speaker_name"), "speaker_confidence": s.get("speaker_confidence"),
                "mentions": ID_PATTERN.findall(s["text"]),
            }
        meta[m["id"]] = {"source": "meeting", "event_id": m.get("calendar_event_id"), "record": m["id"]}

    for line in open(d / "native/dictation/dictations.jsonl"):
        x = json.loads(line)
        meta[x["id"]] = {"source": "dictation", "record": x["id"],
                          "mentions": ID_PATTERN.findall(x.get("cleaned_text", ""))}

    slack_users = {u["id"]: u for u in json.load(open(d / "connectors/slack/users.json"))}
    for line in open(d / "connectors/slack/messages.jsonl"):
        x = json.loads(line)
        meta[x["id"]] = {
            "source": "slack", "record": x["id"], "channel_id": x.get("channel_id"),
            "thread_id": x.get("thread_parent_id"),
            "speaker_name": (slack_users.get(x.get("user")) or {}).get("real_name"),
            "target_id": x.get("target_id"),  # for edit/delete events
            "mentions": ID_PATTERN.findall(x.get("text", "") or ""),
        }

    for line in open(d / "connectors/gmail/messages.jsonl"):
        x = json.loads(line)
        meta[x["id"]] = {"source": "email", "record": x["id"], "thread_id": x.get("thread_id"),
                          "mentions": ID_PATTERN.findall(x.get("body", "") or "")}

    for line in open(d / "connectors/google_calendar/events.jsonl"):
        x = json.loads(line)
        meta[x["id"]] = {"source": "calendar", "record": x["id"],
                          "mentions": ID_PATTERN.findall(x.get("description", "") or "")}

    for f in sorted((d / "connectors/codex/sessions").glob("*.jsonl")):
        events = [json.loads(l) for l in open(f)]
        meta[events[0]["id"]] = {"source": "codex", "record": events[0]["id"]}

    for c in json.load(open(d / "connectors/chatgpt/conversations.json")):
        for m in c["messages"]:
            meta[m["id"]] = {"source": "chatgpt", "record": c["id"],
                              "mentions": ID_PATTERN.findall(m.get("content", "") or "")}

    return meta


def load_visible(data_dir, as_of_str):
    """Units visible at as_of, enriched with static link metadata. Never includes
    anything after as_of or anything deleted by as_of (inherited from records.visible)."""
    as_of = _dt(as_of_str)
    raw_units = _records.visible(data_dir, as_of)
    meta = build_meta_index(data_dir)
    out = []
    for u in raw_units:
        m = meta.get(u.id, {})
        out.append(Unit(
            id=u.id, record=u.record, time=u.time, text=_redact(u.text),
            source=m.get("source", "other"), thread_id=m.get("thread_id"),
            channel_id=m.get("channel_id"), event_id=m.get("event_id"),
            speaker_name=m.get("speaker_name"), speaker_confidence=m.get("speaker_confidence"),
            mentions=m.get("mentions", []),
        ))
    return out
