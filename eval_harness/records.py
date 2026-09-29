"""Index of every citable id in data/: when it became available, which record it belongs to,
its text, and edits/deletions. Used by the scorers (time-leak and deletion checks). Standard library only.

A unit is anything a system can cite: a meeting segment, a dictation, a Slack message or edit,
an email, a calendar event, a Codex session, a ChatGPT message. A record is what contains it
(a meeting, a ChatGPT conversation); for most sources unit and record are the same id.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Unit:
    id: str
    record: str
    time: datetime
    text: str


def _dt(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def load(data_dir):
    d = Path(data_dir)
    units = []
    deleted = {}   # target id -> deletion time
    edits = {}     # target id -> [(time, new text)]

    for f in sorted((d / "native/meetings").glob("*.json")):
        m = json.loads(f.read_text())
        start = _dt(m["start"])
        for s in m["segments"]:
            who = s.get("speaker_name") or s.get("speaker_label") or "Unknown speaker"
            units.append(Unit(s["seg_id"], m["id"], start + timedelta(seconds=s["end_s"]),
                              f"[{m['title']}, {m['start'][:10]}] {who}: {s['text']}"))

    for line in open(d / "native/dictation/dictations.jsonl"):
        x = json.loads(line)
        units.append(Unit(x["id"], x["id"], _dt(x["timestamp"]),
                          f"[Dictation {x['mode']} into {x['target_app']} – {x['target_context']}, "
                          f"{x['delivery_state']}] {x['cleaned_text']}"
                          f"{chr(10) + '(raw transcript: ' + x['raw_transcript'] + ')' if x.get('raw_transcript') else ''}"))

    names = {u["id"]: u["real_name"] for u in json.load(open(d / "connectors/slack/users.json"))}
    chans = {c["id"]: c["name"] for c in json.load(open(d / "connectors/slack/channels.json"))}
    for line in open(d / "connectors/slack/messages.jsonl"):
        x = json.loads(line)
        t = _dt(x["ts"])
        where = chans.get(x["channel_id"], x["channel_id"])
        if x.get("subtype") == "message_deleted":
            deleted[x["target_id"]] = t
            units.append(Unit(x["id"], x["id"], t, f"[Slack {where}] (message {x['target_id']} was deleted)"))
            continue
        if x.get("subtype") == "message_changed":
            edits.setdefault(x["target_id"], []).append((t, x["text"]))
            units.append(Unit(x["id"], x["id"], t, f"[Slack {where}, edit of {x['target_id']}] {x['text']}"))
            continue
        who = names.get(x.get("user"), x.get("bot_name") or x.get("user"))
        units.append(Unit(x["id"], x["id"], t, f"[Slack {where}] {who}: {x['text']}"))

    for line in open(d / "connectors/gmail/messages.jsonl"):
        x = json.loads(line)
        units.append(Unit(x["id"], x["id"], _dt(x["date"]),
                          f"[Email {x['date'][:16]}] From {x['from']} To {', '.join(x['to'])}"
                          f"{' Cc ' + ', '.join(x['cc']) if x['cc'] else ''} | {x['subject']}\n{x['body']}"))

    for line in open(d / "connectors/google_calendar/events.jsonl"):
        x = json.loads(line)
        st, en = x["start"], x["end"]
        when = f"{st.get('dateTime') or st.get('date')} to {en.get('dateTime') or en.get('date')}"
        att = ", ".join(a["email"] for a in x.get("attendees", []))
        units.append(Unit(x["id"], x["id"], _dt(x["updated"]),
                          f"[Calendar, {x['status']}] {x['summary']} | {when} | {x.get('location') or ''} | "
                          f"attendees: {att} | {x.get('description') or ''}"
                          f"{' | repeats ' + str(x['recurrence']) if x.get('recurrence') else ''}"))

    for f in sorted((d / "connectors/codex/sessions").glob("*.jsonl")):
        events = [json.loads(l) for l in open(f)]
        meta, body = events[0], events[1:]
        text = "\n".join(f"{e.get('role', e.get('tool', e['type']))}: {e.get('content') or e.get('input', '')}"
                         for e in body)
        units.append(Unit(meta["id"], meta["id"], _dt(body[-1]["timestamp"] if body else meta["started_at"]),
                          f"[Codex session, repo {meta.get('repo')}]\n{text}"))

    for c in json.load(open(d / "connectors/chatgpt/conversations.json")):
        for m in c["messages"]:
            units.append(Unit(m["id"], c["id"], _dt(m["create_time"]),
                              f"[ChatGPT '{c['title']}'] {m['role']}: {m['content']}"))

    units.sort(key=lambda u: u.time)
    return units, deleted, edits


def context(data_dir):
    """What the scorers need: availability time, record of each unit, word counts, deletions."""
    units, deleted, _ = load(data_dir)
    avail = {u.id: u.time for u in units}
    record_of = {u.id: u.record for u in units}
    for u in units:  # a record id (a whole meeting or conversation) exists once its first unit does
        if u.record not in avail or u.time < avail[u.record]:
            avail[u.record] = min(u.time, avail.get(u.record, u.time))
    words = {u.id: len(u.text.split()) for u in units}
    return {"avail": avail, "record_of": record_of, "words": words, "deleted": deleted}


def visible(data_dir, as_of):
    """Units a system could know at `as_of`: delivered by then, not deleted by then, edits applied."""
    units, deleted, edits = load(data_dir)
    out = []
    for u in units:
        if u.time > as_of or (u.id in deleted and deleted[u.id] <= as_of):
            continue
        newer = [txt for t, txt in edits.get(u.id, []) if t <= as_of]
        out.append(Unit(u.id, u.record, u.time, u.text if not newer else u.text.partition(": ")[0] + ": " + newer[-1] + " (edited)"))
    return out
