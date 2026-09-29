"""Static directory data for the action parser: who's on Slack, who Alex has emailed,
and the current state of the calendar. Unlike the memory system this isn't governed
by as_of -- a real address book/calendar app just reflects current state, so we load
it once from the raw files rather than through the bi-temporal `visible()` view.
"""
import json
import re
from pathlib import Path


def load_directory(data_dir):
    d = Path(data_dir)
    slack_users = json.load(open(d / "connectors/slack/users.json"))
    slack_channels = json.load(open(d / "connectors/slack/channels.json"))

    # email contacts: anyone Alex has exchanged mail with, name -> email
    email_contacts = {}
    for line in open(d / "connectors/gmail/messages.jsonl"):
        x = json.loads(line)
        for field in ("from", "to", "cc"):
            vals = x.get(field) or []
            vals = [vals] if isinstance(vals, str) else vals
            for v in vals:
                m = re.match(r"^(.*?)\s*<([^>]+)>$", v.strip())
                if m:
                    name, email = m.group(1).strip(), m.group(2).strip()
                    if name:
                        email_contacts[name.lower()] = email
                elif "@" in v:
                    email_contacts[v.strip().lower()] = v.strip()

    events = [json.loads(l) for l in open(d / "connectors/google_calendar/events.jsonl")]
    # keep only the latest state per event id (file is one row per event already, but be safe)
    latest = {}
    for e in events:
        if e["id"] not in latest or e["updated"] > latest[e["id"]]["updated"]:
            latest[e["id"]] = e

    return {
        "slack_users": slack_users,
        "slack_channels": slack_channels,
        "email_contacts": email_contacts,
        "events": list(latest.values()),
    }
