"""LLM-based dry-run command parser. Used first when a key is configured -- much
more robust to phrasing the rule-based parser (actions/rules.py) hasn't seen, since
it can actually read the directory and reason about which entry a name refers to.
Falls back to the rule-based parser on any failure so the pipeline always produces
an output.
"""
import json
import re

from memory.llm_client import LLMError, call_llm

ACTION_TYPES = """
slack.send_message  args: to (a Slack user id like U03SARAHK, a DM id like D-ALEX-SARAHK, or a channel id like C10RP), text
gmail.send          args: to (list of emails), cc (list), subject, body
calendar.create_event  args: title, start, end (ISO 8601 with -07:00 offset), attendees (list of emails)
calendar.update_event  args: event_id, plus any changed fields (start/end/title/...)
reminder.create     args: text, due (ISO 8601 with -07:00 offset)
memory.ask          args: question (use this when the command is really a question, not an action)
app.open            args: app
clarify             args: question (use when the command is ambiguous, e.g. two people share a first name)
confirm             args: summary (use for destructive commands like bulk delete, before doing them)
"""

SYSTEM = (
    "You convert a spoken command from Alex Rivera into a DRY-RUN list of actions, using ONLY "
    "the directory data given (Slack users/channels, email contacts, calendar events). Never "
    "invent an id, email or event that isn't in the directory. Times are America/Los_Angeles "
    "(-07:00). If a name is ambiguous (matches more than one real person), use `clarify` instead "
    "of guessing. If the command is destructive (bulk delete/cancel), use `confirm` first instead "
    "of doing it. If the command is actually a question, use `memory.ask`.\n\n"
    f"Action types:\n{ACTION_TYPES}\n"
    "Respond with JSON only: {\"actions\": [{\"type\": \"...\", \"args\": {...}}, ...]}"
)


def _call(prompt):
    return call_llm(prompt, system=SYSTEM, max_tokens=500)


def _directory_summary(directory):
    users = [{"id": u["id"], "name": u["real_name"], "email": u["email"]}
             for u in directory["slack_users"] if not u.get("is_bot")]
    channels = [{"id": c["id"], "name": c["name"], "is_dm": c["is_dm"]} for c in directory["slack_channels"]]
    events = [{"id": e["id"], "title": e["summary"], "start": e["start"], "end": e["end"]}
              for e in directory["events"] if e.get("status") != "cancelled"]
    contacts = dict(list(directory["email_contacts"].items())[:60])
    return json.dumps({"slack_users": users, "slack_channels": channels,
                        "calendar_events": events, "email_contacts": contacts})


def llm_parse(command, as_of_str, directory):
    prompt = f"as_of: {as_of_str}\nCommand: {command}\n\nDirectory:\n{_directory_summary(directory)}"
    try:
        raw = _call(prompt)
        if raw is None:
            return None
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        actions = parsed.get("actions")
        return actions if isinstance(actions, list) else None
    except Exception as e:
        print(f"  [llm action parse failed, falling back to rules: {e}]")
        return None
