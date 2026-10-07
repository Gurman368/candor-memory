"""Rule-based command -> dry-run action(s) parser (see actions/llm_parse.py for the
optional LLM path, which is used first when a key is configured).

Design: rather than one big regex per exact phrasing, this resolves names/channels/
events by looking them up in the directory (actions/directory.py) built from the
data itself, and a small generic relative-date parser. That's what lets it handle
commands worded differently from the 12 training examples, as long as they name a
real person/channel/event the data actually has.
"""
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

LOCAL = ZoneInfo("America/Los_Angeles")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Trigger words that mean "look this fact up", e.g. "the corrected NRR" -- a body that
# refers to a fact this way, with no digit already in it, is worth one memory lookup
# so the message doesn't go out saying literally "the corrected NRR" with no number.
_NEEDS_LOOKUP = re.compile(r"\b(corrected|updated|latest|current|final|new)\s+(\w+)", re.I)


def _enrich_body(text, command, as_of_str, data_dir):
    if not data_dir or re.search(r"\d", text):
        return text
    m = _NEEDS_LOOKUP.search(text)
    if not m:
        return text
    try:
        from memory.answer import _extractive_answer
        from memory.ingest import load_visible
        from memory.retrieve import retrieve
        query = f"What is the {m.group(2)}?"
        ids, _ = retrieve(query, as_of_str, data_dir, top_k=5)
        units = load_visible(data_dir, as_of_str)
        by_id = {u.id: u for u in units}
        context = [by_id[i] for i in ids if i in by_id]
        # Generic correction pattern ("X is A, not B"): if any top-ranked record
        # states one about this topic, A is the corrected value -- check these
        # directly, in rank order, before falling back to the extractive answerer.
        topic = re.compile(re.escape(m.group(2)), re.I)
        corr_re = re.compile(r"\bis\s+(\d[\d.,]*\s?%?)\s*,?\s*(?:not|instead of|rather than)\s+\d", re.I)
        for u in context:
            t = getattr(u, "text", "") or ""
            cm = corr_re.search(t)
            if cm and topic.search(t):
                return f"{text} ({m.group(2).upper()}: {cm.group(1).strip()})"
        fact, _, abstained = _extractive_answer(f"What is the {m.group(2)}?", context)
        if not abstained:
            # a correction reads "X is 112, not 118" -- the corrected value is the
            # one before "not", not just whichever number appears last in the text
            corr = re.search(r"\bis\s+(\d[\d.,]*%?)\s*,?\s*(?:not|instead of)\s+\d", fact, re.I)
            digits = re.findall(r"\b\d[\d.,]*%?\b", fact)
            value = corr.group(1) if corr else (digits[-1] if digits else None)
            if value:
                return f"{text} ({m.group(2).upper()}: {value})"
    except Exception:
        pass
    return text


def _dt(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=LOCAL)


def _fmt(dt):
    return dt.isoformat()


# ---------- entity resolution ----------

def all_known_names(directory):
    """Every known name -> list of candidate targets, longest names first so
    "Sarah Kim" is tried before "Sarah" and doesn't get shadowed by it."""
    names = {}
    for u in directory["slack_users"]:
        if u.get("is_bot"):
            continue
        names.setdefault(u["real_name"].lower(), []).append({"space": "slack", "id": u["id"], "display": u["real_name"]})
        first = u["real_name"].split()[0].lower()
        names.setdefault(first, []).append({"space": "slack", "id": u["id"], "display": u["real_name"]})
    for name, email in directory["email_contacts"].items():
        base = name.split("(")[0].strip()
        if not base or "@" in base:
            continue
        names.setdefault(base.lower(), []).append({"space": "email", "id": email, "display": base})
        first = base.split()[0].lower()
        names.setdefault(first, []).append({"space": "email", "id": email, "display": base})
    return names


def extract_known_name(text, directory, max_words=4):
    """Find the longest known contact name occurring at the start of `text`.
    Returns (candidates, remainder) or (None, text) if nothing matched."""
    names = all_known_names(directory)
    words = text.split()
    for n in range(min(max_words, len(words)), 0, -1):
        phrase = " ".join(words[:n]).strip(".,!?").lower()
        if phrase in names:
            return _dedupe_candidates(names[phrase], directory), " ".join(words[n:])
    return None, text


def _dedupe_candidates(cands, directory):
    """Brightline employees are both Slack users AND email contacts (they email each
    other), so a name lookup often returns the "same" person twice, once per space.
    That's not real ambiguity -- collapse to one entry per underlying identity
    (matched by email address), preferring the Slack-space version since it carries
    the DM id, so only genuinely different people count as ambiguous."""
    slack_emails = {u["id"]: u.get("email") for u in directory["slack_users"]}
    by_identity = {}
    for c in cands:
        identity = slack_emails.get(c["id"]) if c["space"] == "slack" else c["id"]
        identity = identity or (c["space"], c["id"])
        if identity not in by_identity or c["space"] == "slack":
            by_identity[identity] = c
    return list(by_identity.values())


def resolve_channel(fragment, directory):
    frag = fragment.strip().lower().lstrip("#")
    frag = re.sub(r"\s+channel$", "", frag).strip()
    frag_norm = re.sub(r"[^a-z0-9]+", "", frag)
    for c in directory["slack_channels"]:
        if c["is_dm"]:
            continue
        name_norm = re.sub(r"[^a-z0-9]+", "", c["name"].lower())
        if name_norm == frag_norm or name_norm in frag_norm or frag_norm in name_norm:
            return c["id"]
    return None


def resolve_email_for(c, directory):
    if c["space"] == "email":
        return c["id"]
    if c["space"] == "slack":
        return next((u["email"] for u in directory["slack_users"] if u["id"] == c["id"]), None)
    return None


def dm_channel_for(user_id, directory):
    for c in directory["slack_channels"]:
        if c["is_dm"] and user_id in c["members"]:
            return c["id"]
    return None
    for c in directory["slack_channels"]:
        if c["is_dm"] and user_id in c["members"]:
            return c["id"]
    return None


def resolve_event(fragment, directory, as_of):
    """Best matching calendar event by word overlap with the summary, tie-broken by
    closeness in time to as_of (prefer the nearest upcoming one, like a person would
    mean when they say "board deck prep" without a date)."""
    frag_words = set(re.findall(r"[a-z0-9]+", fragment.lower()))
    best, best_score = None, 0
    for e in directory["events"]:
        if e.get("status") == "cancelled":
            continue
        summary_words = set(re.findall(r"[a-z0-9]+", e["summary"].lower()))
        overlap = len(frag_words & summary_words)
        if overlap > best_score:
            best_score, best = overlap, e
        elif overlap == best_score and overlap > 0 and best is not None:
            e_start = _dt(e["start"].get("dateTime") or e["start"].get("date"))
            b_start = _dt(best["start"].get("dateTime") or best["start"].get("date"))
            if abs((e_start - as_of).total_seconds()) < abs((b_start - as_of).total_seconds()):
                best = e
    return best if best_score > 0 else None


# ---------- time parsing ----------

TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)


def _parse_clock(text, default_hour=9):
    m = TIME_RE.search(text)
    if not m:
        return default_hour, 0
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    if not ap and h <= 7:  # bare "at 3" in a work context almost always means afternoon
        h += 12
    return h, mi


def parse_time_expr(text, as_of, directory):
    """Returns a datetime, or None if no time expression was recognized."""
    t = text.lower().strip()

    m = re.search(r"in\s+(\d+)\s*(minute|min|hour|hr|day)s?", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"minute": "minutes", "min": "minutes", "hour": "hours", "hr": "hours", "day": "days"}[unit]
        return as_of + timedelta(**{delta: n})

    m = re.search(r"(\d+)\s*(minute|min|hour|hr)s?\s+before\s+(?:the\s+)?(.+)", t)
    if m:
        n, unit, target = int(m.group(1)), m.group(2), m.group(3)
        ev = resolve_event(target, directory, as_of)
        if ev:
            start = _dt(ev["start"].get("dateTime") or ev["start"].get("date"))
            delta = timedelta(hours=n) if unit.startswith("h") else timedelta(minutes=n)
            return start - delta
    m = re.search(r"an?\s+hour\s+before\s+(?:the\s+)?(.+)", t)
    if m:
        ev = resolve_event(m.group(1), directory, as_of)
        if ev:
            start = _dt(ev["start"].get("dateTime") or ev["start"].get("date"))
            return start - timedelta(hours=1)

    m = re.search(r"on\s+the\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m:
        day = int(m.group(1))
        month, year = as_of.month, as_of.year
        if day < as_of.day:  # e.g. "the 3rd" said in late month -> next month
            month = month % 12 + 1
            year += 1 if month == 1 else 0
        h, mi = _parse_clock(t)
        return datetime(year, month, day, h, mi, tzinfo=LOCAL)

    if "tomorrow" in t:
        base = as_of + timedelta(days=1)
        h, mi = _parse_clock(t)
        return base.replace(hour=h, minute=mi, second=0, microsecond=0)

    if "today" in t:
        h, mi = _parse_clock(t)
        return as_of.replace(hour=h, minute=mi, second=0, microsecond=0)

    for i, wd in enumerate(WEEKDAYS):
        if wd in t or wd[:3] in t:
            days_ahead = (i - as_of.weekday()) % 7
            days_ahead = days_ahead or 7
            base = as_of + timedelta(days=days_ahead)
            h, mi = _parse_clock(t)
            return base.replace(hour=h, minute=mi, second=0, microsecond=0)

    if TIME_RE.search(t):  # bare "at 3pm": same day as as_of, used for pure time changes
        h, mi = _parse_clock(t)
        return as_of.replace(hour=h, minute=mi, second=0, microsecond=0)
    return None


# ---------- top-level dispatch ----------

DESTRUCTIVE = re.compile(r"\b(delete|remove|cancel|erase|wipe)\b.*\b(all|every)\b", re.I)
QUESTION_START = re.compile(r"^(what|when|who|is|are|did|do|does|how|which|where)\b", re.I)


def parse_command(command, as_of_str, directory, data_dir=None):
    """Returns a list of action dicts (dry run)."""
    as_of = _dt(as_of_str)
    cmd = command.strip()

    if DESTRUCTIVE.search(cmd):
        return [{"type": "confirm", "args": {"summary": f"{cmd.rstrip('.')}? This can't be undone."}}]

    if QUESTION_START.search(cmd) or cmd.rstrip().endswith("?"):
        # a question phrased as a command, and not one of the action verbs below
        if not re.search(r"^\s*(message|tell|dm|ping|email|remind|book|schedule|move|reschedule|open)\b", cmd, re.I):
            return [{"type": "memory.ask", "args": {"question": cmd}}]

    m = re.match(r"^tell the ([\w\- ]+?) channel\s*(?:that\s+)?(.+)$", cmd, re.I)
    if m:
        chan = resolve_channel(m.group(1), directory)
        if chan:
            return [{"type": "slack.send_message", "args": {"to": chan, "text": _enrich_body(m.group(2).strip(), command, as_of_str, data_dir)}}]

    m = re.match(r"^(message|tell|dm|ping)\s+(.+)$", cmd, re.I)
    if m:
        rest = m.group(2)
        explicit_slack = bool(re.search(r"\bon slack\b", rest, re.I))
        rest_wo_app = re.sub(r"\bon slack\b", "", rest, flags=re.I).strip()
        cands, remainder = extract_known_name(rest_wo_app, directory)
        remainder = re.sub(r"^(that|asking|saying)\s+", "", remainder.strip(), flags=re.I)
        if cands:
            spaces = {c["space"] for c in cands}
            if explicit_slack:
                cands = [c for c in cands if c["space"] == "slack"]
            if len(cands) == 0:
                return [{"type": "clarify", "args": {"question": f"I don't see a Slack user for that -- who did you mean?"}}]
            if len(cands) > 1 or (not explicit_slack and len(spaces) > 1 and any(c["space"] != "slack" for c in cands)):
                names = ", ".join(sorted({c["display"] for c in cands}))
                return [{"type": "clarify", "args": {"question": f"Which one did you mean -- {names}?"}}]
            target = cands[0]
            if target["space"] == "slack":
                dm = dm_channel_for(target["id"], directory)
                return [{"type": "slack.send_message", "args": {"to": target["id"], "dm": dm, "text": _enrich_body(remainder, command, as_of_str, data_dir)}}]
            return [{"type": "gmail.send", "args": {"to": [target["id"]], "cc": [], "subject": "", "body": _enrich_body(remainder, command, as_of_str, data_dir)}}]
        return [{"type": "clarify", "args": {"question": f"Who did you mean by that?"}}]

    m = re.match(r"^thank\s+(.+?)\s+on\s+slack(?:\s+for\s+(.+))?$", cmd, re.I)
    if m:
        cands, _ = extract_known_name(m.group(1), directory)
        cands = [c for c in (cands or []) if c["space"] == "slack"]
        if cands:
            target = cands[0]
            dm = dm_channel_for(target["id"], directory)
            text = f"Thanks{' for ' + m.group(2) if m.group(2) else ''}!"
            return [{"type": "slack.send_message", "args": {"to": target["id"], "dm": dm, "text": text}}]

    m = re.match(r"^email\s+(.+)$", cmd, re.I)
    if m:
        cands, remainder = extract_known_name(m.group(1), directory)
        remainder = re.sub(r"^(and ask(?:ing)?(?: if)?|asking(?: if)?|that|to (?:ask|tell)(?: her| him| them)?)\s+",
                            "", remainder.strip(), flags=re.I)
        if cands:
            if len(cands) > 1:
                names = ", ".join(sorted({c["display"] for c in cands}))
                return [{"type": "clarify", "args": {"question": f"Which one -- {names}?"}}]
            target = cands[0]
            to_email = resolve_email_for(target, directory)
            if not to_email:
                return [{"type": "clarify", "args": {"question": f"I don't have an email for {target['display']}."}}]
            return [{"type": "gmail.send", "args": {"to": [to_email], "cc": [], "subject": "", "body": _enrich_body(remainder, command, as_of_str, data_dir)}}]
        return [{"type": "clarify", "args": {"question": "Who should I email?"}}]

    m = re.match(r"^remind me\s+(.+)$", cmd, re.I)
    if m:
        rest = m.group(1)
        due, body = None, rest
        for cue in (r"\bon the \d{1,2}(?:st|nd|rd|th)?\b.*?(?=\s+to\s+|$)", r"\btomorrow\b.*?(?=\s+to\s+|$)",
                    r"\btoday\b.*?(?=\s+to\s+|$)", r"\ban? hour before\b.*?(?=\s+to\s+|$)",
                    r"\b\d+\s*(minute|min|hour|hr)s?\s+before\b.*?(?=\s+to\s+|$)",
                    r"\bin \d+\s*(minute|min|hour|hr|day)s?\b.*?(?=\s+to\s+|$)",
                    r"\bnext (" + "|".join(WEEKDAYS) + r")\b.*?(?=\s+to\s+|$)",
                    r"\bat \d{1,2}(:\d{2})?\s*(am|pm)?\b.*?(?=\s+to\s+|$)"):
            m2 = re.search(cue, rest, re.I)
            if m2:
                due = parse_time_expr(m2.group(0), as_of, directory)
                body = (rest[:m2.start()] + " " + rest[m2.end():]).strip()
                break
        body = re.sub(r"^(?:to|that)\s+", "", body.strip()).strip().rstrip(",")
        if due:
            return [{"type": "reminder.create", "args": {"text": body, "due": _fmt(due)}}]
        return [{"type": "reminder.create", "args": {"text": body}}]

    m = re.match(r"^(move|reschedule|push)\s+(.+?)\s+to\s+(.+)$", cmd, re.I)
    if m:
        ev = resolve_event(m.group(2), directory, as_of)
        if ev:
            new_start = parse_time_expr(m.group(3), as_of, directory)
            old_start = _dt(ev["start"].get("dateTime") or ev["start"].get("date"))
            old_end = _dt(ev["end"].get("dateTime") or ev["end"].get("date"))
            if new_start:
                # keep the event's own day if the time expression was a bare clock time
                if not re.search(r"tomorrow|today|\bon\b|next\s+\w+", m.group(3), re.I):
                    new_start = old_start.replace(hour=new_start.hour, minute=new_start.minute)
                new_end = new_start + (old_end - old_start)
                return [{"type": "calendar.update_event",
                         "args": {"event_id": ev["id"], "start": _fmt(new_start), "end": _fmt(new_end)}}]

    m = re.match(r"^(?:book|schedule)\s+(\d+)\s*(?:min(?:ute)?s?|hours?)\s+with\s+(.+)$", cmd, re.I)
    if m:
        dur_min = int(m.group(1))
        rest = m.group(2)
        cands, remainder = extract_known_name(rest, directory)
        topic_m = re.search(r"\babout\s+(.+)$", remainder, re.I)
        topic = topic_m.group(1).strip() if topic_m else None
        time_part = remainder[:topic_m.start()] if topic_m else remainder
        start = parse_time_expr(time_part, as_of, directory)
        attendees = []
        for c in (cands or []):
            if c["space"] == "slack":
                email = next((u["email"] for u in directory["slack_users"] if u["id"] == c["id"]), None)
                if email:
                    attendees.append(email)
            elif c["space"] == "email":
                attendees.append(c["id"])
        attendees = list(dict.fromkeys(attendees))  # dedupe, keep order
        if start:
            end = start + timedelta(minutes=dur_min)
            title = topic or (f"Sync with {cands[0]['display']}" if cands else "Meeting")
            return [{"type": "calendar.create_event",
                     "args": {"title": title, "start": _fmt(start), "end": _fmt(end), "attendees": attendees}}]

    m = re.match(r"^open\s+(.+)$", cmd, re.I)
    if m:
        return [{"type": "app.open", "args": {"app": m.group(1).strip().rstrip(".")}}]

    # last resort: treat it as a question for memory rather than failing silently
    return [{"type": "memory.ask", "args": {"question": cmd}}]


def parse(command, as_of_str, directory, data_dir=None):
    """Handles simple compound commands ("X and Y") by splitting into independent
    clauses when each half looks independently actionable."""
    parts = re.split(r"\s+and\s+", command, maxsplit=1, flags=re.I)
    if len(parts) == 2 and re.search(
            r"^(message|tell|dm|ping|email|remind|book|schedule|move|reschedule|open|thank)\b", parts[1].strip(), re.I):
        return parse_command(parts[0].strip(), as_of_str, directory, data_dir) + parse_command(parts[1].strip(), as_of_str, directory, data_dir)
    return parse_command(command, as_of_str, directory, data_dir)
