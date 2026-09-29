# Mock data

Two weeks (Tue 8 Sep – Fri 18 Sep 2026, plus some future calendar events) of the working life of **Alex Rivera, VP Product at Brightline**, a startup selling route-planning software to freight companies. Alex is the Candor user. Everything is fictional; all domains are reserved `*.example.com` domains.

Timezone: `America/Los_Angeles`. All timestamps are ISO 8601 with offset.

Every record has a stable `id`. Your system returns these ids in `retrieved` and `sources` (see `../BRIEF.md`). Meeting segments have their own ids (`MTG-0909-ACME#0042`); ChatGPT messages too (`CGPT-0913-BOARD#m3`).

**Delivery time.** Each record exists from its timestamp on: a meeting segment from meeting `start` + `end_s`, a dictation, Slack message or email from its timestamp, a calendar event from its `updated` time (the file holds the event's current state), a Codex session from its last event, a ChatGPT message from its `create_time`. A question asked `as_of` a moment can only use records that existed then.

```
data/
  native/                      captured by Candor on Alex's Mac
    meetings/<id>.json         one file per meeting (in-person, hybrid, remote)
    dictation/dictations.jsonl one line per dictation
  connectors/                  pulled from Alex's accounts
    slack/users.json
    slack/channels.json
    slack/messages.jsonl       all channels and DMs, one message per line
    gmail/messages.jsonl
    google_calendar/events.jsonl
    codex/sessions/<id>.jsonl  one Codex CLI session per file
    chatgpt/conversations.json
```

## native/meetings/<id>.json

Diarized transcript from Candor's meeting capture. Speaker labels are what the diarizer produced; `speaker_name` is filled only when Candor identified the person, with a confidence. ASR errors, crosstalk and filler words are real features, not bugs.

```json
{
  "id": "MTG-0909-ACME",
  "title": "Acme Freight – pricing and rollout",
  "type": "hybrid",                         // in_person | hybrid | remote
  "start": "2026-09-09T11:00:00-07:00",
  "end": "2026-09-09T11:50:00-07:00",
  "location": "Brightline HQ – Embarcadero room + Google Meet",
  "capture": {"device": "MacBook Pro (Alex)", "mic": "built-in", "system_audio": true},
  "calendar_event_id": "CAL-ACME-0909",      // null if not on the calendar
  "participants_known": ["alex@brightline.example.com", "..."],
  "segments": [
    {
      "seg_id": "MTG-0909-ACME#0001",
      "start_s": 0.0, "end_s": 6.4,
      "speaker_label": "Speaker 1",
      "speaker_name": "Alex Rivera",          // null if unidentified
      "speaker_confidence": 0.93,
      "channel": "room",                       // room (mic) | remote (system audio)
      "text": "Okay, I think everyone's in. Thanks for making the time."
    }
  ]
}
```

## native/dictation/dictations.jsonl

Voice dictation typed into another app, or kept as a note to self. `raw_transcript` is the ASR output; `cleaned_text` is what was inserted.

```json
{"id": "DCT-0910-02", "timestamp": "2026-09-10T16:03:12-07:00", "mode": "dictation",
 "target_app": "Gmail", "target_context": "Compose – to sarah.patel@acmefreight.example.com",
 "raw_transcript": "...", "cleaned_text": "...", "delivery_state": "sent"}
```
`mode`: `dictation` | `note_to_self`. `delivery_state`: `inserted` | `sent` | `edited_after_insert` | `discarded` | `saved_note`.

## connectors/slack

- `users.json`: `[{"id": "U01ALEX", "name": "alex", "real_name": "Alex Rivera", "title": "...", "email": "..."}]`
- `channels.json`: `[{"id": "C10RP", "name": "route-planner", "is_dm": false, "members": [...]}]` (DMs have `is_dm: true`)
- `messages.jsonl`: `{"id": "SL-...", "channel_id": "C10RP", "user": "U03SARAHK", "ts": "2026-09-10T09:42:05-07:00", "text": "...", "thread_parent_id": null, "reactions": [{"name": "+1", "users": ["U01ALEX"]}], "subtype": null}`. Bot messages have `subtype: "bot_message"` and a `bot_name`.
- Edits and deletions arrive later as their own records (ids `SL-EV-...`), in time order like everything else:
  - `{"id": "SL-EV-...", "subtype": "message_changed", "target_id": "SL-...", "ts": "...", "text": "<new text>"}`: from `ts` on, the target message reads as the new text.
  - `{"id": "SL-EV-...", "subtype": "message_deleted", "target_id": "SL-...", "ts": "..."}`: from `ts` on, the target message is gone. Don't retrieve, cite or repeat it after that.

## connectors/gmail/messages.jsonl

```json
{"id": "EM-...", "thread_id": "TH-...", "date": "...", "from": "Name <email>", "to": ["..."], "cc": [], "subject": "...",
 "body": "plain text", "labels": ["INBOX"], "attachments": [{"filename": "...", "mime_type": "..."}]}
```

## connectors/google_calendar/events.jsonl

Current state of each event (like the Calendar API). Changes show up in `updated` and in Gmail invitation emails.

```json
{"id": "CAL-...", "summary": "...", "description": "...", "location": "...",
 "start": {"dateTime": "..."}, "end": {"dateTime": "..."},        // or {"date": "YYYY-MM-DD"} for all-day
 "organizer": "email", "attendees": [{"email": "...", "responseStatus": "accepted|declined|tentative|needsAction"}],
 "status": "confirmed|cancelled", "recurrence": null, "created": "...", "updated": "..."}
```

## connectors/codex/sessions/<id>.jsonl

One Codex CLI session. First line is session metadata, then one event per line.

```json
{"type": "session_meta", "id": "CDX-0912", "started_at": "...", "cwd": "~/code/eta-predictor", "repo": "eta-predictor"}
{"type": "message", "role": "user", "timestamp": "...", "content": "..."}
{"type": "message", "role": "assistant", "timestamp": "...", "content": "..."}
{"type": "tool_call", "timestamp": "...", "tool": "shell", "input": "...", "output": "..."}
```

## connectors/chatgpt/conversations.json

Simplified ChatGPT export.

```json
[{"id": "CGPT-...", "title": "...", "create_time": "...", "update_time": "...",
  "messages": [{"id": "CGPT-...#m1", "role": "user|assistant", "create_time": "...", "content": "..."}]}]
```
