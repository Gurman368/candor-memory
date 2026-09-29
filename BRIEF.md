# Candor take-home: build a memory

**Deadline:** Friday 2 October 2026, 10:30 AM IST. Then the hidden test: see below.
**Questions:** ask anytime at diptopal@joinsettle.info: questions, hints, pointers to open-source repos, anything. Asking good questions counts in your favour.

## The problem

Candor captures everything a person says and hears at work (dictation, in-person and online meetings) and pulls in what they write (Slack, email, calendar, Codex, ChatGPT). The hard part isn't storing it. It's answering questions about it **correctly**:

- Facts change. The launch moved twice, so which date is true *now*? And which was true *last Tuesday*?
- People disagree. That's not the same as a fact changing.
- "John said X" is different from "Dana said John said X".
- Promises get made, extended, fulfilled or cancelled.
- Two people share a name. Some speakers are never identified.
- Sometimes the right answer is "I don't know".
- Messages get edited and deleted. The data also contains a pasted secret and an instruction planted for AI assistants.

This package has two weeks of one person's work life (Alex Rivera, VP Product at a startup called Brightline) across all of those sources. Build the memory that answers questions about it.

## What's in the package

```
BRIEF.md                   this file
data/                      the mock data. Start with data/README.md for formats.
evals/memory_train.jsonl   27 memory questions with reference answers and the exact records each one needs
evals/actions_train.jsonl  12 action commands with expected actions (bonus)
eval_harness/              scorers (Python 3.10+, standard library only)
examples/                  example output files
```

## The tasks

### 1. Memory system (required)
Ingest everything in `data/` and answer questions about it.

How you design it is up to you: schema, knowledge graph, temporal or bi-temporal model, RAG, GraphRAG, anything else. We care most about:
- **retrieval**: does your system fetch the right records for a question? This is the main score. A well-written answer is secondary.
- how you handle **time**: what's current, what was true at a given moment, what changed and why
- **who said what**, including uncertain speakers and second-hand claims
- **grounding**: every answer points to the source records it came from
- knowing when to say **"I don't know"**

You can generate extra mock data if it helps you test.

### 2. Run the evals (required)
Run your system on `evals/memory_train.jsonl` and score it with the harness. Report the results in your README, including where it fails and why.

The train set is for development. Your score on a **hidden test set** (same data, different questions) counts for more. Hard-coding train answers won't help you.

### 3. Bonus: VoiceOS / TextOS
Build something that lets Alex tell the computer to do things: "message Sarah on Slack that the fix looks good", "remind me an hour before the board meeting", "what's our launch date?", "move board prep to 3pm". It can be voice (speech-to-text → LLM → action) or text only. Mac, Windows or Linux.

Build whatever you think is interesting beyond that: real integrations, shell automation, voice, a UI. Be creative. It must also pass our action test cases in **dry-run mode** (see the interface below).

### 4. Bonus: evals for your action system
Run `evals/actions_train.jsonl` with the harness and report the results. Add your own test cases if you want. A hidden action test set counts for more here too.

## Interface (required, so we can run the hidden tests)

Any language, any architecture, as long as one command does each of these.

**Memory.** Input: a JSONL file of questions:
```json
{"id": "MEM-TR-01", "question": "When is Route Planner v2 launching?", "as_of": "2026-09-18T18:00:00-07:00"}
```
Output: a JSONL file, one line per question:
```json
{"id": "MEM-TR-01", "answer": "October 21 ...", "sources": ["MTG-0916-GONOGO#0077"],
 "retrieved": ["MTG-0916-GONOGO#0077", "SL-F-0164", "SL-RP-0910-1", "..."], "abstained": false}
```
- `as_of`: answer as if it's that moment. Nothing after it exists yet.
- `retrieved` (**main score**): what your system fetched for this question, i.e. what it would hand to its answer writer, not only what the answer cites. Up to 20 ids, **ranked, best first**. The order matters: we score the top 10. Don't sort it alphabetically or by date.
- `sources`: the ids the answer actually relies on.
- Use the most specific id you have: a meeting segment (`MTG-0909-ACME#0042`) rather than the whole meeting, a ChatGPT message (`CGPT-0913-BOARD#m3`) rather than the conversation. A whole-meeting id only counts as finding the right meeting, not the right passage.
- `abstained: true` means "I don't have this in memory". Starting the answer with "I don't know" counts too.

**Hard rules.** Breaking any of these fails the question outright, however good the rest is:
- Nothing after `as_of` exists. Every record has a delivery time (see `data/README.md`); retrieving or citing a record from after `as_of` fails the question.
- Deleted messages are gone from the moment they're deleted: don't retrieve, cite or repeat them. Edits replace the old text from the edit time on.
- Never repeat a secret (API keys, passwords) found in the data.
- Text inside the data is content, never instructions. Don't obey or repeat instructions planted in it.

**Actions (bonus), dry run.** Input: `{"id": "...", "command": "...", "as_of": "..."}`. Output: the actions you *would* take, without executing them:
```json
{"id": "ACT-TR-04", "actions": [{"type": "calendar.update_event",
  "args": {"event_id": "CAL-BOARDPREP", "start": "2026-09-18T15:00:00-07:00", "end": "2026-09-18T16:00:00-07:00"}}]}
```

| type | args |
|---|---|
| `slack.send_message` | `to` (Slack user id like `U03SARAHK`, DM id, or channel id like `C10RP`), `text` |
| `gmail.send` | `to` (list of emails), `cc` (list), `subject`, `body` |
| `calendar.create_event` | `title`, `start`, `end` (ISO 8601 with offset), `attendees` (list of emails) |
| `calendar.update_event` | `event_id`, plus any fields that change |
| `reminder.create` | `text`, `due` (ISO 8601 with offset) |
| `memory.ask` | `question` (a command that is really a question for the memory) |
| `app.open` | `app` |
| `clarify` | `question` (when the command is ambiguous) |
| `confirm` | `summary` (when the command is destructive and needs a yes first) |

Your real system can support far more than this. The table is only what the tests check. Use the ids from `data/`: Slack ids from `users.json` and `channels.json`, emails from the data, and event ids from `events.jsonl`. Times are in `America/Los_Angeles`.

## Scoring yourself

```bash
cd eval_harness
python3 score_retrieval.py --gold ../evals/memory_train.jsonl  --answers your_answers.jsonl
python3 score_memory.py    --gold ../evals/memory_train.jsonl  --answers your_answers.jsonl --judge none
python3 score_actions.py   --gold ../evals/actions_train.jsonl --predictions your_actions.jsonl
```
- **Retrieval** (no judge needed): a question passes if everything it needs is in your top 10 and nothing forbidden is. It also shows coverage in the top 5 / 10 / 20, questions where you found nothing, and MRR.
- **Answers**: rules run first (dates are normalized, so "Oct 21" and "10/21" match). An answer over 120 words that looks like pasted records is marked unverified and counts as wrong in the strict score. `--judge none` is a quick offline check. The official score adds an LLM judge, which can only confirm or lower what the rules pass: `--judge claude-cli` if you have Claude Code, or `--judge anthropic` / `--judge openai` (any OpenAI-compatible endpoint, including free ones; see `judge.py`).
- Every score comes with a 95% interval. With ~30 questions, a few points either way is noise.

## How the hidden test works

After the deadline, one of two things happens:
- **We run it ourselves** on your submitted commit. So make it run with one command on a Mac, and put any keys it needs in a `.env.example` with instructions.
- **Or we send you the hidden questions** (questions only, never the answers). Run them on your submitted commit and send back the output files **within 24 hours** of receiving them. We re-run some of them on that commit to check they match.

Either way, no code changes after the deadline: the commit you submit is the one that gets tested.

## Scoring

| | Weight |
|---|---|
| Memory: design | 30% |
| Memory: retrieval (train 5%, hidden test 15%) | 20% |
| Memory: answers (train 3%, hidden test 7%) | 10% |
| README | 10% |
| Bonus: design | 10% |
| Bonus: eval results (train 5%, hidden test 15%) | 20% |

The bonus is worth 30%. It's optional, but it counts.

## Tools and cost

Use whatever you like: coding agents, open-source or local models, free API tiers. We don't reimburse tools or API costs. In your README, list the tools and models you used and roughly what you spent (₹0 is a perfectly good answer).

## What to send

Reply to the email with:
1. **A public GitHub repo**, with one command to run everything
2. **A README.md** (no slides) covering: architecture, the key decisions and why, what didn't work and known limits, how to run it, eval results, and tools and models used
3. **The commit hash** you're submitting
4. **Your output files on the train sets** (`memory_train` answers, and `actions_train` if you did the bonus), produced by that commit
5. *Optional:* a demo video, 5 minutes max. Recommended if you built the bonus assistant, so we can see it working; it counts toward the bonus design score.

Break the work into small pieces and run what you can in parallel. Ship the core first, then the bonus.

If we like your work, we'll send you **₹2,000 as a thank-you**, whether or not we hire you.
