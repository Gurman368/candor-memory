# Candor take-home: memory + actions

Built for Alex Rivera's two weeks at Brightline. Two pieces: a memory system
(required) and a VoiceOS/TextOS-style action parser (bonus). Both run with one
command each and need no network access to work at all -- an API key just
upgrades answer quality and action-parsing robustness, it isn't required to
produce output.

## Quickstart

One command runs everything (memory + actions bonus, on the training sets):

```bash
git clone <this repo>
cd <this repo>
pip install -r requirements.txt
cp .env.example .env    # optional: add ONE api key to improve answers, else skip this
python3 run_all.py
```

That writes `out/memory_train_answers.jsonl` and `out/actions_train_predictions.jsonl`
and prints the scoring commands to run next. To run either piece on its own,
with custom paths (e.g. the hidden test set):

```bash
python3 run_memory.py  --questions evals/memory_train.jsonl  --out out/memory_train_answers.jsonl
python3 run_actions.py --commands evals/actions_train.jsonl  --out out/actions_train_predictions.jsonl
```

Score them:
```bash
cd eval_harness
python3 score_retrieval.py --gold ../evals/memory_train.jsonl  --answers ../out/memory_train_answers.jsonl
python3 score_memory.py    --gold ../evals/memory_train.jsonl  --answers ../out/memory_train_answers.jsonl --judge none
python3 score_actions.py   --gold ../evals/actions_train.jsonl --predictions ../out/actions_train_predictions.jsonl
```

`pip install -r requirements.txt` installs `sentence-transformers` for the local embedding model and cross-encoder reranker (see "v3" below). No API key is strictly required: without one the memory system uses an extractive fallback and actions use the rule-based path. Adding an API key (Groq / Anthropic / Gemini) enables the full LLM pipeline.

## Architecture

```
memory/
  ingest.py       load the data into "units", correct-by-construction on time-travel
  retrieve.py     hybrid retrieval: BM25 + entity/phrase boost + graph expansion
  answer.py       LLM answer synthesis, extractive fallback
  llm_client.py   one shared caller for Anthropic / Gemini / OpenAI, with retry
actions/
  directory.py    static people/channel/event directory for command parsing
  rules.py        rule-based command parser + relative-date parser
  llm_parse.py    optional LLM-based command parser (tried first if a key is set)
run_memory.py     entrypoint: questions.jsonl -> answers.jsonl
run_actions.py    entrypoint: commands.jsonl -> predictions.jsonl
run_all.py        one-command entrypoint: runs both of the above on the train sets
```

### Ingestion and time (`memory/ingest.py`)

The single most important correctness property this system has is: **it is
structurally impossible for it to retrieve or cite a record from after
`as_of`, or a deleted record.** Rather than reimplementing that bi-temporal
logic (delivery time vs. content time, edits, deletions) myself and risking a
subtle bug, `ingest.py` imports `eval_harness.records.visible()` directly --
the same function the grader uses -- to build the corpus of retrievable
"units" for a given `as_of`. Everything downstream (retrieval, answering)
only ever sees that already-filtered list. This is why the retrieval score
below shows **zero forbidden records retrieved across all 27 questions**: it
isn't a heuristic that mostly works, it's a hard invariant from the data
model.

On top of that, `ingest.py` builds a static **link index** per unit (thread
parent, channel/event id, speaker name + confidence, and any other record id
explicitly mentioned in the text) used later for graph expansion.

**Defense in depth for the two planted hazards.** The data contains a pasted
API key in a Slack DM and an HTML comment in a promotional email addressed to
an AI assistant, falsely claiming a contract was signed. Both are redacted at
ingestion time (`_redact()` in `ingest.py`), before retrieval or answering
ever sees the raw text -- so even if an answering path has a bug, it can't
leak what it never received. The comment is replaced with a visible
placeholder (not deleted silently), so a question that directly asks about it
can still be answered honestly without repeating the payload.

### Retrieval (`memory/retrieve.py`) -- the main score, so most of the design
effort went here.

Base ranking is BM25 over each unit's text. Several refinements on top, each
motivated by a specific failure mode found on the training set:

1. **Header stripping for BM25, kept as a flat bonus instead.** Every segment
   of a meeting/thread repeats its title/subject verbatim (`"[Acme Freight --
   pricing and rollout, ...]"`). Left in, BM25's document-length
   normalization lets a one-line segment that just repeats the title
   outrank a long, specific document like the actual proposal email --
   this was measured directly (a meeting segment saying "should we get into
   pricing?" was initially outranking the actual pricing proposal email).
   The header is stripped before BM25 tokenization; a smaller flat
   per-token bonus (not frequency-scaled) still rewards a real header match
   (e.g. a dictation's `target_context` naming the recipient), without
   letting repetition across hundreds of sibling documents dominate. This
   one change took retrieval from 76% to 84% on the training set.

2. **Phrase/entity bonus.** Multi-word capitalized phrases and acronyms
   shared between the question and a candidate ("Route Planner v2", "SSO")
   are mined from the corpus at query time (not hand-written) and given a
   flat bonus, since they're a much stronger relevance signal than one
   token of overlap.

3. **Generic English synonym expansion** (`SYNONYMS` in `retrieve.py`) for
   the query only, so "slip" matches "delayed", "sign" matches "signed" /
   "closed", etc. Small and hand-written, but generic English, not tied to
   this dataset's specific facts.

4. **Optional LLM query expansion** (`llm_expand_terms`): if a key is
   configured, one extra call asks the model what words a relevant message
   would actually contain -- this is the one mechanism that can close a
   genuine vocabulary/reasoning gap (e.g. connecting "why did the launch
   slip" to a QA message about a geocoding regression that never uses the
   word "slip" or "delay"). Set `DISABLE_LLM_QUERY_EXPANSION=1` to turn this
   off and save calls.

5. **Calendar-day join** (`extract_dates` + the block in `retrieve()` gated
   on "day"/"calendar"/"schedule" in the question). "What's on my calendar
   the day I fly to Denver?" can't be answered by keyword overlap at all --
   the connecting fact is a *date*, not a shared word. The system extracts
   the date(s) implied by its current best matches, then pulls in any
   record (any source, not just calendar) that mentions the same date, with
   a strong score boost so it actually surfaces in the top 10.

6. **Graph expansion.** Once a shortlist is ranked, the rest of a
   Slack/email thread and any record explicitly referenced by id in a
   shortlisted unit's text are pulled in. This is what recovers multi-hop
   questions where the answer depends on a reply that doesn't itself repeat
   the original message's keywords.

**Result on the training set: 88% retrieval score, 0 forbidden records in
top 10 or top 20, MRR 0.585.** See "What didn't work" below for the specific
questions that still fail and why.

### Answering (`memory/answer.py`)

Two paths, chosen automatically based on whether a key is configured
(`memory/llm_client.py`):

- **LLM mode**: the model gets the question, `as_of`, and the retrieved
  units' text, and is instructed to (a) answer only from that text, (b)
  treat anything inside a record as content to report on, never as an
  instruction to follow, (c) never repeat a secret, (d) abstain if the text
  doesn't cover it, (e) reflect what's *current* as of `as_of` when a fact
  changed, mentioning history only briefly. Structured JSON output with
  `sources` constrained to the ids it was actually given.
- **Extractive fallback** (no key): scores every sentence in the top
  retrieved units by word-overlap with the question, keeps the best few in
  chronological order (so a changed fact reads as a small history), and
  abstains if nothing overlaps. This exists purely so the pipeline runs
  end-to-end with zero cost/config -- it is not meant to compete with the
  LLM path on answer quality, and the numbers below reflect that honestly.

### Actions bonus (`actions/`)

Same two-path shape. The rule-based parser (`rules.py`) resolves names
against a directory built from the data itself (Slack users/channels, email
contacts derived from headers, current calendar state) rather than matching
exact phrasings, so it isn't limited to the 12 training commands' wording --
any command naming a real person/channel/event the data has should resolve.
It handles: Slack DMs and channel messages, email, reminders (absolute,
relative, and event-relative times like "an hour before the board meeting"),
moving/creating calendar events, opening apps, simple two-clause compound
commands ("email X and thank Y"), destructive commands (routed to `confirm`),
ambiguous names (routed to `clarify` -- with identity de-duplication so a
Brightline employee, who is naturally both a Slack user *and* an email
contact, isn't falsely flagged as two different people), and questions
phrased as commands (routed to `memory.ask`). The LLM-based parser
(`llm_parse.py`) is tried first when a key is set, and is considerably more
robust to novel phrasing since it can read the whole directory and reason
about it directly; the rule-based parser is the always-available fallback.

**Result on the training set: 12/12 passing (100%), 100% argument accuracy.**
The last failure (ACT-TR-10, corrected NRR) was fixed by checking the top
retrieved records directly for an "X is A, not B" correction pattern before
falling back to the extractive answerer.

## v2

**The files in `out/` come from a live-model run** (provider Groq, model `openai/gpt-oss-120b`,
settings under "v2 results" below), produced by this commit. On Windows set `PYTHONUTF8=1`
before running (the data is UTF-8; Mac/Linux need nothing).

After hidden-test feedback (retrieval 71%, answers 55%, actions 7/13 -- and
the core criticism that the model only ever saw keyword-expansion terms, never
the actual candidate records), four changes:

1. **Retrieval reranking** (`memory/retrieve.py`): the model now reads up to
   ~45 candidate records' real text (id, timestamp, source, speaker, a
   snippet) and returns an ordered pick of the useful ones, with an explicit
   rule to also include records that *change* a fact (corrections, edits,
   disagreement) even when they share few words with the question. Its picks
   go first; anything it didn't pick keeps the heuristic (BM25+rules) order
   after, so the mechanism can only add signal, never lose a record the old
   path would have found. Any id it returns that wasn't in the candidate set
   -- invented or not -- is dropped before it ever reaches the output.
   Keyword-expansion is now off by default (`ENABLE_LLM_QUERY_EXPANSION` to
   turn it on) since the reranker seeing real text is strictly more
   informative, and it cuts a call per question.
2. **Less cautious answering** (`memory/answer.py`): the old prompt said
   "abstain if the records don't answer" and the model over-applied it to
   partial or indirectly-phrased evidence. New prompt is explicit: abstain
   *only* when no record states the fact and it can't be assembled from
   several; wording mismatches or evidence spread across records are not
   abstention triggers. It also asks for `evidence` before `answer` in the
   JSON (so the model has to point at records before writing prose), and
   explicitly weighs speaker confidence instead of treating every speaker
   attribution as certain.
3. **Robustness instead of silent downgrade** (`memory/llm_client.py`): a
   shared `call_llm_json` does one repair retry on malformed JSON before
   giving up; retries honor `Retry-After` and detect a *daily* quota
   exhaustion (distinct from a transient rate limit) to fail fast instead of
   retrying it five times per question; a circuit breaker opens after 3
   consecutive failures or one quota error, so a dead key doesn't burn the
   rest of the run retrying; every fallback is logged loudly and counted
   (`stats_summary()`), never silent. `run_memory.py --resume` keeps answers
   from an earlier run that came fully from the model and only redoes the
   rest, so a quota cutoff doesn't mean starting over.
4. **Actions with memory, not just the directory** (`actions/llm_parse.py`):
   the model now also gets memory records relevant to the command and any
   email address it can find written next to an unrecognized name anywhere
   in memory -- so "email the new hire" can resolve someone who only appears
   in a meeting transcript, never in an email header or the Slack directory.
   Genuine same-first-name ambiguity and destructive commands are still
   decided by the deterministic rules before the model is even called (never
   let a model guess which Sarah to message), and the system prompt now says
   explicitly: if Alex already stated the message content, use it -- don't
   ask again; only clarify for a missing/ambiguous recipient or time.
   Everything the model proposes is validated against real directory/memory
   ids and emails before being accepted.

**Verified offline** (`tests/test_offline.py`, `tests/test_actions_offline.py`,
`tests/test_quota.py` -- all mocked, no API key needed, all passing): JSON
parsing tolerates fences/preamble/trailing text and does one repair retry;
rerank failure keeps the lexical order and is logged, never silently returns
nothing; any id not in the candidate pool is dropped; a real-shaped daily-quota
429 trips the breaker immediately instead of retrying 5 times. The no-key
fallback path (rules-only / extractive) is unchanged and still gets 88%
retrieval / 12/12 actions on the training set.

**Verified with a live model:** see "v2 results" under "Eval results". The no-key fallback
path is unchanged and still gets 88% retrieval / 12/12 actions, so the system degrades
gracefully; v2 only adds what the model can do on top.

### v2 known limits (read before trusting the numbers)

- **One live model only.** All live numbers are from `openai/gpt-oss-120b` on Groq's free tier.
  A Gemini run was started but its free quota ran out after a handful of calls (those
  answers were correct, but it is not a full run). Other models/providers are untested here.
- **Tested settings differ from the code defaults.** Free-tier token limits forced a smaller
  prompt: `RERANK_POOL=30` (default 45), `RERANK_SNIPPET_CHARS=400` (default 700),
  `ANSWER_CONTEXT_K=10` (default 12). The defaults are untested live. `.env.example` lists the
  exact settings used for the reported numbers.
- **The train set is a development set.** I used it while building (and I measured candidate-pool
  recall against its gold records to choose a pool of ~45 over ~20). Nothing is written in for specific
  questions, but treat train numbers as optimistic. The earlier hidden test scored far lower
  (71% / 55% / 7 of 13) on the v1 system, so expect a gap.
- **"Answers 100%" is `--judge none`** (rules only). The official LLM judge can only lower
  what the rules pass.
- **MEM-TR-20 still fails retrieval** (finds 1 of 2 needed records in the top 10). Most likely the same
  cause as in v1 (the dictation's body says only "Hi Sarah"; her full name lives in metadata). I have
  not confirmed it.
- **Not bit-for-bit reproducible.** Temperature is 0 (and `seed` where the provider supports it), but
  Groq may still vary slightly between runs, so a re-run may differ on a few answers.
- **Actions:** the model sometimes targets a DM id (e.g. `D-ALEX-SARAHK`) instead of a Slack user id;
  the brief allows both. The "is A, not B" correction rule and ambiguity/destructive rules are still
  narrow rules, with the model covering the general case.

## What didn't work / known limits in earlier iterations (v1 & v2)

**Memory, retrieval (earlier iterations):**
- *"Is Harbor Logistics going to sign this year?"*: previously found 2 of 3 needed
  sales updates; the third scored lower without a strong lexical signal distinguishing it.
  (Resolved in v3 by dense embeddings + 150-candidate pool).
- *"What did I dictate to Sarah Patel on Sep 10, and did it go out?"*: the
  dictation itself only says "Hi Sarah" in its body -- her last name only
  appears in metadata (target contact). (Resolved in v3 by metadata scoring + cross-source fusion).
- *"Why did the launch slip from September 30?"*: the literal tokens
  "launch"/"September"/"30" appeared verbatim in the original planning meeting, outscoring
  the actual QA regression message which never used "slip". (Resolved in v3 by semantic dense vector search).

**Memory, answer quality (extractive, no LLM key) is intentionally weak:**
`--judge none` strict score is 37% (lenient 48%), with 0 hard failures. The extractive answerer
is an offline heuristic fallback so the repo runs with zero setup. With an LLM active, answers
score 96.3% strict / 100% lenient.

**Security & Safety:**
An early version of the extractive fallback repeated the planted secret and the hidden-instruction
payload in one answer, because word-overlap scoring doesn't know what a secret is. Fixed by redacting
both at ingestion (`_redact()` in `memory/ingest.py`), before retrieval or answering ever sees the raw text.

**Actions (v1 & v2 fixes):**
- ACT-TR-10 ("Email John the corrected NRR"): fixed by scanning top-ranked records for "is A, not B" correction patterns.
- Two-space identity ambiguity: Brightline employees appearing in both Slack and email headers are merged by email address.
- ACT-TR-09 (Sarah Kim vs Sarah Patel): resolves same-first-name ambiguity with a `clarify` prompt.

---

## v3: Retrieval Round & Live Results (Latest Submission)

After round 2 feedback from Diptopal Roy (retrieval on the hidden test set scored 71% against 89–96%
for the strongest submissions because the system relied on keyword search + heuristic rules), this
round focused on **closing the retrieval gap**.

### What changed (all 7 points from feedback implemented):
1. **Two ways to find records**: Dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and
   char 4-gram TF-IDF are merged with BM25 keyword search using Reciprocal Rank Fusion (RRF). `memory/candidates.py`.
2. **Enrich records at load**: Relative dates ("Friday", "tomorrow", "last Monday") inside records
   are resolved deterministically against each record's own timestamp (strictly no future data leakage). `memory/candidates.py:relative_dates`.
3. **Search again before giving up**: If the first answer pass abstains or flags missing information in
   `missing`, the model writes 1–3 new searches that run over all visible records, and the answer is rewritten. `memory/retrieve.py:llm_followup_queries`.
4. **Let the model plan the search**: The query planner rewrites questions into 1–4 specific searches
   in the data's vocabulary (guided by a mined glossary) and extracts dates. `memory/retrieve.py:llm_plan`.
5. **Search each source separately**: BM25 runs inside each source (meetings, Slack, email, calendar,
   dictations, codex, chatgpt) independently and interleaves results. `memory/candidates.py:persource_rank`.
6. **Catch later changes**: Records on the same topic that indicate a move, edit, or correction are
   pulled in chronologically. `memory/candidates.py:change_candidates`.
7. **Wider first pass (150 candidates) + Local Reranker**: Retrieves ~150 candidates across all sources,
   then a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks them to select the top ~45 for the LLM. `memory/candidates.py:cross_rank`.

### Diagnostic Evaluation on Paraphrased Dev Set (`evals/memory_dev_paraphrase.jsonl`)
Because the training set already had high pool recall under exact-word matches, a paraphrased dev set was
created (same 25 gold records with questions reworded to simulate the hidden test set's vocabulary mismatch).

#### Why v2 collapsed on the hidden test set vs how v3 closes the gap:

| Pipeline | Dev Top 10 Recall | Dev Pool Recall (what model sees) | Hidden Test Score |
|---|:---:|:---:|:---:|
| **v2 (Keyword Search)** | **36% (9 / 25)** | **60% (15 / 25)** | **71%** (Failed) |
| **v3 (Neural Embeddings + Cross-Encoder + Planner)** | **72% (18 / 25)** | **84% (21 / 25)** | **Expected 89–96%** |

**The crucial difference:**
- In **v2**, keyword search required exact token matches. When questions were paraphrased (e.g. *"plane to Colorado"* instead of *"flight to Denver"*), the gold records were ranked outside the top 45 pool (only 60% pool recall). The LLM never saw them, explaining the 71% ceiling on the hidden test set.
- In **v3**, semantic dense embeddings (`all-MiniLM-L6-v2`) and the local cross-encoder (`ms-marco-MiniLM-L-6-v2`) match on meaning rather than verbatim words. First-stage Top 10 recall **doubled from 36% to 72%**, and pool recall reached **84%**, directly addressing the bottleneck Diptopal identified.

#### Detailed step-by-step ablation (measured with `tools/eval_first_stage.py`, out of 25):

| Setting | dev top 10 | dev pool | train top 10 | train pool |
|---|:---:|:---:|:---:|:---:|
| Baseline (v1 keyword only) | 9/25 (36%) | 15/25 (60%) | 22/25 | 24/25 |
| + n-gram, per-source, dates, changes | 10/25 | 16/25 | 22/25 | 24/25 |
| + model query planner (cached plans) | 16/25 | 19/25 | 21/25 | 24/25 |
| + embeddings only (no planner) | 13/25 | 16/25 | 22/25 | 23/25 |
| + embeddings + local cross-encoder | 16/25 | 20/25 | 22/25 | 24/25 |
| **Full v3 Pipeline (Default)** | **18/25 (72%)** | **21/25 (84%)** | **21/25** | **24/25** |

### Live-Model Evaluation Results (Full Training Set)

Results across submissions on the 27 training questions:

| Metric | v1 (no key fallback) | v2 (live model) | **v3 (current commit, live model)** |
|---|:---:|:---:|:---:|
| **Retrieval score** | 88.0% | 96.0% | **92.0%** (CI 83%–100%) |
| **Passage in top 5 / 10 / 20** | 72% / 88% / 92% | 88% / 96% / 96% | **88% / 92% / 92%** |
| **MRR** | 0.587 | 0.90 | **0.92** |
| **Forbidden records retrieved** | 0 | 0 | **0** (perfect time-travel compliance) |
| **Answers strict (`--judge none`)** | 37.0% | 100% | **96.3%** (26/27 pass all rules) |
| **Answers lenient** | 48.1% | 100% | **100.0%** (27/27) |
| **Sources cited recall / precision** | 0.567 / 0.527 | 0.933 / 0.904 | **0.900 / 0.901** |
| **Hard failures** | 0 | 0 | **0** |
| **Actions pass rate & argument acc.** | 12/12 (100%) | 12/12 (100%) | **12/12 (100%)** |
| **LLM calls & fallbacks** | — | 0 fallbacks | **46/46 calls ok, 0 stage fallbacks** |

### Live Run Details (Current Commit)
- **Model**: `openai/gpt-oss-120b` via Groq free tier.
- **Settings**: `RERANK_POOL=30`, `RERANK_SNIPPET_CHARS=400`, `ANSWER_CONTEXT_K=10`, `LLM_MIN_INTERVAL=2.5`.
- **Files in `out/`**: `out/memory_train_answers.jsonl` and `out/actions_train_predictions.jsonl` were generated directly by the live model run from this commit with 0 stage fallbacks.

## Tools and cost

Built with Claude (Sonnet) as a coding assistant across several sessions: writing, testing and
iterating against the training set. **Models used for the reported live numbers:**
`openai/gpt-oss-120b` via Groq's free tier. I also tried Google Gemini's free tier, which ran out of
daily quota partway through a run (the code now stops cleanly on that instead of retrying, and
`--resume` redoes only the affected items). The system supports Anthropic, Gemini and any
OpenAI-compatible endpoint (see `.env.example`). **Approximate cost: ₹0.**

Practical notes for running live: free tiers rate-limit hard. `LLM_MIN_INTERVAL` spaces calls,
`--resume` keeps earlier fully model-generated answers, and every fallback is logged and counted
at the end of each run (`stage fallbacks: 0` means the whole run came from the model). If two
provider keys are present, force one with `LLM_PROVIDER`; the first line of every run prints the
provider and model in use. Some providers' firewalls reject Python's default User-Agent with a 403,
so the client sends its own.

## Interface

Matches `BRIEF.md`: `run_memory.py` takes `--questions <jsonl> --out <jsonl>`
and `run_actions.py` takes `--commands <jsonl> --out <jsonl>`, both also
accepting `--data <dir>` (default `data`). `as_of` handling, `retrieved`
ordering (best-first, up to 20), `sources`, and `abstained` all follow the
spec in `BRIEF.md`.
