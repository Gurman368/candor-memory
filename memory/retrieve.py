"""Retrieval over the units visible as_of a question's timestamp.

Design (see README for the full rationale):
  1. BM25 over unit text, generic full-text ranking.
  2. Phrase/entity bonus: multi-word capitalized phrases and acronyms shared between
     the question and a candidate are a much stronger relevance signal than single
     token overlap ("Route Planner v2" vs. the word "planner" alone), so they get an
     extra boost. The phrase list is mined from the corpus itself at query time, not
     hand-written per question, so it generalizes to unseen questions.
  3. Small recency tie-break: among similarly-scored candidates, prefer the more
     recent one -- most "what's true now" questions are best answered by the latest
     matching record, and irrelevant future/deleted records are already excluded by
     `visible()` before we ever see them.
  4. Graph expansion: once we have a ranked shortlist, we pull in units connected to
     it by structure the keyword search can't see -- the rest of a Slack/email
     thread, and any other unit explicitly referenced by id in a candidate's text.
     This is what recovers multi-hop / cross-source questions (e.g. a Slack message
     that only makes sense next to the rest of its thread).
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

from memory.ingest import Unit, load_visible
from memory.llm_client import LLMError, call_llm

STOPWORDS = set("""
a an the of to in on at for and or is are was were be been being this that these those
it its it's i you he she we they my your his her our their me him them us do does did
not no so if then than as with without about into over under again further out up down
what when where who whom which why how all any both each few more most other some such
only own same so than too very can will just don should now here there okay ok yeah um
uh let's lets going gonna get got have has had having day days from""".split())

WORD_RE = re.compile(r"[a-z0-9']+")
PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){1,3}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\d?\b")
HEADER_RE = re.compile(r"^\[[^\]]*\]\s*")

# A meeting/channel/subject header like "[Acme Freight - pricing and rollout, 2026-09-09]" is
# repeated verbatim in EVERY segment of that meeting/thread. Left in, its terms get "free" term
# frequency in hundreds of short documents, which (via BM25's document-length normalization)
# lets a one-line segment that just repeats the title outrank a long, specific document like the
# actual proposal email. So the header is stripped before BM25 tokenization; the speaker/sender
# prefix that follows it is kept, since that's real per-document signal (e.g. "Sarah Kim: ...").
def _strip_header(text):
    return HEADER_RE.sub("", text, count=1)


def tokenize(text):
    text = _strip_header(text)
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


# A small, generic (not data-specific) synonym map for query expansion -- BM25 only
# matches literal tokens, so "slip"/"delay"/"push back" all meaning the same thing
# in English would otherwise miss each other. Expansion is applied to the QUERY only,
# never to documents, so it can't manufacture a match that isn't really about the
# same concept.
SYNONYMS = {
    "slip": {"delay", "delayed", "slipped", "push", "pushed", "postpone", "postponed", "move", "moved", "slid"},
    "delay": {"slip", "slipped", "push", "pushed", "postpone", "postponed"},
    "sign": {"signed", "signing", "contract", "deal", "close", "closed", "closing"},
    "close": {"sign", "signed", "closing", "closed"},
    "launch": {"launching", "ship", "shipping", "shipped", "release", "released", "go-live", "golive"},
    "propose": {"proposal", "proposed", "pricing", "quote", "quoted"},
    "fly": {"flight", "flying", "travel", "traveling"},
    "hire": {"hiring", "hired", "headcount", "role", "position"},
    "own": {"owns", "owner", "assigned", "responsible"},
    "current": {"now", "latest", "currently"},
    "why": {"because", "reason", "regression", "issue", "bug"},
}


def expand_query_tokens(tokens):
    out = set(tokens)
    for t in tokens:
        out |= SYNONYMS.get(t, set())
    return list(out)


_EXPAND_PROMPT = (
    "Give 6-10 short keywords or 2-3 word phrases that would likely appear VERBATIM in "
    "workplace messages (Slack, email, meeting transcripts) discussing this question, "
    "INCLUDING plausible rewordings the question itself doesn't use (e.g. a cause, a "
    "status word like 'delayed'/'signed', a likely proper noun). One per line, no numbering, "
    "no explanation.\n\nQuestion: {q}"
)


def llm_expand_terms(question, timeout=20):
    """Optional LLM-based query expansion: closes vocabulary gaps a fixed synonym list
    can't (paraphrases, causal language, domain nouns), by asking the model what
    words a relevant message would actually contain. Returns [] silently if no API
    key is configured, expansion is disabled, or the call fails -- this is a bonus
    signal, not a dependency."""
    if os.environ.get("DISABLE_LLM_QUERY_EXPANSION"):
        return []
    prompt = _EXPAND_PROMPT.format(q=question)
    try:
        raw = call_llm(prompt, max_tokens=150, timeout=timeout, max_retries=2)
        if raw is None:
            return []
        terms = []
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-*\d.]+", "", line).strip()
            if line:
                terms.append(line)
        return terms
    except LLMError:
        return []


def header_token_set(text):
    """Tokens from the bracketed header only (e.g. a dictation's target contact, an
    email's date), as a flat set -- used for a small non-BM25 bonus so this metadata
    still helps (e.g. matching "Sarah Patel" in a dictation's target_context) without
    letting a title repeated across hundreds of sibling documents dominate BM25 stats."""
    m = HEADER_RE.match(text)
    if not m:
        return set()
    return {w for w in WORD_RE.findall(m.group(0).lower()) if w not in STOPWORDS and len(w) > 1}


def phrases(text):
    out = set(m.strip() for m in PHRASE_RE.findall(text))
    out |= set(ACRONYM_RE.findall(text))
    return {p.lower() for p in out if len(p) > 2}


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.N = len(docs)
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        df = Counter()
        for toks in self.doc_tokens:
            for w in set(toks):
                df[w] += 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.doc_tokens]

    def score(self, query_tokens, i):
        s = 0.0
        dl = self.doc_len[i]
        for w in query_tokens:
            if w not in self.idf:
                continue
            f = self.tf[i].get(w, 0)
            if f == 0:
                continue
            idf = self.idf[w]
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl)))
        return s


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
DATE_RE = re.compile(
    r"\b(\d{4})-(\d{2})-(\d{2})(?=[T\b])|"
    r"\b(" + "|".join(MONTHS) + r"|" + "|".join(m[:3] for m in MONTHS) + r")\.?\s+(\d{1,2})\b|"
    r"\b(\d{1,2})/(\d{1,2})\b", re.I)


def extract_dates(text):
    """(month, day) tuples mentioned anywhere in the text, in any of the formats the
    data uses (ISO, 'September 23', 'Sep 23', '9/23'). Used to answer "what happened
    on the same day as X" questions: a calendar join that keyword overlap alone can't
    do, since the connecting fact is a date, not a shared word."""
    out = set()
    for m in DATE_RE.finditer(text):
        if m.group(1):
            out.add((int(m.group(2)), int(m.group(3))))
        elif m.group(4):
            mo = MONTHS.index(m.group(4).lower()[:3] if len(m.group(4)) == 3 else m.group(4).lower()) + 1 \
                if m.group(4).lower() in MONTHS else [i for i, mn in enumerate(MONTHS) if mn.startswith(m.group(4).lower())][0] + 1
            out.add((mo, int(m.group(5))))
        elif m.group(6):
            mo, d = int(m.group(6)), int(m.group(7))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                out.add((mo, d))
    return out


def retrieve(question, as_of_str, data_dir, top_k=20, shortlist=15):
    units = load_visible(data_dir, as_of_str)
    if not units:
        return [], {}
    by_id = {u.id: u for u in units}
    texts = [u.text for u in units]
    bm25 = BM25(texts)
    q_tokens = expand_query_tokens(tokenize(question))
    q_phrases = phrases(question)
    for term in llm_expand_terms(question):
        q_tokens.extend(tokenize(term))
        q_phrases |= phrases(term)

    # time normalization for the recency tie-break, scaled to [0, 1]
    times = [u.time for u in units]
    tmin, tmax = min(times), max(times)
    span = (tmax - tmin).total_seconds() or 1.0

    q_token_set = set(q_tokens)
    scored = []
    for i, u in enumerate(units):
        s = bm25.score(q_tokens, i)
        low = u.text.lower()
        phrase_hits = sum(1 for p in q_phrases if p in low)
        s += phrase_hits * 3.0
        header_hits = len(q_token_set & header_token_set(u.text))
        s += header_hits * 1.6
        recency = (u.time - tmin).total_seconds() / span
        s += recency * 0.15
        if s > 0:
            scored.append((s, u))

    # Calendar-day join: "what's on my calendar the day I fly to Denver" needs a
    # different record (the board meeting) that shares no words with the question --
    # the only link is the date. If the query is asking about a day/schedule, find
    # the date(s) implied by the best current matches and pull in any calendar/meeting
    # record that falls on the same date.
    if any(w in question.lower() for w in ("day", "calendar", "schedule")):
        scored.sort(key=lambda x: -x[0])
        anchor_dates = set()
        for s, u in scored[:5]:
            anchor_dates |= extract_dates(u.text)
        if anchor_dates:
            score_by_id = {u.id: s for s, u in scored}
            top_score = scored[0][0] if scored else 10.0
            for u in units:
                own_dates = extract_dates(u.text)
                if u.source in ("calendar", "meeting"):
                    own_dates = own_dates | {(u.time.month, u.time.day)}
                if own_dates & anchor_dates:
                    score_by_id[u.id] = max(score_by_id.get(u.id, 0.0), top_score + 1.0)
            scored = [(score_by_id[u.id], u) for u in units if u.id in score_by_id]

    scored.sort(key=lambda x: (-x[0], -x[1].time.timestamp()))
    top = [u for _, u in scored[:shortlist]]

    # graph expansion: thread-mates and explicit id-mentions of the shortlist
    expanded = list(top)
    seen = {u.id for u in top}
    thread_groups = defaultdict(list)
    for u in units:
        key = (u.source, u.thread_id)
        if u.thread_id:
            thread_groups[key].append(u)

    for u in top:
        if u.thread_id:
            for mate in thread_groups.get((u.source, u.thread_id), []):
                if mate.id not in seen:
                    seen.add(mate.id)
                    expanded.append(mate)
        for mid in u.mentions:
            if mid in by_id and mid not in seen:
                seen.add(mid)
                expanded.append(by_id[mid])

    ranked_ids = [u.id for u in expanded][:top_k]
    debug = {"n_visible": len(units), "n_scored": len(scored), "shortlist": [u.id for u in top]}
    return ranked_ids, debug
