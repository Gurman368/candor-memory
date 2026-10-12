"""First-stage candidate generation, beyond keyword search (v3).

Diptopal's feedback: the model can only re-sort what keyword search already found, so the
first stage has to find more. Each retriever below is independent, switchable with an env var
(so each can be measured on its own with tools/eval_first_stage.py), and their ranked lists
are merged with reciprocal rank fusion (RRF).

  ngram    char 4-gram TF-IDF cosine. Standard library only. Matches word forms and spelling
           variants (launch/launching, propose/proposal) that exact-token BM25 misses.   CAND_NGRAM
  embed    OPTIONAL local embedding model (sentence-transformers) that matches on meaning.
           Used only if the package is installed; otherwise skipped and logged.          CAND_EMBED
  source   BM25 run inside each source separately (meeting / slack / email / calendar /
           dictation / codex / chatgpt) and interleaved, so 889 short meeting segments
           can't crowd out a handful of emails.                                          CAND_PERSOURCE
  dates    records tied to a date the question mentions, including dates hidden behind
           words in the RECORD ("Friday", "tomorrow", "next Monday" resolved against that
           record's own timestamp, so nothing leaks in from later).                      CAND_DATES
  changes  after the first search, pull in records about the same thing that say it changed
           or was corrected, even outside the thread.                                    CAND_CHANGES

All of it is generic: nothing here knows any specific question or record.
"""
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import timedelta

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def enabled(name, default=True):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# --------------------------------------------------------------------------- fusion

def rrf(weighted_lists, k=60):
    """Reciprocal rank fusion. weighted_lists = [(ids, weight), ...] -> ids best-first."""
    score = defaultdict(float)
    first = {}
    for li, (ids, w) in enumerate(weighted_lists):
        for rank, i in enumerate(ids):
            score[i] += w / (k + rank + 1)
            first.setdefault(i, (li, rank))
    return sorted(score, key=lambda i: (-score[i], first[i]))


# --------------------------------------------------------------------------- char n-grams

_NORM_RE = re.compile(r"[^a-z0-9]+")
_HEADER_RE = re.compile(r"^\[[^\]]*\]\s*")


def _norm(text):
    return " " + _NORM_RE.sub(" ", _HEADER_RE.sub("", text, count=1).lower()).strip() + " "


def _grams(text, n=4):
    s = _norm(text)
    return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


class NgramIndex:
    def __init__(self, units, n=4):
        self.n = n
        self.ids = [u.id for u in units]
        N = len(units)
        tfs = [_grams(u.text, n) for u in units]
        df = Counter()
        for tf in tfs:
            df.update(tf.keys())
        self.idf = {g: math.log(1 + N / c) for g, c in df.items()}
        self.inv = defaultdict(list)
        self.norm = []
        for di, tf in enumerate(tfs):
            w = {g: (1 + math.log(c)) * self.idf[g] for g, c in tf.items()}
            nrm = math.sqrt(sum(x * x for x in w.values())) or 1.0
            self.norm.append(nrm)
            for g, x in w.items():
                self.inv[g].append((di, x))

    def rank(self, text, k=150):
        q = {g: (1 + math.log(c)) * self.idf.get(g, 0.0) for g, c in _grams(text, self.n).items()}
        qn = math.sqrt(sum(x * x for x in q.values())) or 1.0
        acc = defaultdict(float)
        for g, qw in q.items():
            if qw:
                for di, dw in self.inv.get(g, ()):
                    acc[di] += qw * dw
        order = sorted(acc, key=lambda di: -(acc[di] / (self.norm[di] * qn)))
        return [self.ids[di] for di in order[:k]]


_NGRAM_CACHE = {}


def ngram_rank(units, text, k=150):
    key = hash(tuple((u.id, len(u.text)) for u in units))
    idx = _NGRAM_CACHE.get(key)
    if idx is None:
        _NGRAM_CACHE.clear()  # one visible set at a time keeps memory flat
        idx = _NGRAM_CACHE[key] = NgramIndex(units)
    return idx.rank(text, k)


# --------------------------------------------------------------------------- optional embeddings

_EMBED = {"model": None, "tried": False, "cache": {}, "name": None}


def embed_status():
    """One line for the run banner: is the embedding retriever active?"""
    if not enabled("CAND_EMBED", True):
        return "embeddings: off (CAND_EMBED=0)"
    m = _load_embedder()
    if m is None:
        return "embeddings: not installed -> char n-gram retriever only (pip install sentence-transformers to enable)"
    return f"embeddings: {_EMBED['name']}"


def _load_embedder():
    if _EMBED["tried"]:
        return _EMBED["model"]
    _EMBED["tried"] = True
    try:
        from sentence_transformers import SentenceTransformer  # optional dependency
        name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _EMBED["model"], _EMBED["name"] = SentenceTransformer(name), name
    except Exception as e:  # not installed, offline, or model not downloadable
        print(f"  [embeddings unavailable: {type(e).__name__}: {str(e)[:100]}]", file=sys.stderr)
        _EMBED["model"] = None
    return _EMBED["model"]


def embed_rank(units, question, k=150, model=None):
    """Rank units by cosine similarity of meaning. `model` can be injected for tests; any object
    with encode(list_of_str) -> list of vectors. Returns [] if no embedder is available."""
    model = model or (_load_embedder() if enabled("CAND_EMBED", True) else None)
    if model is None:
        return []
    cache = _EMBED["cache"]
    todo = [u for u in units if (u.id, len(u.text)) not in cache]
    if todo:
        vecs = model.encode([_HEADER_RE.sub("", u.text, count=1)[:1000] for u in todo])
        for u, v in zip(todo, vecs):
            v = [float(x) for x in v]
            nrm = math.sqrt(sum(x * x for x in v)) or 1.0
            cache[(u.id, len(u.text))] = [x / nrm for x in v]
    q = [float(x) for x in model.encode([question])[0]]
    qn = math.sqrt(sum(x * x for x in q)) or 1.0
    q = [x / qn for x in q]
    sims = [(sum(a * b for a, b in zip(q, cache[(u.id, len(u.text))])), u.id) for u in units]
    sims.sort(reverse=True)
    return [i for _, i in sims[:k]]


# --------------------------------------------------------------------------- dates

_WD_RE = re.compile(r"\b(next|this|last|on|by|before|until|every)?\s*(" + "|".join(WEEKDAYS) + r")\b", re.I)


def _next_weekday(d, wd, forward=True):
    delta = (wd - d.weekday()) % 7
    if forward:
        delta = delta or 7
        return d + timedelta(days=delta)
    delta = (d.weekday() - wd) % 7 or 7
    return d - timedelta(days=delta)


def relative_dates(text, anchor_dt):
    """(month, day) pairs hidden behind relative words in `text`, resolved against the time
    the text was written/said (`anchor_dt`): today / tonight / tomorrow / yesterday and weekday
    names ("Friday", "next Monday", "last Tuesday"). Only the record's own timestamp is used."""
    low, d0, out = text.lower(), anchor_dt.date(), set()
    if re.search(r"\b(today|tonight|this morning|this afternoon)\b", low):
        out.add((d0.month, d0.day))
    if re.search(r"\btomorrow\b", low):
        d = d0 + timedelta(days=1)
        out.add((d.month, d.day))
    if re.search(r"\byesterday\b", low):
        d = d0 - timedelta(days=1)
        out.add((d.month, d.day))
    for m in _WD_RE.finditer(low):
        wd = WEEKDAYS.index(m.group(2))
        d = _next_weekday(d0, wd, forward=(m.group(1) or "").lower() != "last")
        out.add((d.month, d.day))
    return out


def unit_dates(u, extract_dates):
    """(mentioned, own): dates the text names (explicit or resolved) and the record's own date."""
    mentioned = set(extract_dates(u.text)) | relative_dates(u.text, u.time)
    return mentioned, {(u.time.month, u.time.day)}


def date_candidates(units, question, as_of_dt, extract_dates, extra_dates=()):
    """Records tied to a date the question refers to. Returns (matching units, the dates)."""
    qd = set(extract_dates(question)) | relative_dates(question, as_of_dt) | set(extra_dates)
    if not qd:
        return [], qd
    hit = []
    for u in units:
        mentioned, own = unit_dates(u, extract_dates)
        own_counts = u.source in ("calendar", "meeting", "dictation")
        if qd & mentioned or (own_counts and qd & own):
            hit.append(u)
    return hit, qd


# --------------------------------------------------------------------------- later changes

CHANGE_CUE = re.compile(
    r"\b(moved?|moving|push(?:ed|ing)?|slip(?:ped)?|delay(?:ed)?|resched\w*|postpon\w*|chang(?:e|ed|ing)|"
    r"instead|actually|correct(?:ed|ion)?|updat(?:e|ed)|revis(?:e|ed)|no longer|cancel\w*|scratch|"
    r"bump(?:ed)?|swap(?:ped)?|new (?:date|plan|number|target)|not\s+\d|is\s+\S+,?\s+not|from\s+\S+\s+to|"
    r"turns out|finally|locked|confirmed|walk(?:ed)? back)\b", re.I)


def change_candidates(units, seeds, question_tokens, tokenize, phrases, exclude, limit=8):
    """Records about the same thing as the top results that say it CHANGED or was corrected,
    wherever they live (not only in the same thread). Returned in date order so the answer
    writer sees every version of the fact."""
    if not seeds:
        return []
    seed_toks = Counter()
    for u in seeds:
        seed_toks.update(set(tokenize(u.text)))
    df = Counter()
    for u in units:
        df.update(set(tokenize(u.text)))
    N = len(units)
    key = {t for t, c in seed_toks.items() if (c >= 2 or t in question_tokens) and df[t] <= max(8, 0.06 * N)}
    key_phr = set()
    for u in seeds:
        key_phr |= phrases(u.text)
    scored = []
    for u in units:
        if u.id in exclude or not CHANGE_CUE.search(u.text):
            continue
        toks = set(tokenize(u.text))
        s = len(key & toks) + 2 * len(key_phr & phrases(u.text))
        if s >= 3:
            scored.append((s, u))
    scored.sort(key=lambda x: (-x[0], -x[1].time.timestamp()))
    chosen = [u for _, u in scored[:limit]]
    return sorted(chosen, key=lambda u: u.time)


# --------------------------------------------------------------------------- corpus glossary

_GLOSS_NOISE = {"https", "http", "github", "linear", "meet", "google", "attendees", "brightline", "example",
                "invite", "calendar", "confirmed", "comma", "request", "opened", "issue", "that's", "don't",
                "i'll", "it's", "i'm", "we're", "pull"}
_GLOSS_CACHE = {}


def build_glossary(units, tokenize, n_bigrams=100, n_tokens=300):
    """Salient vocabulary of THIS corpus: frequent two-word topics ("onboarding mockups", "dark mode")
    and distinctive single words ("denver", "geocoding"). The query planner is shown it so it can
    rewrite a question in the data's own words (a region -> the city the records name, a feature ->
    its internal name). Mined from the visible records only, nothing hand-written."""
    key = hash(tuple((u.id, len(u.text)) for u in units))
    if key in _GLOSS_CACHE:
        return _GLOSS_CACHE[key]
    bi, uni = Counter(), Counter()
    for u in units:
        body = _HEADER_RE.sub("", u.text, count=1)
        seen_b, seen_u = set(), set()
        for sent in re.split(r"[.!?\n]+", body):
            toks = [t for t in tokenize(sent) if t not in _GLOSS_NOISE and "@" not in t]
            seen_u.update(t for t in toks if len(t) > 4 and t.isalpha())
            seen_b.update(zip(toks, toks[1:]))
        bi.update(seen_b)
        uni.update(seen_u)
    bigrams = sorted(((" ".join(b), c) for b, c in bi.items()
                      if 3 <= c <= 60 and all(len(t) > 3 and t.isalpha() for t in b)), key=lambda x: (-x[1], x[0]))
    singles = sorted(((t, c) for t, c in uni.items() if 2 <= c <= 30), key=lambda x: (-x[1], x[0]))
    gloss = ", ".join([b for b, _ in bigrams[:n_bigrams]] + [t for t, _ in singles[:n_tokens]])
    _GLOSS_CACHE.clear()
    _GLOSS_CACHE[key] = gloss
    return gloss


# --------------------------------------------------------------------------- optional local reranker

_CROSS = {"model": None, "tried": False, "name": None}


def _load_cross():
    if _CROSS["tried"]:
        return _CROSS["model"]
    _CROSS["tried"] = True
    try:
        from sentence_transformers import CrossEncoder  # same optional package as the embeddings
        name = os.environ.get("CROSS_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _CROSS["model"], _CROSS["name"] = CrossEncoder(name), name
    except Exception as e:
        print(f"  [local reranker unavailable: {type(e).__name__}: {str(e)[:100]}]", file=sys.stderr)
        _CROSS["model"] = None
    return _CROSS["model"]


def cross_rank(by_id, ids, queries, model=None, text_chars=700):
    """A small local cross-encoder reads (query, record) pairs and scores each one jointly, which is
    more precise than the retrievers' similarity scores. Used to choose the best ~45 of the ~150
    first-pass candidates for the language model. Scores each record against every query given
    (the question, plus the model's planned searches) and keeps the best. [] if unavailable."""
    model = model or (_load_cross() if enabled("CAND_CROSS", True) else None)
    if model is None or not ids:
        return []
    pairs, owner = [], []
    for q in queries:
        for i in ids:
            pairs.append((q, by_id[i].text[:text_chars]))
            owner.append(i)
    scores = model.predict(pairs)
    best = {}
    for i, sc in zip(owner, scores):
        best[i] = max(best.get(i, float("-inf")), float(sc))
    return sorted(ids, key=lambda i: -best[i])


def retrieval_banner():
    """Status lines for the run banner: which optional local models are active."""
    out = [embed_status()]
    if enabled("CAND_CROSS", True):
        out.append(f"local reranker: {_CROSS['name']}" if _load_cross() else
                   "local reranker: not installed (pip install sentence-transformers to enable)")
    else:
        out.append("local reranker: off (CAND_CROSS=0)")
    return out
