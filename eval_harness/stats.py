"""Statistics. Questions about one storyline are correlated, so confidence intervals resample
whole storylines, not single questions."""
import random
from collections import defaultdict


def cluster_bootstrap(items, key="storyline", value="score", iters=2000, seed=7):
    """Mean of `value` with a 95% interval from resampling clusters."""
    if not items:
        return {"n": 0, "clusters": 0, "mean": None, "ci95": None}
    clusters = defaultdict(list)
    for it in items:
        clusters[it.get(key) or it["id"]].append(float(it[value]))
    names = sorted(clusters)
    mean = sum(float(it[value]) for it in items) / len(items)
    if len(names) < 2:
        return {"n": len(items), "clusters": len(names), "mean": round(mean, 4), "ci95": None}
    r = random.Random(seed)
    stats = []
    for _ in range(iters):
        flat = [x for _ in names for x in clusters[r.choice(names)]]
        stats.append(sum(flat) / len(flat))
    stats.sort()
    return {"n": len(items), "clusters": len(names), "mean": round(mean, 4),
            "ci95": [round(stats[int(0.025 * iters)], 4), round(stats[int(0.975 * iters) - 1], 4)]}


def paired_permutation(a, b, key="storyline", value="score", iters=10000, seed=11):
    """Two systems on the same questions: sign-flip test over per-storyline mean differences."""
    bi = {x["id"]: x for x in b}
    diffs = defaultdict(list)
    for x in a:
        if x["id"] in bi:
            diffs[x.get(key) or x["id"]].append(float(x[value]) - float(bi[x["id"]][value]))
    per = [sum(v) / len(v) for v in diffs.values()]
    if not per:
        return {"clusters": 0, "mean_diff": None, "p": None}
    obs = sum(per) / len(per)
    r = random.Random(seed)
    hits = sum(abs(sum(d if r.random() < 0.5 else -d for d in per) / len(per)) >= abs(obs) - 1e-12
               for _ in range(iters))
    return {"clusters": len(per), "mean_diff": round(obs, 4), "p": round((hits + 1) / (iters + 1), 5)}


def cohen_kappa(a, b):
    """Agreement between two graders on the same items, corrected for chance."""
    if not a or len(a) != len(b):
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)
