"""Statistics for paired condition comparisons.

Every comparison in this study is paired: the same 300 cases go through M0..M4
for the same model, so the right tests condition on the item.  Unpaired t-tests
over independent means would throw away exactly the variance reduction the
design was built to get.

Pure standard library -- no SciPy -- so the analysis reproduces anywhere.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    test: str
    wins: int = 0
    losses: int = 0
    ties: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "delta": round(self.delta, 4),
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "p_value": round(self.p_value, 6),
            "test": self.test,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "significant_at_05": self.p_value < 0.05,
        }


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 20260829,
    alpha: float = 0.05,
) -> PairedResult:
    """Bootstrap CI and two-sided p for ``mean(b) - mean(a)`` on paired items."""
    if len(a) != len(b):
        raise ValueError(f"paired inputs must align: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return PairedResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "paired_bootstrap")

    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    observed = sum(diffs) / n
    rng = random.Random(seed)

    deltas: List[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        deltas.append(total / n)
    deltas.sort()

    lo = deltas[max(0, int((alpha / 2) * n_resamples) - 1)]
    hi = deltas[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]

    # p-value by shifting the bootstrap distribution to a null of zero effect
    centred = [d - observed for d in deltas]
    extreme = sum(1 for d in centred if abs(d) >= abs(observed))
    p_value = (extreme + 1) / (n_resamples + 1)

    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return PairedResult(
        n=n,
        mean_a=sum(a) / n,
        mean_b=sum(b) / n,
        delta=observed,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        test="paired_bootstrap",
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
    )


def mcnemar(a: Sequence[float], b: Sequence[float]) -> PairedResult:
    """Exact McNemar test for paired binary outcomes (the accuracy metrics)."""
    if len(a) != len(b):
        raise ValueError("paired inputs must align")
    n = len(a)
    b_only = sum(1 for x, y in zip(a, b) if x < 0.5 <= y)  # a wrong, b right
    a_only = sum(1 for x, y in zip(a, b) if y < 0.5 <= x)  # a right, b wrong
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        # exact binomial two-sided under p=0.5
        tail = sum(
            math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    mean_a = sum(a) / n if n else 0.0
    mean_b = sum(b) / n if n else 0.0
    return PairedResult(
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=mean_b - mean_a,
        ci_low=float("nan"),
        ci_high=float("nan"),
        p_value=p_value,
        test="mcnemar_exact",
        wins=b_only,
        losses=a_only,
        ties=n - discordant,
    )


def holm_bonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, object]]:
    """Holm-Bonferroni correction.

    With 5 models x 4 condition contrasts x 2 benchmarks the family is large
    enough that uncorrected p-values would manufacture significance; Holm is
    uniformly more powerful than plain Bonferroni at the same error control.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict[str, object]] = {}
    prev = 0.0
    for index, (key, p) in enumerate(ordered):
        adjusted = min(1.0, max(prev, (m - index) * p))
        prev = adjusted
        out[key] = {
            "p_raw": round(p, 6),
            "p_adjusted": round(adjusted, 6),
            "reject": adjusted < alpha,
        }
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, int]:
    """Pearson r -- used for the base-ability vs KG-gain relationship (RQ)."""
    n = min(len(xs), len(ys))
    if n < 3:
        return (float("nan"), n)
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs[:n], ys[:n]))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs[:n]))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys[:n]))
    if dx == 0 or dy == 0:
        return (float("nan"), n)
    return (num / (dx * dy), n)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, int]:
    """Spearman rho -- the safer choice with only five models."""

    def rank(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    n = min(len(xs), len(ys))
    if n < 3:
        return (float("nan"), n)
    return pearson(rank(list(xs[:n])), rank(list(ys[:n])))


def cluster_paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str],
    *,
    n_resamples: int = 10000,
    seed: int = 20260829,
    alpha: float = 0.05,
) -> PairedResult:
    """Paired bootstrap resampling **clusters**, not items.

    TCM-CP generates many items per disease from one pathway, so items within a
    disease are not independent: they share a vignette template, a stage set
    and a treatment sub-graph. Resampling items would treat 5,358 correlated
    observations as 5,358 independent ones and shrink the standard error
    accordingly, manufacturing significance. Resampling whole diseases
    propagates the dependence into the interval.

    The same applies wherever several items derive from one source case.
    """
    if not (len(a) == len(b) == len(clusters)):
        raise ValueError("values and cluster labels must align")
    n = len(a)
    if n == 0:
        return PairedResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "cluster_bootstrap")

    grouped: Dict[str, List[float]] = {}
    for value_a, value_b, cluster in zip(a, b, clusters):
        grouped.setdefault(str(cluster), []).append(float(value_b) - float(value_a))
    keys = sorted(grouped)
    observed = sum(sum(v) for v in grouped.values()) / n

    rng = random.Random(seed)
    deltas: List[float] = []
    n_clusters = len(keys)
    for _ in range(n_resamples):
        total, count = 0.0, 0
        for _ in range(n_clusters):
            diffs = grouped[keys[rng.randrange(n_clusters)]]
            total += sum(diffs)
            count += len(diffs)
        deltas.append(total / count if count else 0.0)
    deltas.sort()

    lo = deltas[max(0, int((alpha / 2) * n_resamples) - 1)]
    hi = deltas[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    centred = [d - observed for d in deltas]
    extreme = sum(1 for d in centred if abs(d) >= abs(observed))
    p_value = (extreme + 1) / (n_resamples + 1)

    wins = sum(1 for diffs in grouped.values() for d in diffs if d > 0)
    losses = sum(1 for diffs in grouped.values() for d in diffs if d < 0)
    return PairedResult(
        n=n,
        mean_a=sum(a) / n,
        mean_b=sum(b) / n,
        delta=observed,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        test=f"cluster_bootstrap[{n_clusters}_clusters]",
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
    )


def hierarchical_macro_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    strata: Sequence[str],
    families: Optional[Sequence[str]] = None,
    clusters: Optional[Sequence[str]] = None,
    *,
    n_resamples: int = 10000,
    seed: int = 20260829,
    alpha: float = 0.05,
) -> PairedResult:
    """Paired test on a macro-average, weighting **families** equally.

    TCM-CP's nine subtasks range from 306 items to 988, so a pooled accuracy is
    53% stage lookup and monitoring and 5.5% transition decision. A model that
    reads stages well and moves patients on badly would read as a good pathway
    executor. The prespecified endpoint is therefore a macro-average -- and it
    has three levels, not two:

    1. mean over the items of a **subtask** (``strata``),
    2. mean over the subtasks of a **family** (``families``), so CP4 -- one
       capability probed four ways -- counts once rather than four times,
    3. mean over the six families.

    Getting level 2 wrong is not a rounding difference. An earlier version
    passed the nine subtasks straight in as strata, which gave CP4 a weight of
    4/9 while the reported point estimate gave it 1/6. On a set where only CP4
    improves from 0 to 1, the table said Δ = 0.167 and the test said Δ = 0.444:
    the confidence interval and the p-value described a quantity that appeared
    nowhere in the paper.

    Resampling is by ``clusters`` within each subtask, so TCM-CP's other
    dependence -- many items generated from one disease's pathway -- reaches
    the interval too. Subtasks and families are fixed, not resampled: they are
    the population, not a sample from one.

    ``families`` defaults to the strata themselves, which reduces to a flat
    macro over strata.
    """
    if not (len(a) == len(b) == len(strata)):
        raise ValueError("values and stratum labels must align")
    if families is not None and len(families) != len(a):
        raise ValueError("family labels must align with the values")
    if clusters is not None and len(clusters) != len(a):
        raise ValueError("cluster labels must align with the values")
    n = len(a)
    if n == 0:
        return PairedResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "hierarchical_macro_bootstrap")

    family_of: Dict[str, str] = {}
    # family -> subtask -> cluster -> [paired differences]
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    levels_a: Dict[str, Dict[str, List[float]]] = {}
    levels_b: Dict[str, Dict[str, List[float]]] = {}
    for i in range(n):
        subtask = str(strata[i])
        family = str(families[i]) if families is not None else subtask
        if family_of.setdefault(subtask, family) != family:
            raise ValueError(f"subtask {subtask!r} appears under two families")
        cluster = str(clusters[i]) if clusters is not None else str(i)
        grouped.setdefault(family, {}).setdefault(subtask, {}).setdefault(
            cluster, []
        ).append(float(b[i]) - float(a[i]))
        levels_a.setdefault(family, {}).setdefault(subtask, []).append(float(a[i]))
        levels_b.setdefault(family, {}).setdefault(subtask, []).append(float(b[i]))

    family_keys = sorted(grouped)

    def _macro(levels: Mapping[str, Mapping[str, Sequence[float]]]) -> float:
        per_family = [
            sum(sum(v) / len(v) for v in subtasks.values()) / len(subtasks)
            for subtasks in levels.values()
        ]
        return sum(per_family) / len(per_family)

    observed = _macro(
        {
            family: {
                subtask: [d for c in clusters_.values() for d in c]
                for subtask, clusters_ in subtasks.items()
            }
            for family, subtasks in grouped.items()
        }
    )

    rng = random.Random(seed)
    cluster_keys = {
        family: {subtask: sorted(c) for subtask, c in subtasks.items()}
        for family, subtasks in grouped.items()
    }
    deltas: List[float] = []
    for _ in range(n_resamples):
        family_totals = 0.0
        for family in family_keys:
            subtask_total, n_subtasks = 0.0, 0
            for subtask, names in cluster_keys[family].items():
                picked_sum, picked_n = 0.0, 0
                for _ in range(len(names)):
                    diffs = grouped[family][subtask][names[rng.randrange(len(names))]]
                    picked_sum += sum(diffs)
                    picked_n += len(diffs)
                subtask_total += picked_sum / picked_n if picked_n else 0.0
                n_subtasks += 1
            family_totals += subtask_total / n_subtasks if n_subtasks else 0.0
        deltas.append(family_totals / len(family_keys))
    deltas.sort()

    lo = deltas[max(0, int((alpha / 2) * n_resamples) - 1)]
    hi = deltas[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    centred = [d - observed for d in deltas]
    extreme = sum(1 for d in centred if abs(d) >= abs(observed))
    p_value = (extreme + 1) / (n_resamples + 1)

    wins = sum(
        1
        for f in grouped.values()
        for s in f.values()
        for c in s.values()
        for d in c
        if d > 0
    )
    losses = sum(
        1
        for f in grouped.values()
        for s in f.values()
        for c in s.values()
        for d in c
        if d < 0
    )
    n_subtasks = sum(len(s) for s in grouped.values())
    n_clusters = sum(len(c) for f in cluster_keys.values() for c in f.values())
    label = f"hierarchical_macro_bootstrap[{len(family_keys)}_families,{n_subtasks}_subtasks"
    label += f",{n_clusters}_clusters]" if clusters is not None else "]"
    return PairedResult(
        n=n,
        mean_a=_macro(levels_a),
        mean_b=_macro(levels_b),
        delta=observed,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        test=label,
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
    )


def stratified_macro_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    strata: Sequence[str],
    clusters: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> PairedResult:
    """Flat macro over ``strata`` -- the hierarchical test with one level."""
    return hierarchical_macro_bootstrap(a, b, strata, None, clusters, **kwargs)


def is_binary(values: Sequence[float], *, tolerance: float = 1e-9) -> bool:
    """Whether every observation is 0 or 1.

    Test selection is made from the data rather than from the caller's belief
    about a metric. TCMEval-SDT's composite is a weighted sum of four task
    scores and is continuous on [0, 1]; running McNemar on it would discard the
    magnitude of every paired difference and test a hypothesis about sign
    changes that the metric does not express.
    """
    return all(abs(v) <= tolerance or abs(v - 1.0) <= tolerance for v in values)


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> PairedResult:
    """Wilcoxon signed-rank test on paired differences.

    Reported alongside the bootstrap for continuous metrics: the bootstrap
    gives an interpretable effect size with a confidence interval, Wilcoxon
    gives a distribution-free significance check that does not assume the
    resampling distribution is well behaved at small n. They answer slightly
    different questions and disagreeing is informative.

    Uses a normal approximation with tie and continuity correction, which is
    accurate for the n >= 20 this study works at, and is flagged as
    approximate below that.
    """
    if len(a) != len(b):
        raise ValueError("paired inputs must align")
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    mean_a = sum(a) / len(a) if a else 0.0
    mean_b = sum(b) / len(b) if b else 0.0
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)

    if n == 0:
        return PairedResult(
            n=len(a), mean_a=mean_a, mean_b=mean_b, delta=0.0,
            ci_low=float("nan"), ci_high=float("nan"), p_value=1.0,
            test="wilcoxon_signed_rank", wins=0, losses=0, ties=len(a),
        )

    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    index = 0
    tie_correction = 0.0
    while index < n:
        end = index
        while end + 1 < n and abs(nonzero[order[end + 1]]) == abs(nonzero[order[index]]):
            end += 1
        average = (index + end) / 2 + 1
        group = end - index + 1
        if group > 1:
            tie_correction += group**3 - group
        for k in range(index, end + 1):
            ranks[order[k]] = average
        index = end + 1

    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    expected = n * (n + 1) / 4
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2) / 24
    if variance <= 0:
        p_value = 1.0
    else:
        z = (abs(w_plus - expected) - 0.5) / math.sqrt(variance)
        p_value = min(1.0, 2 * (1 - _normal_cdf(z)))

    return PairedResult(
        n=len(a), mean_a=mean_a, mean_b=mean_b,
        delta=sum(diffs) / len(diffs) if diffs else 0.0,
        ci_low=float("nan"), ci_high=float("nan"), p_value=p_value,
        test="wilcoxon_signed_rank" + ("_approx" if n < 20 else ""),
        wins=wins, losses=losses, ties=len(a) - wins - losses,
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 20260829,
) -> PairedResult:
    """Choose the right paired test for the data at hand.

    Binary outcomes (PA exact accuracy) get exact McNemar; anything continuous
    (the SDT composite and its four task scores) gets a paired bootstrap, which
    yields both a confidence interval on the mean paired difference and a
    two-sided p-value.
    """
    if not a:
        return PairedResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "empty")
    if is_binary(a) and is_binary(b):
        return mcnemar(a, b)
    return paired_bootstrap(a, b, n_resamples=n_resamples, seed=seed)
