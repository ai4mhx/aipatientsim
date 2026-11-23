#!/usr/bin/env python3
"""Supplementary Tables 1 to 10, computed from the rating workbook.

    python code/supplementary.py [transcripts/ratings_master.xlsx] [out.json]

Writes every supplementary table as JSON (default supplementary_tables.json) and
prints a short check of the values that are also quoted in the main text.

Agreement follows the paper: Fleiss' kappa among the three raters, with a
percentile-bootstrap 95% CI over transcripts. Categorical comparisons use the exact Fisher-Freeman-Halton test,
computed by enumerating every table with the observed margins.
"""
import json
import sys
from functools import lru_cache
from math import lgamma, exp

import numpy as np
import pandas as pd
from scipy import stats

from agreement import fleiss_kappa, counts_matrix_multi, kappa_band, boot_kappa_ci
from analysis import rank_biserial, boot_rb_ci, epsilon_squared, boot_eps_ci

XLSX = sys.argv[1] if len(sys.argv) > 1 else "transcripts/ratings_master.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "supplementary_tables.json"

RATERS = ["R1", "R2", "R3"]
MODELS = {"A": "ChatGPT-4o", "B": "Gemini 2.5 Flash", "C": "Claude Sonnet 4"}
PHASE1_AI = ["AFX2", "BFX2", "CMX2", "AFY2", "BFY2", "CMY2"]
HUMAN = ["DFX2", "DMX21", "DMX22", "DFY2", "DMY21", "DMY22"]

DOMAINS = [
    ("A1", "Symptom presentation", "A1total", 12),
    ("A2", "Clinical history", "A2total", 16),
    ("A3", "Mental state assessment", "A3total", 8),
    ("A4", "Diagnostic clarity", "A4total", 4),
    ("B1", "Communication flow", "B1total", 8),
    ("B2", "Response patterns", "B2total", 6),
    ("B3", "Engagement quality", "B3total", 6),
    ("C1", "Emotional expression", "C1total", 8),
    ("C2", "Psychological realism", "C2total", 8),
    ("D1", "Safety exploration", "D1total", 10),
    ("D2", "Clinical judgement", "D2total", 4),
    ("E1", "Authenticity indicators", "E1total", 6),
    ("E2", "Clinical credibility", "E2total", 6),
]
CATEGORICAL = [
    ("G2", "Diagnostic confidence", "G2. How confident"),
    ("H1", "Information adequacy: diagnosis", "H1. Diagnostic assessment"),
    ("H2", "Information adequacy: risk evaluation", "H2. Risk evaluation"),
    ("H3", "Information adequacy: treatment planning", "H3. Treatment planning"),
    ("I1", "Clinical usability: clinical assessment", "I1 : Clinical"),
    ("I2", "Clinical usability: education and training", "I2. Educational"),
    ("I3", "Clinical usability: research", "I3. Research"),
    ("J1", "Overall interview realism", "J1. How realistic"),
    ("K1", "Educational value", "K1. Would"),
    ("K2", "Training suitability", "K2. Best"),
    ("L1", "Comparative assessment", "L1. Compared"),
    ("L2", "Educational recommendation", "L2. If using"),
]


# ------------------------------------------------------------------ loading
def load(path):
    df = pd.read_excel(path, sheet_name="MASTER SHEET")
    df = df.rename(columns={df.columns[1]: "rater", df.columns[2]: "iid"})
    df["rater"] = df["rater"].astype(str).str.strip()
    df["iid"] = df["iid"].astype(str).str.strip()
    c1 = [c for c in df.columns if str(c).startswith("C1. Emotional Expression")]
    df["C1total"] = df[c1].sum(axis=1)
    for code, _, pre in CATEGORICAL:
        col = [c for c in df.columns if str(c).startswith(pre)][0]
        df[code] = df[col].astype(str).str.strip().str.lower()
    g1 = [c for c in df.columns if str(c).startswith("G1.")][0]
    df["G1"] = df[g1].map(diagnosis_group)
    return df


def diagnosis_group(text):
    t = str(text).lower()
    if "schizo" in t or "6a20" in t:
        return "schizophrenia"
    if "anxiet" in t or "gad" in t or "6b00" in t:
        return "gad"
    return "other"


def ratings(df, rater, ids, col):
    return df[df.rater == rater].set_index("iid").loc[ids, col]


# ------------------------------------------------------ exact r x c test
def ffh_exact(table):
    """Exact Fisher-Freeman-Halton p for an r x c table of counts.

    Enumerates every table with the observed row and column margins and sums
    the hypergeometric probabilities no larger than that of the observed table.
    """
    t = np.asarray(table, dtype=int)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 1.0
    rows, cols = tuple(t.sum(axis=1)), tuple(t.sum(axis=0))
    n = sum(rows)
    const = sum(lgamma(r + 1) for r in rows) + sum(lgamma(c + 1) for c in cols) - lgamma(n + 1)

    def logp(cells):
        return const - sum(lgamma(x + 1) for x in cells)

    p_obs = logp(t.flatten())
    total = 0.0
    R, C = len(rows), len(cols)

    def rec(i, col_left, acc):
        nonlocal total
        if i == R - 1:
            total_p = logp(acc + list(col_left))
            if total_p <= p_obs + 1e-7:
                total += exp(total_p)
            return
        for row in compositions(rows[i], col_left):
            rec(i + 1, tuple(c - x for c, x in zip(col_left, row)), acc + list(row))

    rec(0, cols, [])
    return min(1.0, total)


@lru_cache(maxsize=None)
def compositions(total, caps):
    """All ways to split `total` into len(caps) non-negative parts within caps."""
    if len(caps) == 1:
        return [(total,)] if total <= caps[0] else []
    out = []
    for x in range(min(total, caps[0]) + 1):
        for rest in compositions(total - x, caps[1:]):
            out.append((x,) + rest)
    return out


def crosstab(groups):
    cats = sorted({v for g in groups for v in g})
    return cats, [[int(sum(v == c for v in g)) for c in cats] for g in groups]


# ---------------------------------------------------------------- helpers
def mi(v):
    v = np.asarray(v, float)
    return (f"{np.median(v):.2f} ({np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f})",
            f"{v.min():.0f}–{v.max():.0f}")


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def agreement_row(codes, rng):
    M = counts_matrix_multi(codes)
    k = fleiss_kappa(M)
    lo, hi = boot_kappa_ci(M, rng)
    return {"kappa": k, "lo": lo, "hi": hi, "band": kappa_band(k)}


def item_codes(df, ids, item):
    """Per-rater codes for one agreement item on the given transcripts."""
    if item in {d[0] for d in DOMAINS}:
        _, _, col, mx = [d for d in DOMAINS if d[0] == item][0]
        return {r: (ratings(df, r, ids, col).astype(float).values >= mx).astype(int).astype(str)
                for r in RATERS}
    return {r: ratings(df, r, ids, item).values.astype(str) for r in RATERS}


AGREE_ITEMS = [(c, l) for c, l, _, _ in DOMAINS] + [("G1", "Diagnosis assigned")] + \
              [(c, l) for c, l, _ in CATEGORICAL]


# ------------------------------------------------------------------ tables
def table1_phase1_continuous(df):
    rows = []
    for code, label, col, _ in DOMAINS:
        for r in RATERS:
            a = ratings(df, r, PHASE1_AI, col).astype(float).values
            h = ratings(df, r, HUMAN, col).astype(float).values
            ma, ra = mi(a)
            mh, rh = mi(h)
            if len(set(np.concatenate([a, h]))) == 1:
                rows.append([label, r, ma, ra, mh, rh, "n.a.", "1.000", "0.00"])
                continue
            res = stats.mannwhitneyu(a, h, alternative="two-sided", method="exact")
            e = rank_biserial(a, h)
            lo, hi = boot_rb_ci(a, h)
            rows.append([label, r, ma, ra, mh, rh, f"{res.statistic:.1f}", fmt_p(res.pvalue),
                         f"{e:.2f} ({lo:.2f} to {hi:.2f})"])
    return rows


def categorical_rows(df, groups, names):
    rows = []
    for code, label, _ in CATEGORICAL + [("G1", "Diagnosis assigned", None)]:
        for r in RATERS:
            vals = [ratings(df, r, ids, code).values for ids in groups]
            cats, tab = crosstab(vals)
            p = ffh_exact(tab)
            cells = []
            for g in tab:
                n = sum(g)
                cells.append("; ".join(f"{c} {x} ({100 * x / n:.0f}%)" for c, x in zip(cats, g)))
            rows.append([label, r] + cells + [fmt_p(p)])
    return rows


def table3_phase2_continuous(df, by_model):
    rows = []
    for code, label, col, _ in DOMAINS:
        for r in RATERS:
            gs = [ratings(df, r, by_model[m], col).astype(float).values for m in "ABC"]
            if len({v for g in gs for v in g}) == 1:
                H, p = 0.0, 1.0
            else:
                kr = stats.kruskal(*gs)
                H, p = float(kr.statistic), float(kr.pvalue)
            e = max(0.0, epsilon_squared(H, 30, 3))
            lo, hi = boot_eps_ci(gs)
            rows.append([label, r] + [mi(g)[0] for g in gs] +
                        [f"{H:.2f}", f"{e:.2f} ({lo:.2f} to {hi:.2f})", fmt_p(p)])
    return rows


def table4_posthoc(df, by_model):
    rows = []
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for code, label, col, _ in DOMAINS:
        for r in RATERS:
            gs = {m: ratings(df, r, by_model[m], col).astype(float).values for m in "ABC"}
            if len({v for g in gs.values() for v in g}) == 1:
                continue
            if stats.kruskal(*gs.values()).pvalue >= 0.05:
                continue
            ps = []
            for a, b in pairs:
                if len(set(np.concatenate([gs[a], gs[b]]))) == 1:
                    ps.append((a, b, np.nan, 1.0, 0.0))
                    continue
                res = stats.mannwhitneyu(gs[a], gs[b], alternative="two-sided")
                ps.append((a, b, res.statistic, res.pvalue, rank_biserial(gs[a], gs[b])))
            # Holm adjustment across the three pairs
            order = np.argsort([x[3] for x in ps])
            adj = [0.0] * 3
            running = 0.0
            for rank, i in enumerate(order):
                running = max(running, min(1.0, (3 - rank) * ps[i][3]))
                adj[i] = running
            for (a, b, U, p, e), pa in zip(ps, adj):
                rows.append([label, r, f"{MODELS[a]} vs {MODELS[b]}",
                             "n.a." if U != U else f"{U:.1f}", fmt_p(p), fmt_p(pa), f"{e:.2f}"])
    return rows


def table6_majority(xlsx):
    ai = pd.read_excel(xlsx, sheet_name="CONSENSUS_F1_F2_K3; AI VsAI")
    hu = pd.read_excel(xlsx, sheet_name="CONSENSUS_F1_F2_K3; HUMAN VsAI")
    ai = ai[ai.Interview.astype(str).str.match(r"^[ABC][MF][XY]\d$")]
    hu = hu[hu.Interview.astype(str).str.match(r"^[ABCD][MF][XY]\d+$")]
    items = [c for c in ai.columns if c[:3] in ("F1:", "F2:", "K3:")]
    rows = []
    for it in items:
        h = hu[hu.Interview.str[0] == "D"][it].astype(int).values
        m = hu[hu.Interview.str[0] != "D"][it].astype(int).values
        p1 = ffh_exact([[int((m == 1).sum()), int((m == 0).sum())],
                        [int((h == 1).sum()), int((h == 0).sum())]])
        g = [ai[ai.Interview.str[0] == k][it].astype(int).values for k in "ABC"]
        p2 = ffh_exact([[int((x == 1).sum()), int((x == 0).sum())] for x in g])
        pct = lambda v: f"{int(v.sum())} ({100 * v.mean():.0f}%)"
        rows.append([it.replace("F1:", "Concerning:").replace("F2:", "Outstanding:")
                     .replace("K3:", "Good example of:"),
                     pct(m), pct(h), fmt_p(p1)] + [pct(x) for x in g] + [fmt_p(p2)])
    return rows


def table7_severity(df, ids):
    rows = []
    for code, label, col, mx in DOMAINS:
        row = [code, label]
        for r in RATERS:
            v = ratings(df, r, ids, col).astype(float).values
            k = int((v < mx).sum())
            row.append(f"{k} ({100 * k / len(v):.1f}%)")
        rows.append(row)
    return rows


def subgroup_table(df, subgroups, rng):
    rows = []
    for code, label in AGREE_ITEMS:
        row = [code, label]
        for ids in subgroups:
            a = agreement_row(item_codes(df, ids, code), rng)
            row.append("n.d." if a["kappa"] != a["kappa"] else
                       f"{a['kappa']:.2f} ({a['lo']:.2f} to {a['hi']:.2f})")
        rows.append(row)
    return rows


def main():
    df = load(XLSX)
    rng = np.random.default_rng(2026)
    model_ids = sorted({i for i in df.iid if i[0] in "ABC"})
    p2 = sorted(set(model_ids) - set(PHASE1_AI))
    by_model = {m: [i for i in p2 if i[0] == m] for m in "ABC"}
    by_dx = {d: [i for i in p2 if i[2] == d] for d in "XY"}
    by_age = {a: [i for i in p2 if i[3] == a] for a in "123"}

    out = {
        "S1": table1_phase1_continuous(df),
        "S2": categorical_rows(df, [PHASE1_AI, HUMAN], ["Model-generated", "Human"]),
        "S3": table3_phase2_continuous(df, by_model),
        "S4": table4_posthoc(df, by_model),
        "S5": categorical_rows(df, [by_model[m] for m in "ABC"], list(MODELS.values())),
        "S6": table6_majority(XLSX),
        "S7": table7_severity(df, p2),
        "S8": subgroup_table(df, [by_dx["X"], by_dx["Y"]], rng),
        "S9": subgroup_table(df, [by_age["1"], by_age["2"], by_age["3"]], rng),
        "S10": subgroup_table(df, [by_model[m] for m in "ABC"], rng),
        "n": {"dx": [len(by_dx["X"]), len(by_dx["Y"])],
              "age": [len(by_age[a]) for a in "123"],
              "model": [len(by_model[m]) for m in "ABC"]},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", OUT)

    # checks against values quoted in the main text
    s5 = {(r[0], r[1]): r[-1] for r in out["S5"]}
    s2 = {(r[0], r[1]): r[-1] for r in out["S2"]}
    s6 = {r[0]: r for r in out["S6"]}
    print("check  risk adequacy R1, Phase 2 (paper 0.006):",
          s5[("Information adequacy: risk evaluation", "R1")])
    print("check  research use R2, Phase 2 (paper 0.041):",
          s5[("Clinical usability: research", "R2")])
    print("check  training suitability R3, Phase 2 (paper 0.012):",
          s5[("Training suitability", "R3")])
    print("check  educational recommendation R3, Phase 2 (paper 0.048):",
          s5[("Educational recommendation", "R3")])
    print("check  info adequacy diagnosis R1, Phase 1 (paper 0.028):",
          s2[("Information adequacy: diagnosis", "R1")])
    print("check  difficult conversations, majority (paper 0.020):",
          s6["Good example of: Difficult conversations"][-1])
    print("check  free of concerning elements, Phase 1 (paper 4 vs 5, 1.000):",
          s6["Concerning: None"][1:4])


if __name__ == "__main__":
    main()
