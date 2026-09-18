#!/usr/bin/env python3
"""Inter-rater agreement for the PIAQ domains.

Reports Gwet's AC1 (the coefficient used in the paper), Fleiss' kappa, the three
pairwise Cohen's kappas, and raw agreement, side by side for every domain.

Agreement is computed across all three raters on the 30 Phase 2 transcripts, with
each ordinal domain dichotomised at its scale maximum, which is the coding
specified on the CODES tab of the rating workbook.

    python code/agreement.py [path/to/ratings_master.xlsx]
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd

XLSX = sys.argv[1] if len(sys.argv) > 1 else "data/ratings_master.xlsx"

# code, label, total column, scale maximum
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

RATERS = ["R1", "R2", "R3"]
# the six model-generated transcripts matched to human participants in Phase 1;
# Phase 2 is every other model-generated transcript
PHASE1_AI = ["AFX2", "BFX2", "CMX2", "AFY2", "BFY2", "CMY2"]


def load(path):
    df = pd.read_excel(path, sheet_name="MASTER SHEET")
    df = df.rename(columns={df.columns[1]: "rater", df.columns[2]: "iid"})
    df["rater"] = df["rater"].astype(str).str.strip()
    df["iid"] = df["iid"].astype(str).str.strip()
    # emotional expression has no stored total; sum its four item columns
    c1 = [c for c in df.columns if str(c).startswith("C1. Emotional Expression")]
    df["C1total"] = df[c1].sum(axis=1)
    for r in RATERS:
        n = df[df.rater == r].iid.nunique()
        if n != 42:
            raise SystemExit(f"rater {r} has {n} distinct transcripts, expected 42")
    return df


def counts_matrix(codes):
    """codes: dict rater -> 0/1 array aligned by item. Returns items x categories."""
    items = len(next(iter(codes.values())))
    M = np.zeros((items, 2))
    for arr in codes.values():
        for i, v in enumerate(arr):
            M[i, int(v)] += 1
    return M


def percent_agreement(M):
    """Mean proportion of agreeing rater pairs per item (Fleiss P-bar)."""
    n_r = M.sum(axis=1)[0]
    return float((((M * (M - 1)).sum(axis=1)) / (n_r * (n_r - 1))).mean())


def gwet_ac1(M):
    items, q = M.shape
    n_r = M.sum(axis=1)[0]
    pj = M.sum(axis=0) / (items * n_r)
    Pa = percent_agreement(M)
    Pe = (pj * (1 - pj)).sum() / (q - 1)
    if abs(1 - Pe) < 1e-12:
        return np.nan, np.nan
    ac1 = (Pa - Pe) / (1 - Pe)
    # variance after Gwet (2008), linearisation estimator
    pa_i = ((M * (M - 1)).sum(axis=1)) / (n_r * (n_r - 1))
    pe_i = np.array([(M[i] / n_r * (1 - M[i] / n_r)).sum() / (q - 1)
                     for i in range(items)])
    infl = (pa_i - Pa) - 2 * (1 - ac1) * (pe_i - Pe)
    se = float(np.sqrt((infl ** 2).sum() / (items * (items - 1))) / (1 - Pe))
    return float(ac1), se


def fleiss_kappa(M):
    items, q = M.shape
    n_r = M.sum(axis=1)[0]
    pj = M.sum(axis=0) / (items * n_r)
    Pa = percent_agreement(M)
    Pe = float((pj ** 2).sum())
    if abs(1 - Pe) < 1e-12:
        return np.nan
    return float((Pa - Pe) / (1 - Pe))


def cohen_kappa(x, y):
    cats = sorted(set(x) | set(y))
    if len(cats) < 2:
        return np.nan
    po = float((x == y).mean())
    pe = float(sum((x == c).mean() * (y == c).mean() for c in cats))
    if abs(1 - pe) < 1e-12:
        return np.nan
    return (po - pe) / (1 - pe)


def band(ac1):
    """Landis and Koch benchmark bands, applied to AC1."""
    if ac1 != ac1:
        return "n/a"
    for lo, name in [(0.81, "Near-perfect"), (0.61, "Substantial"),
                     (0.41, "Moderate"), (0.21, "Fair")]:
        if ac1 >= lo:
            return name
    return "Slight or none"


def main():
    df = load(XLSX)
    model_ids = sorted({i for i in df.iid if i[0] in "ABC"})
    phase2 = sorted(set(model_ids) - set(PHASE1_AI))
    if len(phase2) != 30:
        raise SystemExit(f"expected 30 Phase 2 transcripts, found {len(phase2)}")

    print(f"Inter-rater agreement, {len(phase2)} Phase 2 transcripts, "
          f"{len(RATERS)} raters")
    print("Domains dichotomised at the scale maximum (CODES tab convention)\n")
    header = (f"{'code':5s} {'domain':26s} {'AC1':>7s} {'SE':>6s} "
              f"{'95% CI':>17s} {'Fleiss':>7s} {'k12':>6s} {'k13':>6s} "
              f"{'k23':>6s} {'raw':>5s}  band")
    print(header)
    print("-" * len(header))

    for code, label, col, mx in DOMAINS:
        codes = {}
        for r in RATERS:
            s = df[df.rater == r].set_index("iid").loc[phase2, col].astype(float)
            codes[r] = (s.values >= mx).astype(int)
        M = counts_matrix(codes)
        ac1, se = gwet_ac1(M)
        lo, hi = (ac1 - 1.96 * se, ac1 + 1.96 * se)
        fk = fleiss_kappa(M)
        pw = {f"{a}{b}": cohen_kappa(codes[a], codes[b])
              for a, b in combinations(RATERS, 2)}
        raw = float(np.mean([(codes[a] == codes[b]).mean()
                             for a, b in combinations(RATERS, 2)]))
        print(f"{code:5s} {label:26s} {ac1:+7.3f} {se:6.3f} "
              f"[{lo:+7.3f},{hi:+7.3f}] {fk:+7.3f} {pw['R1R2']:+6.2f} "
              f"{pw['R1R3']:+6.2f} {pw['R2R3']:+6.2f} {raw:5.2f}  {band(ac1)}")

    print("\nNote. Fleiss' kappa sits at or below zero for domains where AC1 is high.")
    print("That is the first kappa paradox: these domains are scored at the scale")
    print("maximum on 84 to 93 per cent of transcripts, so chance-corrected kappa")
    print("collapses even though observed agreement is high. See README.")


if __name__ == "__main__":
    main()
