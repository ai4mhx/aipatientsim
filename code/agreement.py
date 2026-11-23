#!/usr/bin/env python3
"""Inter-rater agreement for the PIAQ (Table 4 and Fig. 5 of the paper).

Agreement among the three raters is reported as Fleiss' kappa, with a
percentile-bootstrap 95% CI over transcripts (6,000 resamples), interpreted
with the Landis and Koch strength-of-agreement categories.

Agreement is computed on the 30 Phase 2 transcripts. Each ordinal domain is
coded as maximum score or below maximum, which is the coding specified on the
CODES tab of the rating workbook. Categorical items are analysed on their
observed response categories, unweighted, and the diagnosis is grouped as
generalised anxiety disorder, schizophrenia or other.

Kappa is low when nearly all ratings fall in one category, so the script also
prints, for each domain, how many transcripts each rater scored below the
maximum (Supplementary Table 7).

    python code/agreement.py [transcripts/ratings_master.xlsx]
"""
import sys

import numpy as np
import pandas as pd

XLSX = sys.argv[1] if len(sys.argv) > 1 else "transcripts/ratings_master.xlsx"
N_BOOT = 6000
SEED = 2026

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
# code, label, column prefix in the MASTER SHEET
CATEGORICAL = [
    ("G2", "Diagnostic confidence", "G2. How confident"),
    ("H1", "Information adequacy: diagnosis", "H1. Diagnostic assessment"),
    ("H2", "Information adequacy: risk", "H2. Risk evaluation"),
    ("H3", "Information adequacy: treatment", "H3. Treatment planning"),
    ("I1", "Clinical usability: clinical", "I1 : Clinical"),
    ("I2", "Clinical usability: training", "I2. Educational"),
    ("I3", "Clinical usability: research", "I3. Research"),
    ("J1", "Overall interview realism", "J1. How realistic"),
    ("K1", "Educational value", "K1. Would"),
    ("K2", "Training suitability", "K2. Best"),
    ("L1", "Comparative assessment", "L1. Compared"),
    ("L2", "Educational recommendation", "L2. If using"),
]

RATERS = ["R1", "R2", "R3"]
# the six model-generated transcripts matched to human participants in Phase 1;
# Phase 2 is every other model-generated transcript
PHASE1_AI = ["AFX2", "BFX2", "CMX2", "AFY2", "BFY2", "CMY2"]


def diagnosis_group(text):
    t = str(text).lower()
    if "schizo" in t or "6a20" in t:
        return "schizophrenia"
    if "anxiet" in t or "gad" in t or "6b00" in t:
        return "gad"
    return "other"


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


# ------------------------------------------------------------ coefficients
def counts_matrix(codes):
    """Binary codes: dict rater -> 0/1 array. Returns items x 2 counts."""
    items = len(next(iter(codes.values())))
    M = np.zeros((items, 2))
    for arr in codes.values():
        for i, v in enumerate(arr):
            M[i, int(v)] += 1
    return M


def counts_matrix_multi(codes):
    """Any categorical codes: dict rater -> array. Returns items x categories."""
    arrs = [np.asarray(a).astype(str) for a in codes.values()]
    cats = sorted(set(np.concatenate(arrs)))
    M = np.zeros((len(arrs[0]), len(cats)))
    for a in arrs:
        for i, v in enumerate(a):
            M[i, cats.index(v)] += 1
    return M


def percent_agreement(M):
    """Mean proportion of agreeing rater pairs per item (Fleiss P-bar)."""
    n_r = M.sum(axis=1)[0]
    return float((((M * (M - 1)).sum(axis=1)) / (n_r * (n_r - 1))).mean())


def fleiss_kappa(M):
    items, q = M.shape
    n_r = M.sum(axis=1)[0]
    pj = M.sum(axis=0) / (items * n_r)
    Pa = percent_agreement(M)
    Pe = float((pj ** 2).sum())
    if abs(1 - Pe) < 1e-12:
        return np.nan
    return float((Pa - Pe) / (1 - Pe))


def boot_kappa_ci(M, rng, n=N_BOOT):
    """Percentile-bootstrap 95% CI for Fleiss' kappa, resampling transcripts."""
    out = []
    for _ in range(n):
        k = fleiss_kappa(M[rng.integers(0, len(M), len(M))])
        if k == k:
            out.append(k)
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def kappa_band(k):
    """Landis and Koch strength-of-agreement categories, applied to kappa."""
    if k != k:
        return "n/a"
    if k < 0:
        return "Poor"
    for lo, name in [(0.81, "Almost perfect"), (0.61, "Substantial"),
                     (0.41, "Moderate"), (0.21, "Fair")]:
        if k >= lo:
            return name
    return "Slight"


# ------------------------------------------------------------------- codes
def item_codes(df, ids):
    """Yield (code, label, {rater: codes}) for every agreement item."""
    for code, label, col, mx in DOMAINS:
        yield code, label, {
            r: (df[df.rater == r].set_index("iid").loc[ids, col].astype(float).values
                >= mx).astype(int).astype(str) for r in RATERS}
    g1 = [c for c in df.columns if str(c).startswith("G1.")][0]
    yield "G1", "Diagnosis assigned", {
        r: np.array([diagnosis_group(v) for v in
                     df[df.rater == r].set_index("iid").loc[ids, g1].values])
        for r in RATERS}
    for code, label, pre in CATEGORICAL:
        col = [c for c in df.columns if str(c).startswith(pre)][0]
        yield code, label, {
            r: df[df.rater == r].set_index("iid").loc[ids, col]
                 .astype(str).str.strip().str.lower().values for r in RATERS}


def main():
    df = load(XLSX)
    model_ids = sorted({i for i in df.iid if i[0] in "ABC"})
    phase2 = sorted(set(model_ids) - set(PHASE1_AI))
    if len(phase2) != 30:
        raise SystemExit(f"expected 30 Phase 2 transcripts, found {len(phase2)}")
    rng = np.random.default_rng(SEED)

    print(f"Inter-rater agreement, {len(phase2)} Phase 2 transcripts, "
          f"{len(RATERS)} raters")
    print("Fleiss' kappa with bootstrap 95% CI; Landis and Koch categories.\n")
    header = f"{'code':5s} {'item':32s} {'kappa':>7s} {'95% CI':>17s}  strength"
    print(header)
    print("-" * len(header))
    for code, label, codes in item_codes(df, phase2):
        M = counts_matrix_multi(codes)
        k = fleiss_kappa(M)
        lo, hi = boot_kappa_ci(M, rng)
        print(f"{code:5s} {label:32s} {k:+7.3f} [{lo:+7.3f},{hi:+7.3f}]  "
              f"{kappa_band(k)}")

    print("\nTranscripts scored below the scale maximum, by rater (of 30)")
    print(f"{'code':5s} {'domain':32s} " + " ".join(f"{r:>4s}" for r in RATERS))
    for code, label, col, mx in DOMAINS:
        n = [int((df[df.rater == r].set_index("iid").loc[phase2, col]
                  .astype(float).values < mx).sum()) for r in RATERS]
        print(f"{code:5s} {label:32s} " + " ".join(f"{x:4d}" for x in n))

    print("\nNote. Kappa is close to zero when nearly all ratings fall in one")
    print("category, as for diagnostic clarity and usability. See README.")


if __name__ == "__main__":
    main()
