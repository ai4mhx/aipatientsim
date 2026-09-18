#!/usr/bin/env python3
"""Phase 1 and Phase 2 analyses.

Phase 1  six human interviews versus six matched model-generated interviews,
         Mann-Whitney U with exact p, rank-biserial effect size, bootstrap CI.
Phase 2  ChatGPT-4o versus Gemini 2.5 Flash versus Claude Sonnet 4, ten each,
         Kruskal-Wallis with epsilon-squared and bootstrap CI.
Also     diagnostic accuracy against the target diagnosis encoded in each
         transcript identifier.

    python code/analysis.py [transcripts/ratings_master.xlsx]
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

XLSX = sys.argv[1] if len(sys.argv) > 1 else "transcripts/ratings_master.xlsx"
RNG = np.random.default_rng(20260918)
N_BOOT = 6000

RATERS = ["R1", "R2", "R3"]
MODELS = {"A": "ChatGPT-4o", "B": "Gemini 2.5 Flash", "C": "Claude Sonnet 4"}
DIAGNOSIS = {"X": "GAD", "Y": "Schizophrenia"}

PHASE1_AI = ["AFX2", "BFX2", "CMX2", "AFY2", "BFY2", "CMY2"]
HUMAN = ["DFX2", "DMX21", "DMX22", "DFY2", "DMY21", "DMY22"]

DOMAINS = [
    ("Symptom presentation", "A1total", 12),
    ("Clinical history", "A2total", 16),
    ("Mental state assessment", "A3total", 8),
    ("Diagnostic clarity", "A4total", 4),
    ("Communication flow", "B1total", 8),
    ("Response patterns", "B2total", 6),
    ("Engagement quality", "B3total", 6),
    ("Emotional expression", "C1total", 8),
    ("Psychological realism", "C2total", 8),
    ("Safety exploration", "D1total", 10),
    ("Clinical judgement", "D2total", 4),
    ("Authenticity indicators", "E1total", 6),
    ("Clinical credibility", "E2total", 6),
]


def load(path):
    df = pd.read_excel(path, sheet_name="MASTER SHEET")
    df = df.rename(columns={df.columns[1]: "rater", df.columns[2]: "iid"})
    df["rater"] = df["rater"].astype(str).str.strip()
    df["iid"] = df["iid"].astype(str).str.strip()
    c1 = [c for c in df.columns if str(c).startswith("C1. Emotional Expression")]
    df["C1total"] = df[c1].sum(axis=1)
    return df


def med_iqr(v):
    return f"{np.median(v):5.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]"


def rank_biserial(a, b):
    """Positive when a tends to exceed b."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def boot_rb_ci(a, b, n=N_BOOT):
    out = np.empty(n)
    for i in range(n):
        x = RNG.choice(a, len(a), replace=True)
        y = RNG.choice(b, len(b), replace=True)
        gt = (x[:, None] > y[None, :]).sum()
        eq = (x[:, None] == y[None, :]).sum()
        out[i] = 2.0 * (gt + 0.5 * eq) / (len(a) * len(b)) - 1.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def epsilon_squared(H, n, k):
    return (H - k + 1) / (n - k)


def boot_eps_ci(groups, n=2000):
    k, N = len(groups), sum(len(g) for g in groups)
    out = np.empty(n)
    for i in range(n):
        gb = [RNG.choice(g, len(g), replace=True) for g in groups]
        try:
            H = stats.kruskal(*gb).statistic
        except ValueError:
            H = 0.0
        out[i] = max(0.0, epsilon_squared(H, N, k))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def phase1(df):
    print("=" * 96)
    print("PHASE 1  model-generated (n = 6) versus human (n = 6), matched on age "
          "group and diagnosis")
    print("=" * 96)
    print(f"{'domain':26s} {'R':3s} {'model-generated':>20s} {'human':>20s} "
          f"{'U':>6s} {'p':>7s} {'r_rb':>7s} {'95% CI':>18s}")
    for label, col, _mx in DOMAINS:
        for r in RATERS:
            s = df[df.rater == r].set_index("iid")
            a = s.loc[PHASE1_AI, col].astype(float).values
            h = s.loc[HUMAN, col].astype(float).values
            if len(set(np.concatenate([a, h]))) == 1:
                print(f"{label:26s} {r:3s} {med_iqr(a):>20s} {med_iqr(h):>20s} "
                      f"{'n.a.':>6s} {1.0:7.3f} {0.0:+7.2f} "
                      f"{'identical scores':>18s}")
                continue
            res = stats.mannwhitneyu(a, h, alternative="two-sided", method="exact")
            e = rank_biserial(a, h)
            lo, hi = boot_rb_ci(a, h)
            print(f"{label:26s} {r:3s} {med_iqr(a):>20s} {med_iqr(h):>20s} "
                  f"{res.statistic:6.1f} {res.pvalue:7.3f} {e:+7.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]")
        print()


def phase2(df, phase2_ids):
    print("=" * 96)
    print("PHASE 2  three models, n = 10 each, Kruskal-Wallis")
    print("=" * 96)
    by_model = {m: [i for i in phase2_ids if i[0] == m] for m in "ABC"}
    print(f"{'domain':26s} {'R':3s} {'ChatGPT-4o':>13s} {'Gemini':>13s} "
          f"{'Claude':>13s} {'H(2)':>7s} {'p':>7s} {'eps2':>6s} {'95% CI':>16s}")
    for label, col, _mx in DOMAINS:
        for r in RATERS:
            s = df[df.rater == r].set_index("iid")
            gs = [s.loc[by_model[m], col].astype(float).values for m in "ABC"]
            if len({v for g in gs for v in g}) == 1:
                H, p = 0.0, 1.0
            else:
                kr = stats.kruskal(*gs)
                H, p = float(kr.statistic), float(kr.pvalue)
            e = epsilon_squared(H, 30, 3)
            lo, hi = boot_eps_ci(gs)
            meds = [f"{np.median(g):.2f}" for g in gs]
            print(f"{label:26s} {r:3s} {meds[0]:>13s} {meds[1]:>13s} "
                  f"{meds[2]:>13s} {H:7.3f} {p:7.4f} {e:6.3f} [{lo:.3f}, {hi:.3f}]")
        print()


def diagnostic_accuracy(df, phase2_ids):
    col = [c for c in df.columns if str(c).startswith("G1. Clinical Diagnosis")][0]

    def correct(text, target):
        t = str(text).lower()
        if target == "GAD":
            return int("anxiet" in t or "gad" in t or "6b00" in t)
        return int("schizo" in t or "6a20" in t)

    print("=" * 96)
    print("DIAGNOSTIC ACCURACY against the target diagnosis in the identifier")
    print("=" * 96)
    for name, group in [("human", HUMAN), ("Phase 1 model", PHASE1_AI),
                        ("Phase 2 model", phase2_ids)]:
        line = f"{name:16s}"
        for r in RATERS:
            s = df[df.rater == r].set_index("iid")
            ok = sum(correct(s.loc[i, col], DIAGNOSIS[i[2]]) for i in group)
            line += f"  {r} {ok:2d}/{len(group):2d} ({ok / len(group) * 100:5.1f}%)"
        print(line)
    print()


def main():
    df = load(XLSX)
    model_ids = sorted({i for i in df.iid if i[0] in "ABC"})
    p2 = sorted(set(model_ids) - set(PHASE1_AI))
    print(f"{len(df)} ratings, {df.iid.nunique()} transcripts, "
          f"Phase 1 n = {len(PHASE1_AI)} + {len(HUMAN)}, Phase 2 n = {len(p2)}\n")
    phase1(df)
    phase2(df, p2)
    diagnostic_accuracy(df, p2)
    print("Non-significant results indicate absence of evidence for a difference.")
    print("They are not evidence of equivalence: this study was not powered for it.")


if __name__ == "__main__":
    main()
