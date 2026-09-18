#!/usr/bin/env python3
"""Colour figures for the Communications AI & Computing submission.

Figures 1 to 4 carry the values reported in Tables 1 to 4 of the manuscript.
Figure 5 is computed directly from THESIS_DATA__4_.xlsx, MASTER SHEET.
Palette is colourblind-safe (Okabe-Ito derived) and still legible in greyscale
because every series also differs in shape or hatch.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

XLSX = "transcripts/ratings_master.xlsx"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

INK = "#1a1a1a"
GREY = "#6f6f6f"
BLUE = "#0072B2"     # referent-anchored / model-generated
ORANGE = "#D55E00"   # phenomenological / human
GREEN = "#009E73"    # mixed
YELLOW = "#E69F00"
PURPLE = "#8064A2"
SKY = "#56B4E9"
PINK = "#CC79A7"

MODEL_C = {"ChatGPT-4o": "#009E73", "Gemini 2.5 Flash": "#E69F00",
           "Claude Sonnet 4": "#0072B2"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=400)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------- FIGURE 1
def figure1():
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, text, fill="white", edge=INK, fs=6.4, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.0",
                                    linewidth=lw, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, linespacing=1.5)

    def arrow(x1, y1, x2, y2, rad=0.0, col=GREY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=1.0, color=col,
                                     shrinkA=1, shrinkB=1,
                                     connectionstyle=f"arc3,rad={rad}"))

    for x, t, c in [(12, "Sources", GREY), (38, "Blinding", GREY),
                    (62, "Rating", GREY), (87, "Analysis", GREY)]:
        ax.text(x, 56.5, t, ha="center", va="center", fontsize=8.0,
                fontweight="bold", color=c)

    box(1, 39, 22, 12, "Real outpatient interviews\n$\\it{n}$ = 6, GAD 3, schizophrenia 3\n"
        "reference distribution", fill="#FDEBE0", edge=ORANGE)
    box(1, 20, 22, 14, "Model-generated interviews\n$\\it{n}$ = 36, 12 per model\n"
        "ChatGPT-4o, Gemini 2.5 Flash,\nClaude Sonnet 4", fill="#E3F0F8", edge=BLUE)
    box(1, 5, 22, 10, "Structured matrix\nage group $\\times$ sex\n$\\times$ ICD-11 diagnosis",
        fill="#F4F4F4")

    box(27, 22, 21, 24, "Anonymisation\nidentifiers, institution and\norigin cues removed\n\n"
        "Randomised\npresentation order\n\n42 transcripts in total", fill="#FAFAFA")

    box(52, 29, 19, 17, "Three consultant\npsychiatrists\nrating independently,\n"
        "blind to origin, to model\nand to target diagnosis", fill="#FAFAFA")
    box(52, 6, 19, 18, "PIAQ\n12 sections, 4 domains\nA clinical presentation\n"
        "B interactional quality\nC psychological authenticity\nD safety and credibility",
        fs=6.0, fill="#FAFAFA")

    box(75, 37, 24, 13, "Phase 1  human vs model\n$\\it{n}$ = 6 per group, matched\n"
        "Mann-Whitney U, Fisher\n(Fig. 2, Table 2)", fill="#FDEBE0", edge=ORANGE)
    box(75, 21, 24, 13, "Phase 2  model vs model\n$\\it{n}$ = 10 per model\n"
        "Kruskal-Wallis, Fisher\n(Fig. 3 and 5, Table 3)", fill="#E3F0F8", edge=BLUE)
    box(75, 4, 24, 14, "Instrument under test\ninter-rater agreement across\n"
        "all 30 Phase 2 transcripts\n(Fig. 4, Table 4)", fill="#E6F5F0", edge=GREEN, lw=1.4)

    arrow(23, 45, 27, 40); arrow(23, 27, 27, 32); arrow(23, 10, 27, 25, rad=0.15)
    arrow(48, 35, 52, 37); arrow(61.5, 29, 61.5, 24)
    arrow(71, 39, 75, 43); arrow(71, 35, 75, 28); arrow(71, 14, 75, 11)
    save(fig, "Figure1")


# ----------------------------------------------------------------- FIGURE 2
def figure2():
    domains = ["Symptom\npresentation", "Clinical\nhistory", "Mental state\nassessment",
               "Diagnostic\nclarity", "Communication\nflow", "Emotional\nexpression",
               "Psychological\nrealism", "Safety\nexploration"]
    maxima = [12, 16, 8, 4, 8, 8, 8, 10]
    data = {
        "Rater 1": [(11.0, 10.25, 11.75, 11.5, 11.0, 12.0, 0.7),
                    (16.0, 16.0, 16.0, 12.0, 8.0, 13.75, 0.03),
                    (8.0, 8.0, 8.0, 6.5, 5.25, 7.0, 0.13),
                    (4.0, 4.0, 4.0, 3.0, 2.0, 4.0, 0.18),
                    (8.0, 7.25, 8.0, 6.0, 6.0, 7.5, 0.13),
                    (8.0, 8.0, 8.0, 6.0, 4.25, 7.75, 0.07),
                    (8.0, 7.25, 8.0, 5.5, 4.0, 7.0, 0.07),
                    (9.5, 8.25, 10.0, 5.0, 5.0, 8.0, 0.04)],
        "Rater 2": [(12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0),
                    (16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 0.7),
                    (8.0, 7.25, 8.0, 7.5, 7.0, 8.0, 0.59),
                    (4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.7),
                    (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 1.0),
                    (8.0, 5.75, 8.0, 8.0, 8.0, 8.0, 0.82),
                    (5.5, 5.0, 6.75, 8.0, 7.25, 8.0, 0.24),
                    (6.0, 6.0, 6.0, 6.5, 6.0, 7.0, 0.48)],
        "Rater 3": [(12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0),
                    (16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 0.7),
                    (7.0, 7.0, 7.75, 8.0, 5.75, 8.0, 0.82),
                    (4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0),
                    (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 1.0),
                    (7.5, 7.0, 8.0, 8.0, 7.25, 8.0, 0.94),
                    (7.5, 7.0, 8.0, 6.0, 4.0, 8.0, 0.59),
                    (7.0, 6.25, 7.0, 6.0, 6.0, 6.0, 0.13)],
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.6), sharey=True)
    ypos = np.arange(len(domains))[::-1]
    for ax, (rater, rows) in zip(axes, data.items()):
        for yi, (mm, mlo, mhi, hm, hlo, hhi, p), mx in zip(ypos, rows, maxima):
            f = 100.0 / mx
            ax.plot([hlo * f, hhi * f], [yi + 0.17] * 2, color=ORANGE, lw=1.4,
                    solid_capstyle="round", alpha=.55)
            ax.plot([mlo * f, mhi * f], [yi - 0.17] * 2, color=BLUE, lw=1.4,
                    solid_capstyle="round", alpha=.55)
            ax.plot([hm * f], [yi + 0.17], marker="o", mfc="white", mec=ORANGE,
                    mew=1.5, ms=5.2, ls="none", zorder=3)
            ax.plot([mm * f], [yi - 0.17], marker="D", mfc=BLUE, mec=BLUE,
                    ms=4.4, ls="none", zorder=3)
            if p < 0.05:
                d = "model >" if mm * f > hm * f else "human >"
                ax.text(107, yi, f"$\\it{{p}}$ = {p:.3f}\n{d}", fontsize=6.4,
                        va="center", ha="left", fontweight="bold", linespacing=1.15,
                        color=BLUE if mm * f > hm * f else ORANGE)
        ax.set_title(rater, fontweight="bold", color=INK)
        ax.set_xlim(0, 140); ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(-0.8, len(domains) - 0.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", lw=0.4, color="#ececec"); ax.set_axisbelow(True)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(domains, fontsize=7.4)
    fig.supxlabel("Median domain score, % of scale maximum (bars show interquartile range)",
                  fontsize=8.6, y=-0.02)
    fig.legend(handles=[
        Line2D([], [], marker="o", mfc="white", mec=ORANGE, mew=1.5, ms=5.2,
               ls="none", label="Human interviews"),
        Line2D([], [], marker="D", mfc=BLUE, mec=BLUE, ms=4.4, ls="none",
               label="Model-generated interviews")],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save(fig, "Figure2")


# ----------------------------------------------------------------- FIGURE 3
def figure3():
    rows = [
        ("Clinical history",         [(0.006, "C"), (0.064, None), (0.815, None)]),
        ("Mental state assessment",  [(0.034, "C"), (0.113, None), (0.446, None)]),
        ("Communication flow",       [(0.001, "C"), (0.001, "C"), (0.731, None)]),
        ("Response patterns",        [(0.902, None), (0.003, "C"), (0.004, "G")]),
        ("Safety exploration",       [(0.005, "C"), (0.169, None), (0.229, None)]),
        ("Authenticity indicators",  [(0.619, None), (0.023, "C"), (0.307, None)]),
        ("Risk evaluation adequate", [(0.006, "C"), (0.879, None), (0.151, None)]),
        ("Research use suitable",    [(0.0004, "C"), (0.0004, "C"), (1.000, None)]),
        ("Completely realistic",     [(0.176, None), (0.168, None), (0.869, None)]),
        ("Training suitability",     [(1.000, None), (1.000, None), (0.012, "O")]),
        ("Education recommendation", [(0.446, None), (1.000, None), (0.048, "G")]),
    ]
    FILL = {"C": "#0072B2", "G": "#E69F00", "O": "#009E73"}
    NAME = {"C": "Claude Sonnet 4", "G": "Gemini 2.5 Flash", "O": "ChatGPT-4o"}
    HATCH = {"C": "//", "G": "\\\\", "O": "xx"}
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ny = len(rows)
    for i, (dom, cells) in enumerate(rows):
        y = ny - 1 - i
        for j, (p, fav) in enumerate(cells):
            sig = p < 0.05
            face = FILL[fav] if sig else "#fbfbfb"
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=face, alpha=.85 if sig else 1,
                                       hatch=HATCH[fav] if sig else None,
                                       edgecolor="white" if sig else "#d8d8d8", lw=1.2))
            lab = "$\\it{p}$ < 0.001" if p < 0.001 else f"$\\it{{p}}$ = {p:.3f}"
            bb = dict(boxstyle="square,pad=0.30", facecolor="white",
                      edgecolor="none", alpha=.92) if sig else None
            ax.text(j + .5, y + .60, lab, ha="center", va="center", fontsize=6.6,
                    fontweight="bold" if sig else "normal", bbox=bb, zorder=4,
                    color=INK if sig else GREY)
            ax.text(j + .5, y + .30, NAME[fav] if sig else "no evidence of a difference",
                    ha="center", va="center", fontsize=5.8, bbox=bb, zorder=4,
                    color=INK if sig else GREY)
    ax.set_xlim(0, 3); ax.set_ylim(0, ny)
    ax.set_xticks(np.arange(3) + .5)
    ax.set_xticklabels(["Rater 1", "Rater 2", "Rater 3"], fontweight="bold")
    ax.set_yticks(np.arange(ny) + .5)
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.4)
    ax.xaxis.set_ticks_position("top"); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=FILL[k], alpha=.85, hatch=HATCH[k],
                             edgecolor="white", label=f"Significant, {NAME[k]} highest")
                       for k in ["C", "G", "O"]] +
              [Patch(facecolor="#fbfbfb", edgecolor="#d8d8d8",
                     label="No credible evidence of a difference")],
              loc="upper center", bbox_to_anchor=(.5, -.04), ncol=2,
              frameon=False, fontsize=7.2)
    save(fig, "Figure3")


# ----------------------------------------------------------------- FIGURE 4
def figure4():
    d = [
        ("Information adequacy: diagnosis",  0.962, 0.929, 0.995, "R"),
        ("Information adequacy: treatment",  0.960, 0.924, 0.996, "R"),
        ("Clinical usability: training",     0.929, 0.842, 1.000, "R"),
        ("Diagnostic accuracy",              0.903, 0.801, 1.000, "R"),
        ("Diagnostic clarity",               0.876, 0.759, 0.993, "R"),
        ("Clinical usability: clinical",     0.876, 0.759, 0.993, "R"),
        ("Diagnostic confidence",            0.863, 0.769, 0.957, "R"),
        ("Training suitability",             0.819, 0.688, 0.950, "R"),
        ("Clinical judgement",               0.814, 0.664, 0.965, "R"),
        ("Symptom presentation",             0.569, 0.340, 0.799, "M"),
        ("Overall interview realism",        0.545, 0.317, 0.773, "M"),
        ("Educational value",                0.504, 0.330, 0.678, "M"),
        ("Safety exploration",               0.477, 0.224, 0.730, "M"),
        ("Clinical usability: research",     0.446, 0.171, 0.722, "M"),
        ("Comparative assessment",           0.395, 0.087, 0.703, "M"),
        ("Educational recommendation",       0.376, 0.150, 0.603, "M"),
        ("Communication flow",               0.234, -0.062, 0.529, "P"),
        ("Clinical history",                 0.160, -0.137, 0.457, "P"),
        ("Emotional expression",             0.111, -0.170, 0.391, "P"),
        ("Psychological realism",           -0.004, -0.222, 0.214, "P"),
        ("Information adequacy: risk",      -0.048, -0.280, 0.184, "P"),
        ("Mental state assessment",         -0.058, -0.260, 0.143, "P"),
        ("Response patterns",               -0.062, -0.273, 0.149, "P"),
        ("Engagement quality",              -0.098, -0.308, 0.113, "P"),
        ("Authenticity indicators",         -0.098, -0.322, 0.127, "P"),
        ("Clinical credibility",            -0.153, -0.325, 0.019, "P"),
    ]
    st = {"R": dict(marker="o", mfc=BLUE, mec=BLUE, ms=5.4, col=BLUE),
          "M": dict(marker="^", mfc=GREEN, mec=GREEN, ms=5.6, col=GREEN),
          "P": dict(marker="s", mfc="white", mec=ORANGE, ms=5.0, col=ORANGE)}
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    y = np.arange(len(d))[::-1]
    for lo, hi, c in [(0.81, 1.02, "#E8F1F8"), (0.61, 0.81, "#F2F7FA"),
                      (0.41, 0.61, "#EFF7F3"), (0.21, 0.41, "#FBF3EC"),
                      (-0.40, 0.21, "#FCEDE6")]:
        ax.axvspan(lo, hi, color=c, zorder=0)
    for lo, hi, name in [(0.81, 1.02, "Near-perfect"), (0.61, 0.81, "Substantial"),
                         (0.41, 0.61, "Moderate"), (0.21, 0.41, "Fair"),
                         (-0.40, 0.21, "Slight or none")]:
        ax.text((lo + hi) / 2, len(d) - .05, name, ha="center", va="bottom",
                fontsize=7.0, color=GREY)
    ax.axvline(0, color=GREY, lw=.9, ls=":", zorder=1)
    for v in (0.21, 0.81):
        ax.axvline(v, color=GREY, lw=.7, ls="--", zorder=1)
    for yi, (lab, k, lo, hi, cls) in zip(y, d):
        s = st[cls]
        ax.plot([lo, hi], [yi, yi], color=s["col"], lw=1.6, alpha=.55,
                solid_capstyle="round", zorder=2)
        ax.plot([k], [yi], ls="none", zorder=3, mew=1.4,
                **{kk: vv for kk, vv in s.items() if kk != "col"})
    ax.set_yticks(y); ax.set_yticklabels([x[0] for x in d])
    ax.set_ylim(-1.0, len(d) + .9); ax.set_xlim(-0.40, 1.02)
    ax.set_xlabel("Inter-rater agreement, Gwet's AC1 (95% confidence interval)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", mfc=BLUE, mec=BLUE, ms=5.4,
               label="Referent-anchored judgement"),
        Line2D([], [], ls="none", marker="^", mfc=GREEN, mec=GREEN, ms=5.6,
               label="Mixed or partly anchored"),
        Line2D([], [], ls="none", marker="s", mfc="white", mec=ORANGE, mew=1.4,
               ms=5.0, label="Phenomenological judgement")],
        loc="lower right", frameon=True, framealpha=1, edgecolor="#d8d8d8",
        borderpad=.6)
    ax.annotate("Agreement below\nchance expectation", xy=(-0.153, 0.15),
                xytext=(-0.20, 4.6), fontsize=7.2, color=ORANGE,
                arrowprops=dict(arrowstyle="->", lw=.9, color=ORANGE,
                                connectionstyle="arc3,rad=0.25"))
    save(fig, "Figure4")


# ----------------------------------------------------------------- FIGURE 5
def figure5():
    """Computed directly from the raw rating data."""
    df = pd.read_excel(XLSX,
                       sheet_name="MASTER SHEET")
    df = df.rename(columns={df.columns[1]: "rater", df.columns[2]: "iid"})
    df["iid"] = df["iid"].astype(str).str.strip()
    SRC = {"A": "ChatGPT-4o", "B": "Gemini 2.5 Flash", "C": "Claude Sonnet 4",
           "D": "Human"}
    df["src"] = df["iid"].str[0].map(SRC)
    order = ["Human", "ChatGPT-4o", "Gemini 2.5 Flash", "Claude Sonnet 4"]

    ADQ = [("Diagnostic\nassessment", "H1. Diagnostic assessment"),
           ("Risk\nevaluation", "H2. Risk evaluation"),
           ("Treatment\nplanning", "H3. Treatment planning")]
    USE = [("Clinical\nassessment", "I1 : Clinical assessment purposes"),
           ("Education\nand training", "I2. Educational/training use"),
           ("Research\npurposes", "I3. Research purposes")]
    K2 = "K2. Best suited for training:"

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.9),
                             gridspec_kw={"width_ratios": [3, 3, 2.1]})

    def stacked(ax, specs, levels, colours, title):
        n_s, n_l = len(specs), len(order)
        width = 0.8 / n_l
        for si, (lab, col) in enumerate(specs):
            for li, src in enumerate(order):
                v = df[df.src == src][col].astype(str).str.strip()
                tot = len(v)
                bottom = 0.0
                x = si + (li - (n_l - 1) / 2) * width
                for lev, c in zip(levels, colours):
                    frac = (v == lev).sum() / tot if tot else 0
                    ax.bar(x, frac, width * 0.92, bottom=bottom, color=c,
                           edgecolor="white", linewidth=0.6)
                    bottom += frac
        ax.set_xticks(range(n_s))
        ax.set_xticklabels([s[0] for s in specs], fontsize=7.2)
        ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.set_title(title, fontweight="bold", fontsize=8.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    stacked(axes[0], ADQ, ["Yes", "Partially", "No"],
            [BLUE, SKY, "#F0F0F0"], "a  Information adequacy")
    axes[0].set_ylabel("Ratings (%)")
    stacked(axes[1], USE, ["Yes", "No"], [GREEN, "#F0F0F0"],
            "b  Judged usable for")

    ax = axes[2]
    levels = ["Medical students", "Residents", "Experienced clinicians",
              "Not suitable for training"]
    cols = [YELLOW, ORANGE, PURPLE, "#F0F0F0"]
    for li, src in enumerate(order):
        v = df[df.src == src][K2].astype(str).str.strip()
        bottom = 0.0
        for lev, c in zip(levels, cols):
            frac = (v == lev).sum() / len(v)
            ax.bar(li, frac, 0.7, bottom=bottom, color=c, edgecolor="white", lw=0.6)
            bottom += frac
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Human", "ChatGPT-4o", "Gemini 2.5 Flash",
                        "Claude Sonnet 4"], fontsize=6.4, rotation=35,
                       ha="right")
    ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_title("c  Best suited for training", fontweight="bold", fontsize=8.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # group key for panels a and b
    hatch_key = [Patch(facecolor=BLUE, label="Yes"),
                 Patch(facecolor=SKY, label="Partially"),
                 Patch(facecolor="#F0F0F0", edgecolor="#cccccc", label="No")]
    axes[0].legend(handles=hatch_key, loc="lower left", bbox_to_anchor=(0, -0.34),
                   ncol=3, frameon=False, fontsize=6.8)
    axes[1].legend(handles=[Patch(facecolor=GREEN, label="Yes"),
                            Patch(facecolor="#F0F0F0", edgecolor="#cccccc",
                                  label="No")],
                   loc="lower left", bbox_to_anchor=(0, -0.34), ncol=2,
                   frameon=False, fontsize=6.8)
    axes[2].legend(handles=[Patch(facecolor=c, label=l) for l, c in
                            zip(["Medical students", "Residents",
                                 "Experienced clinicians", "Not suitable"], cols)],
                   loc="lower left", bbox_to_anchor=(-0.08, -0.40), ncol=1,
                   frameon=False, fontsize=6.2)

    fig.text(0.5, 1.00, "Within each group of bars, from left to right: "
             "Human, ChatGPT-4o, Gemini 2.5 Flash, Claude Sonnet 4",
             ha="center", fontsize=7.0, color=GREY)
    save(fig, "Figure5")


figure1(); figure2(); figure3(); figure4(); figure5()
