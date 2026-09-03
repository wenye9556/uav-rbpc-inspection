# -*- coding: utf-8 -*-
"""TRB paper figures v3 — follows the academic-figure-design skill:

  * flat, minimal, top-conference style; pure white background
  * soft muted palette only (no high-saturation colors)
  * families distinguished by subtle hue/value changes
  * short labels; no occlusion; captions carry the message
  * output: SVG only

Run: .venv/Scripts/python.exe agent_analysis/scripts/make_paper_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve()
ROOT = next((d for d in [_here.parent, *_here.parents]
             if (d / 'results').is_dir() and (d / 'step12_branch_price.py').exists()),
            _here.parents[2])
FIG = ROOT / "results" / "figures"
RES = ROOT / "results"
# LaTeX compile assets (PDF twins).  results/figures stays SVG-only by
# author decision; the PDFs live inside the latex project only.
LATEX_FIG = ROOT / "results" / "figures"  # release copy: PDFs next to SVGs

# ---- muted publication palette (seaborn "deep"-adjacent, softened) -------
BLUE = "#5B8DB8"      # R-BPC / certified (primary)
BLUE_LIGHT = "#B9CFE4"
ORANGE = "#E3A857"    # reference / baselines
GREEN = "#86B79B"     # coverage quantities
INK = "#3A3A3A"       # text
MUTE = "#9A9A9A"      # secondary text / structure

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.constrained_layout.use": True, "savefig.bbox": "tight",
    "axes.edgecolor": "#BBBBBB", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def read_main(path: Path) -> dict:
    return list(csv.DictReader(open(path / "E1_lex_certify.csv",
                                    encoding="utf-8-sig")))[0]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.svg")
    if LATEX_FIG.is_dir():
        fig.savefig(LATEX_FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"saved {name}.svg")


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=9.5,
                 color=INK, pad=7)


def soften(ax, ygrid=True):
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    if ygrid:
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.7)
        ax.set_axisbelow(True)


# ----------------------------------------------------------------- fig 1
def fig_scale_envelope():
    runs = {"n=10": RES / "experiments" / "E1_n10_v31",
            "n=12": RES / "experiments" / "E1_n12",
            "n=15": RES / "experiments" / "E1_n15_v31",
            "n=18": RES / "experiments" / "E1_n18"}
    labels = list(runs)
    rt = [float(read_main(d)["runtime_s"]) / 60.0 for d in runs.values()]
    nodes = []
    for d in runs.values():
        tel = json.loads((d / "E1_lex_certify_telemetry.json")
                         .read_text(encoding="utf-8"))
        nodes.append(tel["aggregate"]["pricing_nodes"] / 1e6)
    base_rt = float(read_main(ROOT / "results" / "model_experiments"
                              / "E1_lex_certify")["runtime_s"]) / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    xs = range(len(labels))
    ax.bar(xs, rt, width=0.5, color=BLUE)
    for x, v in zip(xs, rt):
        ax.text(x, v + 6, f"{v:.0f}", ha="center", fontsize=8, color=MUTE)
    ax.axhline(base_rt, color=ORANGE, ls=(0, (4, 3)), lw=1.1)
    ax.text(2.62, base_rt + 10, f"baseline {base_rt:.0f} min",
            ha="right", fontsize=8, color=ORANGE)
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("runtime (min)")
    ax.set_ylim(0, 470)
    soften(ax)
    panel(ax, "a", "Certified runtime")

    ax = axes[1]
    ax.bar(xs, nodes, width=0.5, color=BLUE)
    for x, v in zip(xs, nodes):
        ax.text(x, v + 0.5, f"{v:.1f}M", ha="center", fontsize=8, color=MUTE)
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("pricing prefix nodes (M)")
    ax.set_ylim(0, 21)
    soften(ax)
    panel(ax, "b", "Pricing tree size")
    save(fig, "fig_scale_envelope")


# ----------------------------------------------------------------- fig 2
def fig_acceleration():
    versions = [("reference", RES / "model_experiments" / "E1_lex_certify"),
                ("compiled", RES / "experiments" / "E1_n10_v20"),
                ("cached", RES / "experiments" / "E1_n10_v30"),
                ("hoisted", RES / "experiments" / "E1_n10_v31")]
    names, total, pb = [], [], []
    for name, d in versions:
        names.append(name)
        total.append(float(read_main(d)["runtime_s"]) / 60.0)
        a = json.loads((d / "E1_lex_certify_telemetry.json")
                       .read_text(encoding="utf-8"))["aggregate"]
        pb.append(a["pricing_prefix_bound_runtime_s"] / 60.0)

    fig, ax = plt.subplots(figsize=(4.3, 2.7))
    xs = range(len(names))
    ax.bar(xs, total, width=0.55, color=BLUE_LIGHT,
           edgecolor="#8FAECB", linewidth=0.6, label="total runtime")
    ax.bar(xs, pb, width=0.55, color=ORANGE, label="prefix-bound compute")
    for x, v in zip(xs, total):
        ax.text(x, v + 9, f"{v:.0f}", ha="center", fontsize=8, color=MUTE)
    ax.set_xticks(list(xs), names)
    ax.set_ylabel("runtime (min)")
    ax.set_ylim(0, 375)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    soften(ax)
    panel(ax, "a", "Runtime by solver version (n=10)")
    save(fig, "fig_acceleration")


# ----------------------------------------------------------------- fig 3
def fig_frontier_certified():
    rows = list(csv.DictReader(open(
        RES / "experiments" / "E1_frontier_n8" / "E1_frontier.csv",
        encoding="utf-8-sig")))
    ks = sorted({int(r["K"]) for r in rows})
    bs = sorted({int(r["batteries"]) for r in rows})
    grid = {(int(r["K"]), int(r["batteries"])): int(r["coverage_incumbent"])
            for r in rows}

    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    mat = [[grid[(k, b)] for b in bs] for k in ks]
    im = ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "soft_blues", ["#F4F8FB", "#DCE9F4", "#B9CFE4", "#8FB4D6",
                       "#6B9BD1", "#4C7FAC", "#33638E"]), vmin=0, vmax=8,
        aspect="auto")
    for i in range(len(ks)):
        for j in range(len(bs)):
            v = mat[i][j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color="white" if v >= 5 else INK)
    ki, bj = ks.index(2), bs.index(7)
    ax.scatter([bj], [ki], marker="*", s=290, color="#F0C060",
               edgecolor=INK, linewidth=0.7, zorder=5)
    ax.set_xticks(range(len(bs)), [str(b) for b in bs])
    ax.set_yticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("battery packs B")
    ax.set_ylabel("fleet size K")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    panel(ax, "a", "Certified coverage (n=8)")
    cb = fig.colorbar(im, ax=ax, label="covered turbines", shrink=0.85)
    cb.outline.set_edgecolor("#CCCCCC")
    save(fig, "fig_frontier_certified")


# ----------------------------------------------------------------- fig 4
def fig_tiers_battery():
    tiers = [("S", RES / "experiments" / "E1_tierS"),
             ("M", RES / "experiments" / "E1_n10_v31"),
             ("L", RES / "experiments" / "E1_tierL")]
    tcov = [int(read_main(d)["coverage_incumbent"]) for _, d in tiers]

    batts, bcov = [], []
    for b, d in [(4, RES / "experiments" / "E1_b3_batt4"),
                 (5, RES / "experiments" / "E1_b3_batt5"),
                 (6, RES / "experiments" / "E1_b3_batt6"),
                 (7, RES / "experiments" / "E1_n10_v31")]:
        batts.append(b)
        bcov.append(int(read_main(d)["coverage_incumbent"]))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    xs = range(len(tiers))
    ax.bar(xs, tcov, width=0.48, color=BLUE)
    for x, v in zip(xs, tcov):
        ax.text(x, v + 0.2, str(v), ha="center", fontsize=8.5, color=MUTE)
    ax.axhline(10, color=MUTE, ls=(0, (4, 3)), lw=1)
    ax.text(-0.42, 10.2, "full coverage", ha="left", fontsize=7.5,
            color=MUTE)
    ax.set_xticks(list(xs), [f"{t}\n{b} Wh battery"
                             for (t, _), b in zip(tiers, (263, 474, 977))])
    ax.set_ylim(0, 11.3)
    ax.set_ylabel("certified coverage C*")
    soften(ax)
    panel(ax, "a", "Capability tiers (n=10, K=2, B=7)")

    ax = axes[1]
    ax.plot(batts, bcov, "o-", color=BLUE, lw=1.6, ms=5.5,
            markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.4)
    for b, v in zip(batts, bcov):
        ax.annotate(str(v), (b, v), textcoords="offset points",
                    xytext=(0, 7), fontsize=8, ha="center", color=MUTE)
    ax.set_xticks(batts)
    ax.set_xlabel("battery packs B (K=2, tier M)")
    ax.set_ylabel("certified coverage C*")
    ax.set_ylim(4.4, 8.7)
    soften(ax)
    panel(ax, "b", "Marginal value of one battery pack")
    save(fig, "fig_tiers_battery")


# ----------------------------------------------------------------- fig 5
def fig_quality_ladder():
    rows = list(csv.DictReader(open(
        RES / "algorithm_experiments" / "A1_accuracy" / "A1_accuracy.csv",
        encoding="utf-8-sig")))
    ns = ["6", "8", "10"]
    methods = [("greedy", "research_greedy", ORANGE),
               ("pool IP", "research_restricted_pool", GREEN),
               ("R-BPC", "exact_branch_price_cut", BLUE)]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    width = 0.24
    for mi, (label, key, color) in enumerate(methods):
        xs, vals = [], []
        for ni, n in enumerate(ns):
            r = next(x for x in rows
                     if x["n_turbines"] == n and x["method"] == key)
            gap = float(r["energy_gap_to_best_pct"] or 0.0)
            xs.append(ni + (mi - 1) * width)
            vals.append(max(gap, 1e-5))
            if gap > 0.01 and mi == 0:
                ax.annotate(f"{gap:.3f}%", (ni + 0.02, gap),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=7.5, color=ORANGE)
        ax.bar(xs, vals, width=width, color=color, label=label)
    ax.set_yscale("log")
    ax.set_ylim(6e-6, 1.2)
    ax.set_xticks(range(len(ns)), [f"n = {n}" for n in ns])
    ax.set_ylabel("energy gap to certified optimum (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    soften(ax)
    panel(ax, "a", "Solution quality")
    save(fig, "fig_quality_ladder")


# ----------------------------------------------------------------- fig 6
def fig_dtau():
    rows = list(csv.DictReader(open(
        RES / "algorithm_experiments" / "A2_speed" / "A2_speed.csv",
        encoding="utf-8-sig")))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    # n=10 points: Δτ=5 (main instance, E1_n10_v31) and Δτ=2.5 (E-3),
    # both certified C*=8 — coverage saturates while energy still improves.
    n10 = [("5", read_main(RES / "experiments" / "E1_n10_v31"),
            BLUE), ("2.5", read_main(RES / "experiments" / "E1_dtau25"),
                    BLUE)]
    for n, color, mk in [("6", ORANGE, "o"), ("8", BLUE, "s"),
                         ("10", GREEN, "^")]:
        if n == "10":
            xs = [2.5, 5.0]
            cov = [int(k[1]["coverage_incumbent"]) for k in n10]
            ax.plot(xs, cov, mk + "-", color=color, label="n = 10",
                    lw=1.6, ms=5.5, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=1.3)
            for x, v in zip(xs, cov):
                ax.annotate(str(v), (x, v), textcoords="offset points",
                            xytext=(0, 7), fontsize=7.5, ha="center",
                            color=MUTE)
            continue
        sub = sorted((r for r in rows if r["n_turbines"] == n),
                     key=lambda r: float(r["dtau_min"]))
        dt = [float(r["dtau_min"]) for r in sub]
        cov = [int(r["exact_coverage_incumbent"]) for r in sub]
        ax.plot(dt, cov, mk + "-", color=color, label=f"n = {n}",
                lw=1.6, ms=5, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.3)
        for x, v in zip(dt, cov):
            ax.annotate(str(v), (x, v), textcoords="offset points",
                        xytext=(0, 7), fontsize=7.5, ha="center", color=MUTE)
    ax.set_xlabel(r"launch-grid resolution $\Delta\tau$ (min)")
    ax.set_ylabel("certified coverage C*")
    ax.invert_xaxis()
    ax.set_ylim(4.3, 8.9)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    soften(ax)
    panel(ax, "a", "Coverage vs. launch-grid resolution")

    ax = axes[1]
    for n, color, mk in [("6", ORANGE, "o"), ("8", BLUE, "o")]:
        sub = sorted((r for r in rows if r["n_turbines"] == n),
                     key=lambda r: float(r["dtau_min"]))
        dt = [float(r["dtau_min"]) for r in sub]
        ratio = [float(r["runtime_ratio"]) for r in sub]
        ax.plot(dt, ratio, mk + "--", color=color, lw=1.4, ms=4.5,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.2, label=f"n = {n}")
    ax.axhline(1.0, color=MUTE, ls=":", lw=0.9)
    ax.text(15.2, 1.03, "parity", fontsize=7.5, color=MUTE, ha="left")
    ax.set_xlabel(r"$\Delta\tau$ (min)")
    ax.set_ylabel("pool-IP time / R-BPC time")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="center right", fontsize=7.5)
    soften(ax)
    panel(ax, "b", "Runtime ratio")
    save(fig, "fig_dtau")


# ----------------------------------------------------------------- fig 7
def fig_plan_structure():
    _ps = ROOT / "agent_analysis" / "derived" / "plan_structure_summary.json"
    if not _ps.is_file():
        _ps = ROOT / "results" / "diagnostics" / "plan_structure_summary.json"
    summary = json.loads(_ps.read_text(encoding="utf-8"))
    main = summary["n10_M_K2_B7_v31"]
    pairs = main["launch_tau_state_pairs"]
    h_hist = main["horizons_min_hist"]
    stops = main["stops_hist"]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    ax = axes[0]
    taus = [t for t, _ in pairs]
    ax.scatter(taus, [1] * len(taus), s=58, color=BLUE, zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("mission clock (min)")
    ax.set_xlim(-10, 370)
    ax.set_ylim(0.55, 1.45)
    soften(ax, ygrid=False)
    panel(ax, "a", "Launch timing")

    ax = axes[1]
    hs = sorted(h_hist.items())
    ax.bar([float(h) for h, _ in hs], [c for _, c in hs],
           width=2.2, color=BLUE)
    ax.set_xlabel("recovery horizon h (min)")
    ax.set_ylabel("sorties")
    ax.set_ylim(0, 6)
    soften(ax)
    panel(ax, "b", "Recovery horizon")

    ax = axes[2]
    ax.bar([int(k) for k in stops], list(stops.values()), width=0.4,
           color=BLUE)
    ax.set_xlabel("stops per sortie")
    ax.set_ylabel("sorties")
    ax.set_xticks([1, 2])
    ax.set_ylim(0, 7)
    soften(ax)
    panel(ax, "c", "Stops per sortie")
    save(fig, "fig_plan_structure")


def main() -> int:
    fig_scale_envelope()
    fig_acceleration()
    fig_frontier_certified()
    fig_tiers_battery()
    fig_quality_ladder()
    fig_dtau()
    fig_plan_structure()
    print(f"\nall figures (SVG) -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
