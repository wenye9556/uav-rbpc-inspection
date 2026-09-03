# -*- coding: utf-8 -*-
"""step17_paper_figure.py — 论文正式图统一入口(更新: 简化重绘, 第二轮审阅意见)。

与 step16_visualize.py 的分工不变: step16 = 快速核查(中文)+gif; 本文件 = 投稿图。

图清单(算法在前、模型实验在后; 弃选理由见 doc_experiments.md「论文图清单」节):
  figure1  算法双证据(A1 精度 + A2 速度 + 列经济性)
  figure2  E1 资源绑定地图(K×B 热图 三档)
  figure3  E1 架次物理(绑定约束散点 + 回收时距)
  figure4  E1 能力足迹地图(空间覆盖 三档)
  figure5  E2 担保闸门(头条: 安全–产出前沿)
  figure6  E2 计划 vs 回放存活(哑铃图)

更新 改动(依作者第二轮审阅四点意见, 逐条对应):
  ①只出 PNG(不再产 PDF); ②删除全部图角落 provenance 文字戳(不再在图上写口径);
  ③fig_algorithm 面板内散落的方法名标注("B&P (ours)/=pool MILP"、"extensive+Gurobi"
    等)一律移除, 改为图底部一个共享标准图例; ④UAV 档位一律只写 S/M/L, 不再标 Wh 容量
    (具体容量数值放论文正文/图注, 不占图内空间); ⑤figure3(原 E1_sortie_physics)瘦身:
    去掉对角线旁 "E binds/T binds" 说明文字与决策网格上限箭头注记, 两张子图的 S/M/L
    图例合并为一个图级共享图例, 坐标轴按数据范围收紧(不再强制方形留白), 刻度字号调小;
  ⑥文件更名 paper_figure.py → step17_paper_figure.py, 图片改名 figure1..figure6, 出图
    顺序改为"先算法(figure1)后模型实验(figure2~6)"(原 F1~F6 命名弃用, 见 doc_process.md
    更新 节)。

铁律不变: 只读 results/ 真实 CSV 零再求解; 同 CSV 多结果合同 拒绝出图; 膝点复用
step13.e1_select_from_df 本尊(失败回退本地等价并声明); 缺文件跳过该图不阻塞。

用法: python step17_paper_figure.py [--results-dir results] [--only figure1,figure5]
      [--dpi 300]
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import step13_experiment_model as S13

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

log = logging.getLogger("step17_paper_figure")
EXPECTED_RESULT_CONTRACT = S13.RESULT_CONTRACT

# ---------------------------------------------------------------- 统一视觉系统
ACCENT = "#0C5DA5"        # 本文方法(vp / B&P / 列生成) —— 全篇唯一饱和主色
DANGER = "#D1495B"        # 破约 / 闸门 / 损失 —— 红只表示"违反"
_V = plt.cm.viridis
TIER_COLOR = {"S": _V(0.22), "M": _V(0.52), "L": _V(0.80)}   # 与 figure2 热图同板
TIER_LABEL = {"S": "S", "M": "M", "L": "L"}                  # 更新: 只写字母, 不写 Wh
GRAY_DATA, GRAY_MID, GRAY_SOFT = "#4D5A66", "#7C8894", "#C3C9CF"
METHOD_ORDER = ["nominal", "gaussian", "SAA", "budget_G2", "box", "cantelli", "vp"]
METHOD_TEXT = {"nominal": "nominal", "gaussian": "Gaussian", "SAA": "SAA",
               "budget_G2": "budget $\\Gamma{=}2$", "box": "box",
               "cantelli": "Cantelli", "vp": "VP (ours)"}
METHOD_STYLE = {"nominal":   dict(color="#B9BEC4", marker="o", ls=":"),
                "gaussian":  dict(color="#95A0AB", marker="s", ls="-"),
                "SAA":       dict(color="#7C8894", marker="^", ls="-"),
                "budget_G2": dict(color="#65727E", marker="v", ls="-"),
                "box":       dict(color="#515E6A", marker="P", ls="-"),
                "cantelli":  dict(color="#3E4B57", marker="D", ls="-"),
                "vp":        dict(color=ACCENT,    marker="*", ls="-")}


def _style():
    plt.rcParams.update({
        "font.size": 8.2, "axes.titlesize": 9.2, "axes.labelsize": 8.4,
        "legend.fontsize": 7.4, "legend.title_fontsize": 7.8,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,   # 更新: 刻度数字调小
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.titlepad": 5.0, "legend.frameon": False,
        "figure.constrained_layout.use": True, "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------- 读数与口径
def _read(path: Path, name: str) -> pd.DataFrame | None:
    if not path.is_file():
        log.warning("缺 %s (%s) —— 跳过依赖它的图", name, path)
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "result_contract" not in df.columns:
        raise SystemExit(f"{name}: 缺少 result_contract，拒绝读取旧/未知口径 CSV。")
    contracts = set(df["result_contract"].dropna().astype(str))
    if contracts != {EXPECTED_RESULT_CONTRACT}:
        raise SystemExit(f"{name}: result_contract 必须为 {EXPECTED_RESULT_CONTRACT}，得到 {sorted(contracts)}。")
    return df


def _as_bool(s: pd.Series) -> pd.Series:
    """宽松布尔解析，仅供非正式展示字段。未知值按 False 处理。"""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().eq("true")


def _strict_bool(s: pd.Series, name: str) -> pd.Series:
    """正式证据字段的 fail-closed 布尔解析。

    只接受 bool、0/1 和常见 true/false 文本；缺失或未知值直接拒绝，
    防止旧 CSV、损坏字段或空值被静默解释为“安全”。
    """
    if s.isna().any():
        raise SystemExit(f"{name}: 包含缺失布尔值，拒绝生成正式图。")
    out = []
    for v in s.tolist():
        if isinstance(v, (bool, np.bool_)):
            out.append(bool(v)); continue
        if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
            out.append(bool(int(v))); continue
        if isinstance(v, (float, np.floating)) and float(v) in (0.0, 1.0):
            out.append(bool(int(v))); continue
        t = str(v).strip().lower()
        if t in {"true", "1", "yes"}:
            out.append(True)
        elif t in {"false", "0", "no"}:
            out.append(False)
        else:
            raise SystemExit(f"{name}: 非法布尔值 {v!r}，拒绝生成正式图。")
    return pd.Series(out, index=s.index, dtype=bool)



def _gap_summary(df: pd.DataFrame | None, prefix: str = "") -> str | None:
    """Compact, scope-aware solver certificate note for figure footers."""
    if df is None or df.empty:
        return None
    def col(name):
        return f"{prefix}{name}" if f"{prefix}{name}" in df.columns else name
    pieces = []
    cg = col("coverage_gap_abs")
    if cg in df.columns:
        vals = pd.to_numeric(df[cg], errors="coerce").dropna()
        if len(vals):
            pieces.append(f"global discrete coverage gap: {vals.min():g}–{vals.max():g}")
    eg = col("energy_gap_pct")
    if eg in df.columns:
        vals = pd.to_numeric(df[eg], errors="coerce").dropna()
        if len(vals):
            pieces.append(f"global energy gap: {vals.min():.3g}–{vals.max():.3g}%")
    cond = col("conditional_energy_gap_pct")
    if cond in df.columns:
        vals = pd.to_numeric(df[cond], errors="coerce").dropna()
        if len(vals) and not (eg in df.columns and pd.to_numeric(df[eg], errors="coerce").notna().any()):
            pieces.append(f"conditional energy gap: {vals.min():.3g}–{vals.max():.3g}%")
    rg = col("restricted_pool_gap_pct")
    if rg in df.columns:
        vals = pd.to_numeric(df[rg], errors="coerce").dropna()
        if len(vals):
            pieces.append(f"restricted-pool gap: {vals.min():.3g}–{vals.max():.3g}%")
    reason = col("global_energy_gap_reason")
    if reason in df.columns:
        vals = sorted({str(v) for v in df[reason].dropna() if str(v).strip()})
        if vals and not (eg in df.columns and pd.to_numeric(df[eg], errors="coerce").notna().any()):
            pieces.append("global energy gap undefined: " + "; ".join(vals[:2]))
    scope = col("bound_scope")
    if scope in df.columns:
        vals = sorted({str(v) for v in df[scope].dropna() if str(v).strip()})
        if vals:
            pieces.append("scope=" + "/".join(vals[:2]))
    return " | ".join(pieces) if pieces else None


def _add_gap_footer(fig, *notes):
    text = " ; ".join(n for n in notes if n)
    if text:
        fig.text(0.5, 0.002, text, ha="center", va="bottom", fontsize=6.1,
                 color=GRAY_MID, wrap=True)

def _save(fig, out_dir: Path, name: str, dpi: int) -> Path:
    """更新: 只出 PNG(不再产 PDF)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    fig.savefig(p, dpi=dpi)
    log.info("写 %s", p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- 膝点(同源)
def _knee_table(e1: pd.DataFrame) -> pd.DataFrame | None:
    """膝点必须复用 step13 的唯一正式选择器，不允许图层自行降级。"""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import step13_experiment_model as S13                    # noqa: E402
    except Exception as e:                                       # pragma: no cover
        raise SystemExit(
            f"figure2: 无法导入 step13 的正式 E1 选择器 "
            f"({type(e).__name__}: {e})；拒绝使用本地近似膝点。"
        ) from e
    return S13.e1_select_from_df(e1)


# ================================================================ figure1 算法双证据(先算法)
def figure1_algorithm(a1, a2, out_dir, dpi):
    ncol = int(a1 is not None) + 2 * int(a2 is not None)
    if ncol == 0:
        return None
    fig, axes = plt.subplots(1, ncol, figsize=(7.2, 2.75))
    axes = list(np.atleast_1d(axes))
    fig_handles = []

    if a1 is not None:
        ax = axes.pop(0)
        g = a1[a1["method"].isin(["research_greedy", "greedy"])].sort_values("n_turbines")
        pm = a1[a1["method"].isin(["research_restricted_pool", "restricted_exact", "pool_milp"])].sort_values("n_turbines")
        bp = a1[a1["method"].isin(["exact_branch_price_cut", "global_exact", "bp"])].sort_values("n_turbines")
        ax.plot(g["n_turbines"], g["covered"], ":", marker="s", color=GRAY_MID,
                ms=4.5, lw=1.2, zorder=3)
        ax.plot(bp["n_turbines"], bp["covered"], "-", marker="*", color=ACCENT,
                ms=10, lw=1.5, zorder=5)
        ax.plot(pm["n_turbines"], pm["covered"], "o", mfc="none", mec=GRAY_DATA,
                ms=8, mew=1.1, ls="none", zorder=4)
        for _, r in g.iterrows():          # 数值结果标注(非方法名文字), 保留
            if r["gap_to_best"] > 0:
                ax.annotate(f"$-${int(r['gap_to_best'])}",
                            (r["n_turbines"], r["covered"]),
                            textcoords="offset points", xytext=(-2, -11),
                            fontsize=6.8, color=GRAY_MID)
        ax.set_xticks(sorted(a1["n_turbines"].unique()))
        ax.set_xlim(g["n_turbines"].min() - 4, g["n_turbines"].max() + 4)
        ax.set_xlabel("instance size $n$")
        ax.set_ylabel("turbines covered")
        ax.set_title("(a) accuracy")
        fig_handles += [
            Line2D([], [], color=GRAY_MID, ls=":", marker="s", ms=5, label="greedy"),
            Line2D([], [], color=GRAY_DATA, ls="none", marker="o", mfc="none",
                   mec=GRAY_DATA, ms=7, mew=1.1, label="restricted-pool research baseline"),
            Line2D([], [], color=ACCENT, ls="-", marker="*", ms=9,
                   label="exact branch-price-and-cut")]

    if a2 is not None:
        s = a2.sort_values("ext_pool")
        ax = axes.pop(0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot(s["ext_pool"], s["t_gurobi_total_s"], "o-", color=GRAY_DATA,
                ms=4.5, lw=1.3, zorder=3)
        ax.plot(s["ext_pool"], s["t_ours_s"], "*", color=ACCENT, ms=10,
                ls="none", zorder=4)
        for _, r in s.iterrows():
            # runtime_ratio = T_extensive / T_BP。大于 1 才是加速，小于 1 必须标为减速。
            ratio = r.get("runtime_ratio", r.get("speedup"))
            if pd.notna(ratio):
                ratio = float(ratio)
                if ratio > 1.0:
                    label = f"{ratio:.1f}× faster"
                elif ratio > 0:
                    label = f"{1.0 / ratio:.1f}× slower"
                else:
                    label = "n/a"
                ax.annotate(label, (r["ext_pool"], r["t_ours_s"]),
                            textcoords="offset points", xytext=(0, -13), ha="center",
                            fontsize=6.2, color=ACCENT)
        ax.set_xlabel("full column space $|\\Omega|$")
        ax.set_ylabel("wall-clock (s)")
        ax.set_title("(b) runtime")
        fig_handles.append(
            Line2D([], [], color=GRAY_DATA, ls="-", marker="o", ms=5,
                   label="extensive + Gurobi"))

        ax = axes.pop(0)
        frac = 100.0 * s["ours_pool"] / s["ext_pool"]
        lbl = [f"n{int(r.n_turbines)}\n$\\Delta\\tau${r.dtau_min:g}"
               for r in s.itertuples()]
        ax.bar(np.arange(len(s)), frac, color=ACCENT, alpha=0.88, width=0.66)
        for i, f in enumerate(frac):
            ax.text(i, f + 0.35, f"{f:.0f}%", ha="center", fontsize=6.8,
                    color=GRAY_DATA)
        ax.set_xticks(np.arange(len(s)), lbl, fontsize=6.2)
        ax.set_ylabel("generated / $|\\Omega|$ (%)")
        ax.set_title("(c) column economy")
        ax.set_ylim(0, frac.max() * 1.2)
        ax.grid(axis="x", alpha=0)

    fig.legend(handles=fig_handles, loc="outside lower center",
              ncols=len(fig_handles))
    exact_a1 = (None if a1 is None else
                a1[a1["method"].isin(["exact_branch_price_cut", "global_exact", "bp"])])
    _add_gap_footer(fig, _gap_summary(exact_a1), _gap_summary(a2, prefix="exact_"))
    return _save(fig, out_dir, "figure1", dpi)


# ================================================================ figure2 资源绑定地图
def figure2_E1_resource_map(e1, out_dir, dpi):
    uavs = [u for u in ("S", "M", "L") if u in set(e1["uav"])]
    sel = _knee_table(e1)
    vmax = float(e1["safe_served"].max())
    n_reach = int(e1["n_reach"].iloc[0]) if "n_reach" in e1.columns else 90

    fig, axes = plt.subplots(1, len(uavs), figsize=(7.2, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, uk in zip(axes, uavs):
        sub = e1[e1["uav"] == uk]
        piv = (sub.pivot_table(index="K", columns="batteries", values="safe_served")
               .sort_index().sort_index(axis=1))
        Bs, Ks = piv.columns.to_numpy(), piv.index.to_numpy()
        im = ax.pcolormesh(np.append(Bs - 0.5, Bs[-1] + 0.5),
                           np.append(Ks - 0.5, Ks[-1] + 0.5),
                           piv.to_numpy(), cmap="viridis", vmin=0, vmax=vmax,
                           rasterized=True)
        bad = sub[(sub["batteries"] > 0) & ~_strict_bool(sub["plan_holds"], "figure2.plan_holds")]
        if len(bad):
            ax.plot(bad["batteries"], bad["K"], "x", color=DANGER, ms=4.5, mew=1.3)
        kstar = [int(Ks[np.argmax(piv[b].to_numpy() >= piv[b].max())]) for b in Bs]
        ax.step(Bs, kstar, where="mid", color="white", lw=1.8, zorder=4)
        ax.step(Bs, kstar, where="mid", color="black", lw=0.5, zorder=5)
        if sel is not None and uk in set(sel["uav"]):
            r = sel[sel["uav"] == uk].iloc[0]
            if pd.notna(r["knee_B"]):
                ax.plot([r["knee_B"]], [r["knee_K"]], "*", ms=15, mfc="white",
                        mec="black", mew=1.1, zorder=6)
        cov = sub["coverable_note"].iloc[0]
        ax.set_title(f"{TIER_LABEL[uk]}: max {int(sub['safe_served'].max())}"
                     f"/{n_reach} · coverable {cov}", fontsize=8.6)
        ax.set_xlabel("batteries $B$")
        ax.set_xticks(Bs[::4]); ax.set_yticks(Ks)
        ax.grid(False)
    axes[0].set_ylabel("fleet size $K$")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.012, aspect=22)
    cb.set_label("reliably served turbines")
    fig.legend(handles=[
        Line2D([], [], color=GRAY_DATA, lw=1.8, label="$K$-binding frontier"),
        Line2D([], [], ls="none", marker="*", ms=11, mfc="white", mec="black",
               label="knee $(K,B)\\rightarrow$ E2/A")],
        loc="outside lower center", ncols=2)
    return _save(fig, out_dir, "figure2", dpi)


# ================================================================ figure3 架次物理(瘦身)
def figure3_E1_sortie_physics(details, out_dir, dpi):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) 逐架次相对裕度: 对角线上方=能量绑定, 下方=时间绑定(不再写文字, 靠图例+颜色)
    xmax = max(d["rel_margin_E"].max() for d in details.values()) * 100 * 1.08
    ymax = max(d["rel_margin_T"].max() for d in details.values()) * 100 * 1.18
    for uk, d in details.items():
        x, y = d["rel_margin_E"] * 100, d["rel_margin_T"] * 100
        eng = d["binding"].astype(str).eq("energy")
        a.scatter(x[~eng], y[~eng], s=24, color=TIER_COLOR[uk], alpha=0.9,
                  marker="o", edgecolors="white", linewidths=0.4, zorder=3)
        a.scatter(x[eng], y[eng], s=32, color=TIER_COLOR[uk], marker="D",
                  edgecolors="black", linewidths=0.6, zorder=4)
    lim = max(xmax, ymax)
    a.plot([0, lim], [0, lim], ls="--", lw=0.8, color=GRAY_MID, zorder=2)
    a.set_xlim(0, xmax); a.set_ylim(0, ymax)
    a.set_xlabel("energy margin per sortie (%)")
    a.set_ylabel("time margin (%)")
    a.set_title("(a) binding constraint")

    # (b) 选定 h* vs 停靠数(紧轴距; 网格上限不再进图, 移入图注/正文)
    off = {"S": -0.15, "M": 0.0, "L": 0.15}
    rng = np.random.default_rng(0)
    for uk, d in details.items():
        jx = d["stops"] + off[uk] + rng.uniform(-0.05, 0.05, len(d))
        b.scatter(jx, d["h_min"], s=28, color=TIER_COLOR[uk], alpha=0.9,
                  edgecolors="white", linewidths=0.4, zorder=3)
    hs_all = [d["h_min"] for d in details.values()]
    lo, hi = min(h.min() for h in hs_all), max(h.max() for h in hs_all)
    pad = max((hi - lo) * 0.12, 2.0)
    b.set_ylim(lo - pad, hi + pad)
    b.set_xticks(sorted({int(s) for d in details.values() for s in d["stops"]}))
    b.set_xlabel("turbines per sortie (stops)")
    b.set_ylabel("selected horizon $h^{\\star}$ (min)")
    b.set_title("(b) recovery horizon")

    # 共享图例(合并两张子图的 S/M/L + energy-binding, 图底部一次性给出)
    fig.legend(handles=[
        *[Line2D([], [], ls="none", marker="o", color=TIER_COLOR[u],
                 label=TIER_LABEL[u]) for u in details],
        Line2D([], [], ls="none", marker="D", mfc="none", mec="black",
               label="energy-binding")],
        loc="outside lower center", ncols=len(details) + 1)
    return _save(fig, out_dir, "figure3", dpi)


# ================================================================ figure4 能力足迹地图
def figure4_capability_footprint(details, turbines_csv, out_dir, dpi):
    if not turbines_csv.is_file():
        log.warning("缺风机坐标 %s —— 跳过 figure4", turbines_csv)
        return None
    tb = pd.read_csv(turbines_csv)
    lat0, lon0 = tb["lat"].mean(), tb["lon"].mean()
    x = (tb["lon"] - lon0) * 111_320 * np.cos(np.radians(lat0)) / 1000.0
    y = (tb["lat"] - lat0) * 110_540 / 1000.0
    served = {u: set(t.strip() for s in d["turbines"] for t in str(s).split(";"))
              for u, d in details.items()}

    fig, axes = plt.subplots(1, len(details), figsize=(7.2, 2.15),
                             sharex=True, sharey=True)
    for ax, uk in zip(np.atleast_1d(axes), details):
        hit = tb["turbine_id"].isin(served[uk])
        ax.scatter(x[~hit], y[~hit], s=11, color="#DBDEE1", edgecolors="none",
                   zorder=2)
        ax.scatter(x[hit], y[hit], s=20, color=TIER_COLOR[uk],
                   edgecolors="white", linewidths=0.35, zorder=3)
        ax.set_title(f"{TIER_LABEL[uk]}: {int(hit.sum())}/{len(tb)}", fontsize=8.6)
        ax.set_aspect("equal")
        ax.set_xlabel("east (km)")
        ax.grid(alpha=0.18)
        ax.margins(0.05)
    np.atleast_1d(axes)[0].set_ylabel("north (km)")
    return _save(fig, out_dir, "figure4", dpi)


# ================================================================ figure5 担保闸门(头条)
def figure5_E2_guarantee_gate(e2, out_dir, dpi):
    if "run_status" in e2.columns:
        e2 = e2[e2["run_status"].astype(str).str.lower().isin(["ok", "completed"])].copy()
    if not len(e2):
        return None
    required = {"component_eps", "mission_eps_budget", "emp_viol_upper95",
                "formal_reliability_claim_eligible", "evidence_scope"}
    missing = sorted(required - set(e2.columns))
    if missing:
        raise SystemExit(f"figure5: 缺正式担保字段 {missing}。")
    eligible = _strict_bool(e2["formal_reliability_claim_eligible"],
                            "figure5.formal_reliability_claim_eligible")
    e2 = e2[eligible & e2["emp_viol_upper95"].notna()
            & e2["evidence_scope"].astype(str).isin(["confirmatory-purged-disjoint-real-joint-holdout",
                   "confirmatory-purged-disjoint-real-joint-holdout-with-terminal-sensor-error-out-of-scope"])].copy()
    if not len(e2):
        raise SystemExit("figure5: 没有具备冻结、purged、disjoint 真实联合留出资格的行；拒绝用点估计替代正式上界。")
    component_eps = float(e2["component_eps"].dropna().iloc[0])
    gate = float(e2["mission_eps_budget"].dropna().iloc[0])
    e2["_risk_upper95"] = pd.to_numeric(e2["emp_viol_upper95"], errors="raise")
    qs = sorted(e2["q"].unique())
    pos = e2.loc[e2["_risk_upper95"] > 0, "_risk_upper95"]
    floor = (max(min(float(pos.min()) * 0.5, 1e-3), 2e-4) if len(pos) else 2e-4)
    msz = {q: float(26 + 74 * i / max(len(qs) - 1, 1)) for i, q in enumerate(qs)}

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xscale("log")
    ax.axvspan(gate, 1.0, color=DANGER, alpha=0.06, zorder=0)
    ax.axvline(gate, color=DANGER, ls="--", lw=1.2, zorder=2)
    ax.text(gate * 1.1, 0.03, "$2\\varepsilon$ gate",
            transform=ax.get_xaxis_transform(), fontsize=7.4, color=DANGER)
    ax.text(0.985, 0.97, "union-budget threshold exceeded", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.6, color=DANGER)

    meth_handles = []
    for c in METHOD_ORDER:
        s = e2[e2["criterion"] == c].sort_values("q")
        if not len(s):
            continue
        st, main = METHOD_STYLE[c], c == "vp"
        x = s["_risk_upper95"].clip(lower=floor).to_numpy()
        y = s["safe_served"].to_numpy()
        ax.plot(x, y, st["ls"], color=st["color"], lw=2.4 if main else 1.0,
                alpha=1.0 if main else 0.9, zorder=5 if main else 3)
        for q, xi_, yi in zip(s["q"], x, y):
            ax.scatter([xi_], [yi], s=msz[q] * (1.7 if main else 1.0),
                       color=st["color"], marker=st["marker"],
                       edgecolors="white", linewidths=0.5,
                       zorder=6 if main else 4)
        meth_handles.append(Line2D([], [], color=st["color"], ls=st["ls"],
                                   lw=2.2 if main else 1.0, marker=st["marker"],
                                   ms=9 if main else 4.5, label=METHOD_TEXT[c]))
        if main:
            ax.annotate(f"{100 * s['_risk_upper95'].iloc[-1]:.1f}%",
                        (x[-1], y[-1]), textcoords="offset points",
                        xytext=(5, -11), fontsize=7.2, color=ACCENT,
                        fontweight="bold")
    if (e2["_risk_upper95"] <= 0).any():
        ax.annotate("0", (floor, e2.loc[e2["_risk_upper95"] <= 0, "safe_served"].max()),
                    textcoords="offset points", xytext=(-9, -2), fontsize=6.8,
                    color=GRAY_MID)
    size_handles = [Line2D([], [], ls="none", marker="o", color=GRAY_MID,
                           ms=np.sqrt(msz[q]),
                           label=f"$q={q:g}$ · $H_s$ "
                                 f"{e2[np.isclose(e2['q'], q)]['Hs0'].iloc[0]:.2f} m")
                    for q in qs]
    leg1 = fig.legend(handles=meth_handles, loc="outside right upper",
                      title="criterion")
    fig.legend(handles=size_handles, loc="outside right lower",
               title="sea-state window")
    fig.add_artist(leg1)
    ax.set_xlim(floor * 0.6, 1.0)
    ax.set_ylim(0, e2["safe_served"].max() * 1.10)
    ax.set_xlabel("simultaneous one-sided 95% failure upper bound")
    ax.set_ylabel("reliably served turbines")
    ax.set_title("Safety–throughput across sea states")
    _add_gap_footer(fig, _gap_summary(e2))
    return _save(fig, out_dir, "figure5", dpi)


# ================================================================ figure6 计划 vs 存活
def figure6_E2_plan_vs_reality(e2, out_dir, dpi):
    if "run_status" in e2.columns:
        e2 = e2[e2["run_status"].astype(str).str.lower().isin(["ok", "completed"])].copy()
    if not len(e2):
        return None
    qs = sorted(e2["q"].unique())
    q_show = [qs[len(qs) // 2], qs[-1]]
    order = [c for c in METHOD_ORDER if c in set(e2["criterion"])]
    ypos = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(1, len(q_show), figsize=(7.2, 2.9), sharey=True)
    for ax, q in zip(np.atleast_1d(axes), q_show):
        w = e2[np.isclose(e2["q"], q)]
        for c, y in zip(order, ypos):
            r = w[w["criterion"] == c]
            if not len(r):
                continue
            r = r.iloc[0]
            ok = bool(_strict_bool(pd.Series([r["holds"]]), "figure6.holds").iloc[0])
            dot = ACCENT if c == "vp" else GRAY_DATA
            lost = int(r["covered"] - r["safe_served"])
            if lost > 0:
                ax.plot([r["safe_served"], r["covered"]], [y, y], "-",
                        color=(DANGER if not ok else GRAY_SOFT),
                        lw=2.4, alpha=0.9, zorder=2)
                ax.annotate(f"$-${lost}",
                            ((r["covered"] + r["safe_served"]) / 2, y + 0.18),
                            ha="center", fontsize=6.6,
                            color=(DANGER if not ok else GRAY_MID))
            ax.plot([r["covered"]], [y], "o", mfc="white", mec=dot, ms=7,
                    mew=1.5, zorder=3)
            ax.plot([r["safe_served"]], [y], "o", color=dot, ms=7, zorder=4)
        ax.set_yticks(ypos, [METHOD_TEXT[c] for c in order])
        ax.set_xlabel("turbines")
        ax.set_xlim(-2, e2["covered"].max() * 1.07)
        ax.set_title(f"$q={q:g}$ · $H_s$ {w['Hs0'].iloc[0]:.2f} m")
        ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
    fig.legend(handles=[
        Line2D([], [], ls="none", marker="o", mfc="white", mec=GRAY_DATA,
               ms=7, mew=1.5, label="planned"),
        Line2D([], [], ls="none", marker="o", color=GRAY_DATA, ms=7,
               label="survives"),
        Line2D([], [], color=DANGER, lw=2.4, label="loss (broken)"),
        Line2D([], [], color=GRAY_SOFT, lw=2.4, label="loss (holds)")],
        loc="outside lower center", ncols=4, columnspacing=1.1)
    _add_gap_footer(fig, _gap_summary(e2))
    return _save(fig, out_dir, "figure6", dpi)


# ---------------------------------------------------------------- 主入口
FIGS = ["figure1", "figure2", "figure3", "figure4", "figure5", "figure6"]


def main():
    ap = argparse.ArgumentParser(description="论文正式图统一入口(读真实结果 CSV, 不再求解)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="默认 <results>/figures/paper")
    ap.add_argument("--turbines-csv", type=Path, default=None,
                    help="默认 <本文件目录>/data/turbines_Rodsand_II_clean.csv")
    ap.add_argument("--only", type=str, default="",
                    help=f"只出指定图, 逗号分隔 {','.join(FIGS)}")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _style()
    root = args.results_dir
    out = args.out_dir or (root / "figures" / "paper")
    want = [f.strip().lower() for f in args.only.split(",") if f.strip()] or FIGS

    e1 = _read(root / "model_experiments/E1_frontier/E1_frontier.csv", "E1_frontier")
    e2 = _read(root / "model_experiments/E2_robust_comparison/E2_robust_raw.csv", "E2_raw")
    a1 = _read(root / "algorithm_experiments/A1_accuracy/A1_accuracy.csv", "A1")
    a2 = _read(root / "algorithm_experiments/A2_speed/A2_speed.csv", "A2")
    details = {}
    if e1 is not None:
        for uk in ("S", "M", "L"):
            d = _read(root / f"model_experiments/E1_frontier/E1_detail_Kmax_{uk}.csv",
                      f"E1_detail_{uk}")
            if d is not None:
                details[uk] = d

    done = []
    if "figure1" in want and (a1 is not None or a2 is not None):
        done.append(figure1_algorithm(a1, a2, out, args.dpi))
    if "figure2" in want and e1 is not None:
        done.append(figure2_E1_resource_map(e1, out, args.dpi))
    if "figure3" in want and details:
        done.append(figure3_E1_sortie_physics(details, out, args.dpi))
    if "figure4" in want and details:
        tcsv = args.turbines_csv or (Path(__file__).resolve().parent
                                     / "data/turbines_Rodsand_II_clean.csv")
        done.append(figure4_capability_footprint(details, tcsv, out, args.dpi))
    if "figure5" in want and e2 is not None:
        done.append(figure5_E2_guarantee_gate(e2, out, args.dpi))
    if "figure6" in want and e2 is not None:
        done.append(figure6_E2_plan_vs_reality(e2, out, args.dpi))

    done = [d for d in done if d is not None]
    if not done:
        raise SystemExit("没有任何图被生成 —— 检查 --results-dir 与 --only。")
    log.info("完成: %d 个文件 → %s", len(done), out)


if __name__ == "__main__":
    main()
