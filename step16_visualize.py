#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step16_visualize.py — 出图统一入口(更新 三合一: 论文静态图 + 任务 GIF)。

子命令:
  plots   论文静态图(更新 新增): 读 E1_frontier.csv / E2_robust_raw.csv →
          fig_E1_frontier.png(三档 safe–B 曲线族, ★=E1_select 膝点, 空心点=plan_holds=False 格)
          fig_E2_frontier.png(常规窗安全–产出前沿 + q=0.8 恶劣窗对照)
  gif     任务动画: 船·无人机·换电·巡检全细节时间轴 GIF,
          按真实 AIS + E1_detail_Kmax*.csv 驱动; --probe 先体检航迹/进场时段。

用法(作者机器):
  python step16_visualize.py plots \
      --e1 results/model_experiments/E1_frontier/E1_frontier.csv \
      --e2 results/model_experiments/E2_robust_comparison/E2_robust_raw.csv \
      --out-dir results/figs
  python step16_visualize.py gif --track-mmsi 219018788 \
      --detail results/model_experiments/E1_frontier/E1_detail_Kmax_L.csv --out uav_mission.gif

依赖: numpy pandas matplotlib pillow(仅 gif)。plots 不触数据/求解, 只读结果 CSV。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import step9_model as M
import step13_experiment_model as S13

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("step16")

# Stable UAV-lane colors shared by every GIF frame.  Keep this module-local so
# the animation entry point has no hidden dependency on a paper-figure module.
PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)



def _require_result_contract(df: pd.DataFrame, name: str, required: set[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} 缺少当前图形契约字段: {missing}")
    contracts = set(df["result_contract"].dropna().astype(str))
    if contracts != {S13.RESULT_CONTRACT}:
        raise ValueError(f"{name} result_contract 必须为 {S13.RESULT_CONTRACT}，得到 {sorted(contracts)}")


def _bool_series_strict(s: pd.Series, name: str) -> pd.Series:
    if s.isna().any():
        raise ValueError(f"{name} 含缺失值，拒绝把未知状态补成通过。")
    if s.dtype == bool:
        return s
    vals = s.astype(str).str.strip().str.lower()
    if not vals.isin({"true", "false"}).all():
        raise ValueError(f"{name} 含非布尔值。")
    return vals.eq("true")

def _solver_gap_note(df: pd.DataFrame) -> str | None:
    """Scope-aware footer text; never substitutes a restricted-pool gap for a global gap."""
    parts = []
    if "coverage_gap_abs" in df.columns:
        x = pd.to_numeric(df["coverage_gap_abs"], errors="coerce").dropna()
        if len(x):
            parts.append(f"全局离散覆盖Gap={x.min():g}–{x.max():g}")
    if "energy_gap_pct" in df.columns:
        x = pd.to_numeric(df["energy_gap_pct"], errors="coerce").dropna()
        if len(x):
            parts.append(f"全局能耗Gap={x.min():.3g}–{x.max():.3g}%")
    if "conditional_energy_gap_pct" in df.columns:
        x = pd.to_numeric(df["conditional_energy_gap_pct"], errors="coerce").dropna()
        if len(x) and not ("energy_gap_pct" in df.columns and
                           pd.to_numeric(df["energy_gap_pct"], errors="coerce").notna().any()):
            parts.append(f"条件能耗Gap={x.min():.3g}–{x.max():.3g}%（非全局词典序Gap）")
    if "restricted_pool_gap_pct" in df.columns:
        x = pd.to_numeric(df["restricted_pool_gap_pct"], errors="coerce").dropna()
        if len(x):
            parts.append(f"受限列池Gap={x.min():.3g}–{x.max():.3g}%")
    if "global_energy_gap_reason" in df.columns:
        reasons = sorted({str(v) for v in df["global_energy_gap_reason"].dropna()
                          if str(v).strip()})
        if reasons and not ("energy_gap_pct" in df.columns and
                            pd.to_numeric(df["energy_gap_pct"], errors="coerce").notna().any()):
            parts.append("全局能耗Gap未定义：" + "; ".join(reasons[:2]))
    if "bound_scope" in df.columns:
        scopes = sorted({str(v) for v in df["bound_scope"].dropna() if str(v).strip()})
        if scopes:
            parts.append("界范围=" + "/".join(scopes[:2]))
    return " | ".join(parts) if parts else None


def _add_solver_gap_footer(fig, df: pd.DataFrame) -> None:
    note = _solver_gap_note(df)
    if note:
        fig.text(0.5, 0.002, note, ha="center", va="bottom", fontsize=7,
                 color="dimgray", wrap=True)


def _setup_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "PingFang SC"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    log.warning("未找到中文字体, 图内标签改用英文。")
    return False


def _get_track(args, turbines, lat0lon0, track_csv_auto):
    lat0, lon0 = lat0lon0
    cand = None
    if args.track_mmsi:
        cand = Path(__file__).resolve().parent / "tracks" / f"track_{args.track_mmsi}.csv"
    elif args.track_csv:
        cand = Path(args.track_csv)
    cands = [cand] if cand else (track_csv_auto if isinstance(track_csv_auto, (list, tuple))
                                 else ([track_csv_auto] if track_csv_auto else []))
    best = None
    for c in cands:
        tr = S13._load_track_ds(c, lat0, lon0)
        if tr is None:
            continue
        dwell = sum(d for *_, d in S13._infarm_segments(tr, turbines, args.pair_radius))
        if best is None or dwell > best[0]:
            best = (dwell, c, tr)
    if best is not None:
        _, c, tr = best
        segs = S13._infarm_segments(tr, turbines, args.pair_radius)
        if args.track_start_min is not None:
            t0 = args.track_start_min * 60.0
        elif segs:
            want = min(args.window_min * 60.0, max(d for *_, d in segs))
            t0 = next(a for a, _, d in segs if d >= want - 1e-6)
        else:
            t0 = 0.0
        i0 = max(int(np.searchsorted(tr.t, t0)) - 1, 0)
        i1 = int(np.searchsorted(tr.t, t0 + args.window_min * 60.0 + 3600.0))
        tr = M.ShipTrack(tr.t[i0:i1 + 1] - tr.t[i0], tr.P[i0:i1 + 1])
        return tr, f"AIS: {Path(c).name} (window@{t0/60:.0f}min)", False
    tr = S13._synth_transit_track(turbines, speed_mps=3.0)
    if tr.duration_sec() < args.window_min * 60.0:
        slow = max(3.0 * tr.duration_sec() / (args.window_min * 60.0), 0.05)
        tr = S13._synth_transit_track(turbines, speed_mps=slow)
    return tr, "SYNTHETIC TRACK", True


def _lanes(flights):
    """Use optimized UAV identities when available; otherwise fall back to display coloring."""
    if flights and all(f.get("uav_id") is not None for f in flights):
        for f in flights:
            f["lane"] = int(f["uav_id"])
        return 1 + max(f["lane"] for f in flights)
    free = []
    for f in sorted(flights, key=lambda f: f["tau"]):
        got = None
        service_end = float(f.get("service_end", f["end"]))
        for i, (avail, lane) in enumerate(free):
            if avail <= f["tau"] + 1e-9:
                got = lane; free[i] = (service_end, lane); break
        if got is None:
            got = len(free); free.append((service_end, got))
        f["lane"] = got
    return 1 + max(f["lane"] for f in flights)


def _flight_schedule(row, track, tpos):
    tau = float(row["tau_min"]) * 60.0; h = float(row["h_min"]) * 60.0
    tids = [x for x in str(row["turbines"]).split(";") if x]
    raw = row.get("schedule_json", None)
    if raw is None or pd.isna(raw) or not str(raw).strip():
        raise ValueError("任务动画要求 schedule_json；禁止用固定巡航速度重建优化路线。")
    try:
        sched = json.loads(str(raw))
    except Exception as exc:
        raise ValueError("schedule_json 不是有效JSON") from exc
    if sched.get("result_contract") != "route-nominal-schedule":
        raise ValueError(f"未知任务时间轴合同: {sched.get('result_contract')!r}")
    wp = [(tau + float(w["t_s"]), np.array([float(w["x_m"]), float(w["y_m"])], float))
          for w in sched.get("waypoints", [])]
    if len(wp) < 2 or abs(wp[-1][0] - (tau + h)) > 1e-6:
        raise ValueError("schedule_json 时间轴不完整或与 h_min 不一致")
    hold = [(str(v["turbine_id"]), tau + float(v["start_s"]),
             tau + float(v["end_s"])) for v in sched.get("inspections", [])]
    path = np.asarray(sched.get("path_m", []), float)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2:
        raise ValueError("schedule_json 缺少有效 path_m")
    ps0 = row.get("post_service_start_min", np.nan)
    ps1 = row.get("post_service_end_min", np.nan)
    ps0 = None if pd.isna(ps0) else float(ps0) * 60.0
    ps1 = None if pd.isna(ps1) else float(ps1) * 60.0
    return dict(wp=wp, hold=hold, tau=tau, end=tau + h, tids=tids,
                path=path,
                uav_id=(None if pd.isna(row.get("uav_id", np.nan)) else int(row.get("uav_id"))),
                battery_group=(None if pd.isna(row.get("battery_group", np.nan))
                               else int(row.get("battery_group"))),
                post_service_mode=str(row.get("post_service_mode", "none_after_last_mission")),
                service_start=ps0, service_end=(ps1 if ps1 is not None else tau + h))


def _pos_on(wp, t):
    if not (wp[0][0] <= t <= wp[-1][0]):
        return None
    for (t1, p1), (t2, p2) in zip(wp[:-1], wp[1:]):
        if t1 <= t <= t2:
            a = 0.0 if t2 <= t1 else (t - t1) / (t2 - t1)
            return np.asarray(p1) + (np.asarray(p2) - np.asarray(p1)) * a
    return None


# ---------------------------------------------------------------- E1 前沿图
def plot_e1(e1_csv: Path, out: Path, ks_show=(1, 2, 3, None), dpi=150):
    """三档 UAV 并排: safe_served vs B, 每档画 K∈ks_show(None=该档 K_max) 曲线族。"""
    import matplotlib.pyplot as plt
    import step13_experiment_model as S13

    df = pd.read_csv(e1_csv, encoding="utf-8-sig")
    _require_result_contract(df, "E1", {"result_contract", "uav", "K", "batteries",
                                          "safe_served", "plan_holds"})
    df = df.copy(); df["plan_holds"] = _bool_series_strict(df["plan_holds"], "E1.plan_holds")
    sel = S13.e1_select_from_df(df)          # 膝点(与实验完全同一函数, 免手抄)
    uavs = [u for u in ("S", "M", "L") if u in set(df["uav"])] or \
        sorted(df["uav"].drop_duplicates())
    fig, axes = plt.subplots(1, len(uavs), figsize=(5.2 * len(uavs), 4.4),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ks_show)))
    for ax, uk in zip(axes, uavs):
        sub = df[df["uav"] == uk]
        Kmax = int(sub["K"].max())
        ks = [K if K is not None else Kmax for K in ks_show]
        ks = sorted({k for k in ks if k in set(sub["K"])})
        for K, col in zip(ks, colors):
            cur = sub[sub["K"] == K].sort_values("batteries")
            ax.plot(cur["batteries"], cur["safe_served"], "-", color=col, lw=1.8,
                    label=f"K={K}" + ("(max)" if K == Kmax else ""))
            # 实心=同时置信上界下 plan_holds=True；缺失值已在读入阶段 fail-closed。
            bad = cur[~cur["plan_holds"]]
            good = cur[cur["plan_holds"]]
            ax.plot(good["batteries"], good["safe_served"], "o", color=col, ms=4)
            if len(bad):
                ax.plot(bad["batteries"], bad["safe_served"], "o", mfc="none",
                        mec=col, ms=6, mew=1.4)
        srow = sel[sel["uav"] == uk]
        if len(srow) and pd.notna(srow.iloc[0]["knee_B"]):
            r = srow.iloc[0]
            if bool(r.get("degenerate_knee", False)):
                ax.plot([r["knee_B"]], [r["knee_safe"]], "x", ms=11, mew=2,
                        color="crimson", zorder=5,
                        label="候选膝点(资源轴未收敛，不可自动选型)")
            else:
                ax.plot([r["knee_B"]], [r["knee_safe"]], "*", ms=17, color="crimson",
                        zorder=5, label=f"膝点 (K={int(r['knee_K'])},B={int(r['knee_B'])})")
            sat_txt = "已饱和" if bool(r["sat_reached"]) else "触顶未饱和"
            ax.axvline(int(sub["batteries"].max()), color="gray", ls=":", lw=1)
            ax.text(int(sub["batteries"].max()), ax.get_ylim()[0] + 1,
                    f" B_cap({sat_txt})", rotation=90, va="bottom", fontsize=8, color="gray")
        lbl = sub["uav_label"].iloc[0] if "uav_label" in sub else uk
        ax.set_title(f"{uk} 档: {lbl}", fontsize=11)
        ax.set_xlabel("电池数 B")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("safe_served(回放可靠覆盖台数)")
    fig.suptitle("E1 三轴前沿: safe_served–B 曲线族(空心点=该格 plan_holds=False)", fontsize=12)
    _add_solver_gap_footer(fig, df)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    log.info("写 %s", out)


# ---------------------------------------------------------------- E2 前沿图
_E2_ORDER = ["nominal", "gaussian", "SAA", "budget_G2", "box", "cantelli", "vp"]
_E2_COLOR = {"nominal": "#999999", "gaussian": "#4c72b0", "SAA": "#55a868",
             "budget_G2": "#c44e52", "box": "#8172b2", "cantelli": "#ccb974",
             "vp": "#d62728"}


def plot_e2(e2_csv: Path, out: Path, q_bad=0.8, dpi=150):
    import matplotlib.pyplot as plt

    df = pd.read_csv(e2_csv, encoding="utf-8-sig")
    _require_result_contract(df, "E2", {"result_contract", "criterion", "q", "safe_served",
                                          "max_col_viol", "emp_viol_upper95",
                                          "formal_reliability_claim_eligible", "evidence_scope"})
    if "run_status" in df.columns:
        df = df[df["run_status"].astype(str).str.lower() == "ok"].copy()
    if not len(df):
        raise ValueError("E2 CSV 没有成功完成的结果行")
    component_eps = float(df["component_eps"].iloc[0]) if "component_eps" in df.columns else \
        (float(df["eps"].iloc[0]) if "eps" in df.columns else 0.05)
    mission_eps = (float(df["mission_eps_budget"].iloc[0])
                   if "mission_eps_budget" in df.columns else
                   (float(df["eps_budget"].iloc[0]) if "eps_budget" in df.columns else component_eps))
    eligible = _bool_series_strict(df["formal_reliability_claim_eligible"],
                                   "E2.formal_reliability_claim_eligible")
    if eligible.any() and df.loc[eligible, "emp_viol_upper95"].isna().any():
        raise ValueError("E2 正式资格行缺少逐架次 Bonferroni 同时上界。")
    formal_plot = bool(eligible.all() and df["emp_viol_upper95"].notna().all())
    risk_col = "emp_viol_upper95" if formal_plot else "max_col_viol"
    reg = df[df["q"] < q_bad - 1e-9]
    bad = df[np.isclose(df["q"], q_bad)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    # (a) 常规窗: 安全–产出前沿
    for c in _E2_ORDER:
        s = reg[reg["criterion"] == c]
        if not len(s):
            continue
        x = s["safe_served"].sum()
        y = 100.0 * pd.to_numeric(s[risk_col], errors="raise").max()
        a1.scatter([x], [y], s=90, color=_E2_COLOR[c],
                   marker=("*" if c == "vp" else "o"),
                   edgecolors="k", linewidths=0.6, zorder=4)
        a1.annotate(c, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
    a1.axhline(100 * mission_eps, color="crimson", ls="--", lw=1.2)
    a1.axhline(100 * component_eps, color="black", ls=":", lw=1.0)
    a1.text(a1.get_xlim()[0], 100 * mission_eps,
            f" 联合任务预算={100*mission_eps:.0f}%", color="crimson", fontsize=8, va="bottom")
    a1.text(a1.get_xlim()[0], 100 * component_eps,
            f" 严格单阈值={100*component_eps:.0f}%", color="black", fontsize=8, va="bottom")
    a1.set_xlabel(f"Σ safe_served(常规窗 q<{q_bad:g} 合计)")
    a1.set_ylabel("逐架次同时单侧上界(%)" if formal_plot else "最坏单列点估计(%)")
    a1.set_title("(a) 正式独立测试：同时置信上界" if formal_plot
                 else "(a) 机制/部分证据：点估计，不构成正式安全结论")
    a1.grid(alpha=0.3)

    # (b) 恶劣窗 q=0.8: safe_served 条形 + emp_viol 标注
    cs = [c for c in _E2_ORDER if c in set(bad["criterion"])]
    xs = np.arange(len(cs))
    vals = [int(bad[bad["criterion"] == c]["safe_served"].iloc[0]) for c in cs]
    emp = [100.0 * float(bad[bad["criterion"] == c][risk_col].iloc[0]) for c in cs]
    bars = a2.bar(xs, vals, color=[_E2_COLOR[c] for c in cs], edgecolor="k", lw=0.5)
    for x, b, e in zip(xs, bars, emp):
        over = e > 100 * mission_eps + 1e-9
        a2.text(x, b.get_height() + 0.25, f"{e:.1f}%",
                ha="center", fontsize=8.5,
                color=("crimson" if over else "black"),
                fontweight=("bold" if over else "normal"))
    a2.set_xticks(xs, cs, rotation=25, ha="right")
    a2.set_ylabel("safe_served")
    a2.set_title(f"(b) 恶劣窗 q={q_bad:g}: 柱=可靠覆盖，标注="
                 + ("同时上界" if formal_plot else "点估计（非证书）"))
    a2.grid(alpha=0.3, axis="y")
    fig.suptitle(("E2 七判据对照（独立真实联合留出；逐架次 Bonferroni 同时上界）"
                  if formal_plot else
                  "E2 七判据对照（机制/部分证据；不得解释为正式95%可靠性结论）"), fontsize=12)
    _add_solver_gap_footer(fig, df)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    log.info("写 %s", out)

def _gif_resolve_radius(args, xi_amb):
    """更新: gif 半径解析 —— 'auto'=按 --detail 的 uav 档位物理外包络(与实验同口径)。"""
    import step9_model as _M
    s = str(args.pair_radius).strip().lower()
    if s != "auto":
        return float(s)
    uk = None
    try:
        dd = pd.read_csv(args.detail)
        if "uav" in dd.columns and len(dd):
            uk = str(dd["uav"].iloc[0])
    except Exception:
        pass
    h_max = float(max(xi_amb.horizons)) if getattr(xi_amb, "horizons", None) else 60.0
    try:
        p = _M.apply_uav_profile(_M.Params(), uk) if uk else _M.Params()
    except SystemExit:
        p = _M.Params()
    r = _M.max_flight_radius_m(p, min(h_max, 90.0))["R_max_m"]
    log.info("gif pair_radius[auto]=%.0fm(档位=%s)", r, uk or "基线")
    return float(r)


def gif_main(args):
    """GIF 子命令主体。"""
    turbines, wx_df, xi_amb, lat0lon0, sc_csv, src, track_auto = S13.load_all(args.n_turbines, farm=args.farm, allow_synth=True)
    args.pair_radius = _gif_resolve_radius(args, xi_amb)   # 更新: auto→物理半径
    track, kind, is_synth = _get_track(args, turbines, lat0lon0, track_auto)
    log.info("航迹: %s | 切窗后时长 %.1f min", kind, track.duration_sec() / 60.0)
    if args.probe:
        # 体检用未切窗原始候选: 重新只读探针
        cand = (Path(args.track_csv) if args.track_csv else
                (Path(__file__).resolve().parent / "tracks" / f"track_{args.track_mmsi}.csv"
                 if args.track_mmsi else None))
        tr = S13._load_track_ds(cand, *lat0lon0) if cand else track
        segs = S13._infarm_segments(tr, turbines, args.pair_radius)
        sp = np.linalg.norm(np.diff(tr.P, axis=0), axis=1) / np.maximum(np.diff(tr.t), 1e-9)
        print(f"航迹 {len(tr)} 点 | 时长 {tr.duration_sec()/60:.1f} min | 速度中位 {np.median(sp):.2f} m/s")
        print(f"进场时段(≤{args.pair_radius/1000:.0f}km), 共 {len(segs)} 段:")
        for a, b, d in segs[:12]:
            print(f"  {a/60:8.1f} → {b/60:8.1f} min  (驻留 {d/60:.1f} min)")
        return

    det = pd.read_csv(args.detail, encoding="utf-8-sig")
    tpos = {t.tid: np.asarray(t.local, float) for t in turbines}
    miss = sorted({tid for r in det.itertuples() for tid in str(r.turbines).split(";") if tid and tid not in tpos})
    if miss:
        raise SystemExit(f"明细风机不在布局(farm/子集不一致?): {miss[:5]} ...")
    flights = [_flight_schedule(r, track, tpos) for _, r in det.iterrows()]
    n_lane = _lanes(flights)
    done_at = {}
    for f in flights:
        for tid, a, b in f["hold"]:
            done_at[tid] = b

    use_cn = _setup_font()
    import matplotlib.pyplot as plt
    from matplotlib import animation
    L = (dict(t="时间", served="已服务", act="在飞", swap="换电中", bat="已用电池", ship="母船",
              turb="风机", done="已服务", insp="巡检中")
         if use_cn else dict(t="t", served="served", act="airborne", swap="swapping", bat="batteries",
                             ship="ship", turb="turbine", done="served", insp="inspecting"))
    TP = np.array([t.local for t in turbines], float)
    fig, ax = plt.subplots(figsize=(9.6, 6.8)); ax.set_aspect("equal")
    xs = np.concatenate([TP[:, 0], track.P[:, 0]]); ys = np.concatenate([TP[:, 1], track.P[:, 1]])
    ax.set_xlim(xs.min() - 1500, xs.max() + 1500); ax.set_ylim(ys.min() - 1500, ys.max() + 2600)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.scatter(TP[:, 0], TP[:, 1], s=26, c="#9aa4ad", zorder=2, label=L["turb"])
    sc_done = ax.scatter([], [], s=58, c="#1a9c46", zorder=3, label=L["done"])
    sc_insp = ax.scatter([], [], s=300, facecolors="none", linewidths=2.2, zorder=5)
    trail, = ax.plot([], [], "-", lw=1.6, c="#1f5fbf", alpha=0.85, zorder=4)
    shipm, = ax.plot([], [], marker=(3, 0, 0), ms=13, c="#1f5fbf", zorder=7, label=L["ship"])
    head = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color="#1f5fbf", lw=1.6), zorder=7)
    plan_lines = [ax.plot([], [], "--", lw=1.1, alpha=0.7, c=PALETTE[f["lane"] % len(PALETTE)],
                          zorder=4)[0] for f in flights]
    uav_pts = [ax.plot([], [], "o", ms=9, mec="k", mew=0.5,
                       c=PALETTE[f["lane"] % len(PALETTE)], zorder=8)[0] for f in flights]
    uav_lab = [ax.text(0, 0, "", fontsize=8, weight="bold", zorder=9, visible=False) for _ in flights]
    swap_txt = ax.text(0.012, 0.018, "", transform=ax.transAxes, fontsize=10, color="#b34700",
                       va="bottom", zorder=9)
    hud = ax.text(0.012, 0.985, "", transform=ax.transAxes, va="top", fontsize=11,
                  bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.92), zorder=9)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(f"UAV inspection — {kind} — K(display)={n_lane}", fontsize=10.5)
    if is_synth:
        ax.text(0.5, 0.5, "SYNTHETIC TRACK\n非真实 AIS — 仅演示" if use_cn else "SYNTHETIC TRACK — DEMO",
                transform=ax.transAxes, ha="center", va="center", fontsize=26, color="red",
                alpha=0.25, rotation=18, zorder=1)

    n_fr = int(np.floor(args.window_min / args.step_min)) + 1
    tt = np.arange(0.0, args.window_min * 60.0 + 1, 30.0)
    tP = np.array([track.pos(x) for x in tt])

    def update(fi):
        t = fi * args.step_min * 60.0
        trail.set_data(tP[tt <= t + 1e-9, 0], tP[tt <= t + 1e-9, 1])
        sp = track.pos(t); v = track.vel(t)
        shipm.set_data([sp[0]], [sp[1]])
        nv = np.linalg.norm(v)
        head.set_position((sp[0], sp[1]))
        head.xy = (sp[0] + (v[0] / nv * 700 if nv > 1e-6 else 0),
                   sp[1] + (v[1] / nv * 700 if nv > 1e-6 else 0))
        act = swp = 0; insp_pts = []; insp_cols = []
        for f, pl, up, ul in zip(flights, plan_lines, uav_pts, uav_lab):
            pos = _pos_on(f["wp"], t)
            col = PALETTE[f["lane"] % len(PALETTE)]
            if pos is not None:
                act += 1
                up.set_data([pos[0]], [pos[1]]); up.set_visible(True)
                ul.set_position((pos[0] + 220, pos[1] + 220)); ul.set_text(f"D{f['lane']+1}")
                ul.set_color(col); ul.set_visible(True)
                pl.set_data(f["path"][:, 0], f["path"][:, 1]); pl.set_visible(True)
                for tid, a, b in f["hold"]:
                    if a <= t <= b:
                        insp_pts.append(tpos[tid]); insp_cols.append(col)
            else:
                up.set_visible(False); ul.set_visible(False); pl.set_visible(False)
        sw_msgs = []
        for f in flights:
            a, b = f.get("service_start"), f.get("service_end")
            if a is None or b is None or not (a <= t < b):
                continue
            mode = f.get("post_service_mode")
            if mode == "battery_swap":
                label = L["swap"]
            elif mode == "quick_reuse":
                label = ("快检" if use_cn else "quick-check")
            else:
                continue
            r = b - t
            batt = f.get("battery_group")
            btxt = "" if batt is None else f" B{batt+1}"
            sw_msgs.append(f"D{f['lane']+1}{btxt} {label} {int(r//60)}:{int(r%60):02d}")
        swp = len(sw_msgs)
        swap_txt.set_text("  |  ".join(sw_msgs))
        sc_insp.set_offsets(np.array(insp_pts) if insp_pts else np.empty((0, 2)))
        sc_insp.set_edgecolors(insp_cols if insp_cols else "none")
        done = [tid for tid, b in done_at.items() if b <= t]
        sc_done.set_offsets(np.array([tpos[d] for d in done]) if done else np.empty((0, 2)))
        used_batts = {f.get("battery_group") for f in flights
                      if f["tau"] <= t and f.get("battery_group") is not None}
        total_batts = len({f.get("battery_group") for f in flights
                           if f.get("battery_group") is not None})
        hud.set_text(f"{L['t']} = {int(t//3600)}:{int(t%3600//60):02d}:{int(t%60):02d}  |  "
                     f"{L['served']} {len(done)}/{len(tpos)}  |  {L['act']} {act}  |  "
                     f"service {swp}  |  {L['bat']} {len(used_batts)}/{total_batts}")
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_fr, blit=False)
    out = Path(args.out)
    log.info("渲染 %d 帧 → %s (fps=%d, step=%.2gmin, dpi=%d) ...", n_fr, out, args.fps, args.step_min, args.dpi)
    ani.save(out, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    log.info("完成: %s", out.resolve())


def plots_main(args):
    """plots 子命令主体(原 step16_plots_e1e2.main)。"""
    _setup_font()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.e1.is_file():
        plot_e1(args.e1, args.out_dir / "fig_E1_frontier.png", dpi=args.dpi)
    else:
        log.warning("跳过 E1(未找到 %s)", args.e1)
    if args.e2.is_file():
        plot_e2(args.e2, args.out_dir / "fig_E2_frontier.png", dpi=args.dpi)
    else:
        log.warning("跳过 E2(未找到 %s)", args.e2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    here = Path(__file__).resolve().parent

    p = sub.add_parser("plots", help="论文静态图(E1/E2)")
    p.add_argument("--e1", type=Path,
                   default=here / "results/model_experiments/E1_frontier/E1_frontier.csv")
    p.add_argument("--e2", type=Path,
                   default=here / "results/model_experiments/E2_robust_comparison/E2_robust_raw.csv")
    p.add_argument("--out-dir", type=Path, default=here / "results/figs")
    p.add_argument("--dpi", type=int, default=150)

    g = sub.add_parser("gif", help="任务动画 GIF")
    g.add_argument("--detail", default="results/model_experiments/E1_frontier/E1_detail_Kmax.csv")
    g.add_argument("--track-csv", default=None)
    g.add_argument("--track-mmsi", default=None)
    g.add_argument("--track-start-min", type=float, default=None)
    g.add_argument("--n-turbines", type=int, default=None)
    g.add_argument("--farm", default="Rodsand_II", choices=["Rodsand_II", "Nysted", "Anholt"])
    g.add_argument("--pair-radius", type=str, default="auto",
                   help="GIF 场景半径。'auto'(更新)=按 --detail 的 uav 档位物理外包络"
                        "(与实验同口径; 无明细/无档位列时用基线档); 数字=显式米数")
    g.add_argument("--window-min", type=float, default=360.0)
    g.add_argument("--step-min", type=float, default=0.5, help="每帧任务时间(分钟, 默认已调慢)")
    g.add_argument("--fps", type=int, default=6)
    g.add_argument("--dpi", type=int, default=110)
    g.add_argument("--out", default="uav_mission.gif")
    g.add_argument("--probe", action="store_true")

    args = ap.parse_args()
    _setup_font()
    if args.cmd == "plots":
        plots_main(args)
    else:
        gif_main(args)


if __name__ == "__main__":
    main()
