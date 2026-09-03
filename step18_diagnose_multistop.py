#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step18_diagnose_multistop.py

Formal-model diagnostic only.  This script never emits an optimization
certificate and never changes the feasible region.  Its purpose is to answer:

  1) Why do current E1 incumbents use only singleton routes?
  2) Are multi-stop routes physically/DRCC feasible under the *same* formal
     model?
  3) If they are feasible, can a single such route pass the exact entity
     resource audit with K=1, B=1?
  4) Did the formal exact BPC simply fail to discover/use them before its
     pricing deadline?
  5) Which physical constraints are responsible for multi-stop near misses?

The diagnostic is intentionally independent of the certificate path.  Every
candidate used for a "formal-feasible" statement is evaluated by the same
route_feasible_at_h / _candidate_from_physics oracle as step12.  Heuristic
route pools are used only to *find candidates*; any candidate counted as a
formal seed is revalidated by step12 before classification.

Typical command is printed by --help and documented in README_FOR_AI.md.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import step9_model as M
import step10_model_routing as RM
import step11_algorithm_route_drcc as RA
import step12_branch_price as BP
import step13_experiment_model as E
import step15_replay as RP

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "model_experiments" / "multistop_diagnostics"

DIAGNOSTIC_CONTRACT = "multistop-root-cause-diagnostic-v1"
FORMAL_KAPPA_MODE = "vp_unimodal"


def _tid(t):
    return str(getattr(t, "tid", t))


def _ordered_tids(c):
    seq = c.get("ordered_tids")
    if seq:
        return tuple(str(x) for x in seq)
    route = c.get("route")
    if route is not None:
        return tuple(str(x) for x in route.turbine_ids())
    tids = c.get("tids")
    return tuple(str(x) for x in (tids or ()))


def _boolish(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _finite(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _print_header(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _read_existing_e1(path: Path | None):
    if path is None or not path.is_file():
        return None
    df = pd.read_csv(path)
    return df


def _existing_e1_audit(df: pd.DataFrame | None):
    rows = []
    if df is None or df.empty:
        return pd.DataFrame(rows)
    for uk in sorted(df["uav"].astype(str).unique()):
        g = df[df["uav"].astype(str) == uk].copy()
        for K, B, role in [(1, 1, "minimal-K1B1"),
                           (int(g["K"].max()), int(g["batteries"].max()), "max-observed-resource")]:
            q = g[(g["K"] == K) & (g["batteries"] == B)]
            if q.empty:
                continue
            r = q.iloc[-1]
            rows.append(dict(
                uav=uk, role=role, K=int(K), batteries=int(B),
                covered=int(r.get("covered", r.get("coverage_incumbent", 0)) or 0),
                safe_served=int(r.get("safe_served", 0) or 0),
                mean_stops=_finite(r.get("mean_stops")),
                max_stops_observed=int(r.get("max_stops_observed", 0) or 0),
                multi_stop_ratio=_finite(r.get("multi_stop_ratio")),
                termination_reason=str(r.get("termination_reason", "unknown")),
                solve_scope=str(r.get("solve_scope", "unknown")),
                coverage_incumbent=_finite(r.get("coverage_incumbent")),
                coverage_upper_bound=_finite(r.get("coverage_upper_bound")),
                coverage_gap_abs=_finite(r.get("coverage_gap_abs")),
                coverage_optimal=_boolish(r.get("coverage_optimal")),
                coverage_certificate=_boolish(r.get("coverage_global_certificate_available")),
                lexicographic_optimal=_boolish(r.get("lexicographic_optimal")),
                global_certificate=_boolish(r.get("global_certificate_available")),
                pricing_complete=_boolish(r.get("pricing_complete")),
                pricing_bound_available=_boolish(r.get("pricing_bound_available")),
                resource_audit_complete=_boolish(r.get("resource_audit_complete")),
                runtime_s=_finite(r.get("runtime_s")),
                pool_size=_finite(r.get("pool_size")),
                on_demand_generated_columns=_finite(r.get("on_demand_generated_columns")),
            ))
    return pd.DataFrame(rows)


def _profile_floor_summary(p, turbines, h_max):
    # Strictly diagnostic lower floors.  These are not feasibility certificates.
    # They reveal whether service time/energy alone already consumes the mission.
    p_orbit = float(M.P_zeng(float(p.v_orbit), p) if getattr(p, "use_zeng", False) else p.P_hov)
    dz = max(float(turbines[0].H_tip) - float(p.z_cruise), 0.0) if turbines else 0.0
    if getattr(p, "use_zeng", False):
        p_up = float(M.P_zeng(0.0, p) + 7.27 * 9.81 * float(p.v_z))
    else:
        p_up = float(p.P_climb)
    t_vert = dz / float(p.v_z) if p.v_z > 0 else math.inf
    e_vert = p_up * t_vert / 3600.0
    e_inspection = p_orbit * float(p.tau_insp) / 3600.0
    return dict(
        B_k_Wh=float(p.B_k), B_use_Wh=float(p.B_use),
        safe_reserve=float(p.safe_reserve),
        h_max_min=float(h_max),
        tau_insp_min=float(p.tau_insp) / 60.0,
        inspection_power_W=p_orbit,
        vertical_span_m=dz,
        vertical_time_s=t_vert,
        vertical_energy_Wh=e_vert,
        per_stop_service_time_s=t_vert + float(p.tau_insp),
        per_stop_service_energy_Wh=e_vert + e_inspection,
        dock_base_s=float(p.t_dock_base_s),
        service_only_2stop_time_min=2.0 * (t_vert + float(p.tau_insp)) / 60.0,
        service_only_3stop_time_min=3.0 * (t_vert + float(p.tau_insp)) / 60.0,
        service_only_4stop_time_min=4.0 * (t_vert + float(p.tau_insp)) / 60.0,
        service_only_2stop_energy_Wh=2.0 * (e_vert + e_inspection),
        service_only_3stop_energy_Wh=3.0 * (e_vert + e_inspection),
        service_only_4stop_energy_Wh=4.0 * (e_vert + e_inspection),
    )


def _geometry_summary(turbines, opts):
    tp = np.asarray([t.local for t in turbines], float)
    pair = []
    for i in range(len(tp)):
        for j in range(i + 1, len(tp)):
            pair.append(float(np.linalg.norm(tp[i] - tp[j])))
    stand = []
    for opt in opts:
        if len(tp):
            stand.append(float(np.min(np.linalg.norm(tp - np.asarray(opt.ship.P_launch, float), axis=1))))
    def stats(v):
        if not v:
            return dict(min=None, median=None, max=None)
        a = np.asarray(v, float)
        return dict(min=float(np.min(a)), median=float(np.median(a)), max=float(np.max(a)))
    return dict(pairwise_turbine_distance_m=stats(pair), launch_standoff_to_nearest_turbine_m=stats(stand))


def _failure_row(diag, uk, nstops, seq, opt, h, source):
    flags = diag.get("failure_flags") or {}
    margins = diag.get("margins") or {}
    return dict(
        uav=uk, source=source, stops=int(nstops),
        ordered_tids=">".join(seq),
        tau=float(opt.tau_min), h=float(h),
        launch_state=str(getattr(opt.ship, "c_state", "unknown")),
        feasible=bool(diag.get("feasible", False)),
        primary_reason=str(diag.get("primary_reason", diag.get("reason", "unknown"))),
        margin_E_Wh=_finite(diag.get("margin_E", margins.get("energy_Wh"))),
        margin_T_s=_finite(diag.get("margin_T", margins.get("time_s"))),
        route_airspeed_margin_ms=_finite(margins.get("route_airspeed_ms")),
        landing_wind_margin_ms=_finite(margins.get("landing_wind_ms")),
        escort_airspeed_margin_ms=_finite(diag.get("escort_margin_ms", margins.get("escort_airspeed_ms"))),
        E_plan_Wh=_finite(diag.get("E_plan_Wh")),
        E_soc_required_Wh=_finite(diag.get("E_soc_required_Wh")),
        nominal_time_margin_s=_finite(diag.get("nominal_time_margin_s")),
        time_drcc_tightening_s=_finite(diag.get("time_drcc_tightening_s")),
        time_core_nom_s=_finite(diag.get("time_core_nom_s")),
        time_wait_nom_s=_finite(diag.get("time_wait_nom_s")),
        failure_flags=json.dumps({k: bool(v) for k, v in flags.items() if bool(v)}, ensure_ascii=False, sort_keys=True),
    )


def _exact_candidate(opt_index, opt, seq, h, p, xi_amb, wamb, args, t_launch):
    risk_policy = BP._risk_policy_for_mode(FORMAL_KAPPA_MODE)
    c = BP._candidate_from_physics(
        opt_index, opt, seq, h, p, xi_amb, wamb,
        t_launch, args.landing_clear_min, args.deck_mode, args.deck_delta_min,
        chance_mode="drcc", budget_gamma=float(args.budget_gamma),
        deadline=None, risk_policy=risk_policy)
    return c


def _diag_route(opt, seq, h, p, xi_amb, wamb, args):
    route = RM.Route(rid=-1, turbines=list(seq), ship=opt.ship)
    return RM.route_feasible_at_h(
        route, int(h) if float(h).is_integer() else float(h),
        p, opt.wx, xi_amb, weather_unc=wamb,
        chance_mode="drcc", budget_gamma=float(args.budget_gamma),
        risk_policy=BP._risk_policy_for_mode(FORMAL_KAPPA_MODE))


def _sequence_geom_score(seq):
    if len(seq) <= 1:
        return 0.0
    return float(sum(np.linalg.norm(np.asarray(a.local) - np.asarray(b.local))
                     for a, b in zip(seq[:-1], seq[1:])))


def _prioritized_sequences(turbines, nstops, cap):
    seqs = list(itertools.permutations(turbines, nstops))
    seqs.sort(key=lambda s: (_sequence_geom_score(s), tuple(_tid(t) for t in s)))
    if cap and cap > 0:
        seqs = seqs[:cap]
    return seqs


def _prioritized_opts(opts, turbines):
    tp = np.asarray([t.local for t in turbines], float)
    def score(pair):
        i, opt = pair
        if not len(tp):
            return (0.0, i)
        d = float(np.min(np.linalg.norm(tp - np.asarray(opt.ship.P_launch, float), axis=1)))
        # DP / low-speed launch states first, then geometry.
        state = str(getattr(opt.ship, "c_state", "unknown"))
        srank = {"动力定位": 0, "低速": 1, "直航": 2, "转弯": 3}.get(state, 4)
        return (srank, d, float(opt.tau_min), i)
    return sorted(list(enumerate(opts)), key=score)


def _scan_multistop(uk, turbines, opts, p, xi_amb, wamb, args, t_launch):
    horizons = tuple(float(h) for h in RM.decision_horizons_of(xi_amb))
    opt_order = _prioritized_opts(opts, turbines)
    deadline = time.monotonic() + float(args.exact_scan_seconds_per_uav)
    all_rows = []
    feasible_cols = []
    complete_by_stop = {}
    evals_by_stop = Counter()

    for nstops in range(2, int(args.max_stops) + 1):
        cap = 0 if nstops == 2 and args.pair_exhaustive else int(args.higher_stop_sequences)
        seqs = _prioritized_sequences(turbines, nstops, cap)
        complete = True
        stop_eval_cap = int(args.route_evals_per_stop)
        for seq in seqs:
            if time.monotonic() >= deadline:
                complete = False
                break
            for oi, opt in opt_order:
                if time.monotonic() >= deadline:
                    complete = False
                    break
                seq_ids = tuple(_tid(t) for t in seq)
                for h in horizons:
                    if time.monotonic() >= deadline:
                        complete = False
                        break
                    if float(opt.tau_min) + float(h) > float(args.window_min):
                        continue
                    diag = _diag_route(opt, seq, h, p, xi_amb, wamb, args)
                    all_rows.append(_failure_row(diag, uk, nstops, seq_ids, opt, h, "direct-formal-physics"))
                    evals_by_stop[nstops] += 1
                    if bool(diag.get("feasible", False)):
                        try:
                            c = _exact_candidate(oi, opt, seq, h, p, xi_amb, wamb, args, t_launch)
                        except Exception:
                            c = None
                        if c is not None:
                            feasible_cols.append(c)
                    if stop_eval_cap > 0 and evals_by_stop[nstops] >= stop_eval_cap:
                        complete = False
                        break
                if stop_eval_cap > 0 and evals_by_stop[nstops] >= stop_eval_cap:
                    break
            if stop_eval_cap > 0 and evals_by_stop[nstops] >= stop_eval_cap:
                break
        full_sequence_count = math.factorial(len(turbines)) // math.factorial(
            max(0, len(turbines) - nstops))
        universe_complete = bool(complete and len(seqs) == full_sequence_count)
        complete_by_stop[nstops] = universe_complete
        # ``complete`` over a selected top-N subset is not a formal route-universe
        # completeness statement.  Only an actually exhaustive permutation layer
        # is reported True here.
    return all_rows, feasible_cols, complete_by_stop, dict(evals_by_stop)


def _heuristic_pool_probe(uk, turbines, opts, p, xi_amb, wamb, args, t_swap, t_launch):
    ledger = []
    deadline = time.monotonic() + float(args.heuristic_pool_seconds_per_uav)
    try:
        cols = RA.build_route_columns(
            turbines, opts, p, xi_amb, args.window_min, args.deck_delta_min,
            int(args.max_stops), wamb, "drcc", float(args.budget_gamma),
            FORMAL_KAPPA_MODE, 8.0, pool_h_mode="pareto",
            diagnostics_sink=ledger, deadline=deadline)
        status = "completed"
    except TimeoutError:
        cols = []
        status = "timeout"
    except Exception as exc:
        cols = []
        status = f"error:{type(exc).__name__}:{exc}"

    raw_multi = [c for c in cols if len(_ordered_tids(c)) >= 2]
    revalidated = []
    rejected = []
    for raw in raw_multi[: int(args.revalidate_seed_cap)]:
        try:
            c = BP._revalidate_seed_column(
                raw, turbines, opts, p, xi_amb, wamb, args.window_min,
                t_launch, args.landing_clear_min, args.deck_mode,
                args.deck_delta_min, FORMAL_KAPPA_MODE, "drcc",
                float(args.budget_gamma), deadline=None)
            revalidated.append(c)
        except Exception as exc:
            rejected.append((raw, f"{type(exc).__name__}:{exc}"))

    resource_feasible = []
    resource_unknown = []
    for c in revalidated:
        audit = BP._audit_integer_selection(
            [c], (0,), 1, 1, p,
            float(p.quick_inspection_min), float(t_swap),
            int(p.quick_inspection_capacity), int(p.swap_station_capacity),
            deadline=None)
        if audit.status is RA.ResourceAuditStatus.FEASIBLE:
            resource_feasible.append(c)
        elif audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
            resource_unknown.append(c)

    summary = dict(
        uav=uk, heuristic_pool_status=status,
        heuristic_columns=len(cols),
        heuristic_multistop_columns=len(raw_multi),
        revalidated_multistop_columns=len(revalidated),
        revalidation_rejected=len(rejected),
        single_route_resource_feasible_multistop=len(resource_feasible),
        single_route_resource_unknown_multistop=len(resource_unknown),
        max_revalidated_stops=max((len(_ordered_tids(c)) for c in revalidated), default=0),
        max_resource_feasible_stops=max((len(_ordered_tids(c)) for c in resource_feasible), default=0),
    )
    return summary, ledger, revalidated, resource_feasible


def _failure_summary(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for (uk, n), g in df.groupby(["uav", "stops"]):
        out.append(dict(
            uav=uk, stops=int(n), evaluated=len(g), feasible=int(g["feasible"].sum()),
            feasible_fraction=float(g["feasible"].mean()),
            top_primary_reasons=json.dumps(
                Counter(g.loc[~g["feasible"], "primary_reason"].astype(str)).most_common(8),
                ensure_ascii=False),
            energy_failed=int(g["failure_flags"].str.contains("energy_drcc_failed", regex=False).sum()),
            time_failed=int(g["failure_flags"].str.contains("time_drcc_failed", regex=False).sum()),
            airspeed_failed=int(g["failure_flags"].str.contains("route_airspeed_failed", regex=False).sum()),
            escort_failed=int(g["failure_flags"].str.contains("escort_airspeed_failed", regex=False).sum()),
            landing_failed=int(g["failure_flags"].str.contains("landing_gate_failed", regex=False).sum()),
        ))
    return pd.DataFrame(out)


def _near_misses(rows, n=30):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    bad = df[~df["feasible"]].copy()
    if bad.empty:
        return bad
    def penalty(r):
        e = max(0.0, -float(r["margin_E_Wh"])) if pd.notna(r["margin_E_Wh"]) else 1e9
        t = max(0.0, -float(r["margin_T_s"])) if pd.notna(r["margin_T_s"]) else 1e9
        a = max(0.0, -float(r["route_airspeed_margin_ms"])) if pd.notna(r["route_airspeed_margin_ms"]) else 1e6
        esc = max(0.0, -float(r["escort_airspeed_margin_ms"])) if pd.notna(r["escort_airspeed_margin_ms"]) else 1e6
        # Normalized only for ranking; never used as feasibility.
        return e / 50.0 + t / 300.0 + a / 2.0 + esc / 2.0
    bad["_near_miss_penalty"] = bad.apply(penalty, axis=1)
    return bad.sort_values(["stops", "_near_miss_penalty"]).head(int(n))


def _ablation_probe(candidates, p, xi_amb, wamb, args, limit=12):
    """Diagnostic counterfactuals only; never formal conclusions."""
    rows = []
    for item in candidates[:limit]:
        opt, seq, h = item
        variants = []
        variants.append(("formal-full", p, wamb))
        p_no_geo = copy.deepcopy(p); p_no_geo.soc_correction = "none"
        variants.append(("no-geo2d-diagnostic", p_no_geo, wamb))
        variants.append(("no-weather-uncertainty-diagnostic", p, None))
        p_zero_insp = copy.deepcopy(p); p_zero_insp.tau_insp = 0.0
        variants.append(("zero-inspection-time-diagnostic", p_zero_insp, wamb))
        for name, pv, wv in variants:
            try:
                d = _diag_route(opt, seq, h, pv, xi_amb, wv, args)
                rows.append(dict(
                    variant=name, ordered_tids=">".join(_tid(t) for t in seq),
                    tau=float(opt.tau_min), h=float(h),
                    feasible=bool(d.get("feasible", False)),
                    reason=str(d.get("primary_reason", d.get("reason", "unknown"))),
                    margin_E_Wh=_finite(d.get("margin_E")),
                    margin_T_s=_finite(d.get("margin_T")),
                ))
            except Exception as exc:
                rows.append(dict(variant=name, ordered_tids=">".join(_tid(t) for t in seq),
                                 tau=float(opt.tau_min), h=float(h), feasible=False,
                                 reason=f"error:{type(exc).__name__}:{exc}",
                                 margin_E_Wh=None, margin_T_s=None))
    return pd.DataFrame(rows)


def _classify_root_cause(existing_row, pool_summary, direct_resource_cols, scan_rows, scan_complete):
    messages = []
    k1_cov = None
    k1_cert = False
    k1_ub = None
    if existing_row is not None:
        k1_cov = existing_row.get("covered")
        k1_cert = bool(existing_row.get("coverage_certificate", False))
        k1_ub = existing_row.get("coverage_upper_bound")

    resource_multistop = int(pool_summary.get("single_route_resource_feasible_multistop", 0)) > 0
    resource_multistop = resource_multistop or bool(direct_resource_cols)

    if resource_multistop:
        if k1_cert and k1_cov is not None and int(k1_cov) < 2:
            messages.append("CRITICAL: 找到 K=1,B=1 可独立资源执行的 multi-stop，但现有 K1B1 coverage 已被正式证明 <2；这构成证书/实例一致性冲突，禁止使用当前结果。")
        elif k1_cov is not None and int(k1_cov) < 2:
            messages.append("CONFIRMED ALGORITHMIC DISCOVERY BOTTLENECK: 同一正式物理模型存在 K=1,B=1 可执行 multi-stop，但当前 anytime incumbent 仍为1；multi-stop 未及时进入 incumbent，优先检查 singleton-only warm start / exact pricing timeout。")
        else:
            messages.append("multi-stop 物理+资源可行；当前解仍偏 singleton 时，原因更可能是 Stage-2 能耗选择或并发资源时序，而不是物理不可行。")
    else:
        fdf = _failure_summary(scan_rows)
        if not fdf.empty:
            top = []
            for _, r in fdf.iterrows():
                top.append(f"{int(r['stops'])}-stop: feas={int(r['feasible'])}/{int(r['evaluated'])}, reasons={r['top_primary_reasons']}")
            messages.append("当前扫描尚未发现 K=1,B=1 可独立执行 multi-stop。主要物理拒绝统计: " + " | ".join(top))
        if all(scan_complete.get(n, False) for n in scan_complete) and scan_complete:
            messages.append("对声明为 complete 的 stop 层，未发现可行 multi-stop 是物理 oracle 的完整扫描结论；未 complete 的更高 stop 层仍只能视为诊断证据。")
        else:
            messages.append("物理扫描受时间/样本上限影响，未发现 multi-stop 不能升级成“不存在”的证明。")

    if existing_row is not None and not bool(existing_row.get("pricing_complete", False)):
        messages.append("EXISTING BPC PRICING NOT CLOSED: 现有 K1B1 行 pricing_complete=False，因此 singleton incumbent 不能解释为已证明的路线结构最优。")
    if existing_row is not None and k1_ub is not None and k1_cov is not None and float(k1_ub) > float(k1_cov):
        messages.append(f"EXISTING COVERAGE GAP: K1B1 incumbent={k1_cov}, UB={k1_ub}；仍存在未排除的额外覆盖。")
    return messages


def _save(df, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_parser():
    ap = argparse.ArgumentParser(
        description="诊断 E1 为什么全部 singleton；不改变模型、不签发证书。")
    ap.add_argument("--study-mode", choices=["formal", "mechanism"], default="formal")
    ap.add_argument("--n-turbines", type=int, default=8)
    ap.add_argument("--farm", default="Rodsand_II")
    ap.add_argument("--uavs", default="S,M,L")
    ap.add_argument("--turbines-csv", type=Path, required=True)
    ap.add_argument("--wind-csv", type=Path, required=True)
    ap.add_argument("--wave-csv", type=Path, required=True)
    ap.add_argument("--xi-moments-csv", type=Path, default=None)
    ap.add_argument("--xi-train-samples", type=Path, required=True)
    ap.add_argument("--weather-moments-csv", type=Path, required=True)
    ap.add_argument("--recovery-scenarios-csv", type=Path, default=None)
    ap.add_argument("--track-csv", type=Path, required=True)
    ap.add_argument("--track-start-min", type=float, default=None)
    ap.add_argument("--window-min", type=float, default=360.0)
    ap.add_argument("--dtau-min", type=float, default=5.0)
    ap.add_argument("--pair-radius", default="auto")
    ap.add_argument("--infarm-radius", type=float, default=3000.0)
    ap.add_argument("--hs-quantile", type=float, default=0.5)
    ap.add_argument("--weather-alignment", choices=["timestamp", "representative_quantile"], default="timestamp")
    ap.add_argument("--recovery-predictor", choices=["cv_noleak", "true_track"], default="cv_noleak")
    ap.add_argument("--soc-correction", choices=["none", "geo2d"], default="geo2d")
    ap.add_argument("--soc-risk-allocation", choices=["fixed", "optimized"], default="optimized")
    ap.add_argument("--time-recourse", choices=["wait_only", "wait_and_speed"], default="wait_and_speed")
    ap.add_argument("--max-stops", type=int, default=4)
    ap.add_argument("--deck-mode", choices=["interval", "slot"], default="interval")
    ap.add_argument("--deck-delta-min", type=float, default=2.5)
    ap.add_argument("--landing-clear-min", type=float, default=1.0)
    ap.add_argument("--t-swap-min", type=float, default=None)
    ap.add_argument("--t-launch-min", type=float, default=None)
    ap.add_argument("--quick-inspection-capacity", type=int, default=1)
    ap.add_argument("--swap-stations", type=int, default=1)
    ap.add_argument("--budget-gamma", type=float, default=2.0)
    ap.add_argument("--existing-e1-csv", type=Path,
                    default=HERE / "results" / "model_experiments" / "E1_frontier" / "E1_frontier.csv")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)

    # Diagnostic work controls.  These are not optimization tolerances.
    ap.add_argument("--heuristic-pool-seconds-per-uav", type=float, default=180.0)
    ap.add_argument("--exact-scan-seconds-per-uav", type=float, default=600.0)
    ap.add_argument("--pair-exhaustive", choices=["on", "off"], default="on")
    ap.add_argument("--higher-stop-sequences", type=int, default=120,
                    help="3/4-stop 按最短台间几何路径优先抽样的序列数；0=全排列。")
    ap.add_argument("--route-evals-per-stop", type=int, default=0,
                    help="每 stop 层 exact physics 最大评估数；0=只受时间限制。")
    ap.add_argument("--revalidate-seed-cap", type=int, default=128)
    ap.add_argument("--near-miss-count", type=int, default=40)
    return ap


def main():
    args = build_parser().parse_args()
    args.pair_exhaustive = (args.pair_exhaustive == "on")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _print_header("STEP18 / MULTI-STOP ROOT-CAUSE DIAGNOSTIC")
    print("contract:", DIAGNOSTIC_CONTRACT)
    print("NOTE: 本脚本只诊断，不签发 coverage/energy/global certificate。")

    for pth in (args.turbines_csv, args.wind_csv, args.wave_csv,
                args.xi_train_samples, args.weather_moments_csv, args.track_csv):
        if pth is None or not Path(pth).is_file():
            raise SystemExit(f"缺少输入文件: {pth}")

    turbines, wx_df, xi_amb0, lat0lon0, sc_csv, src, track_csv = E.load_all(
        args.n_turbines, farm=args.farm, allow_synth=False,
        turbines_csv=args.turbines_csv, wind_csv=args.wind_csv, wave_csv=args.wave_csv,
        xi_moments_csv=args.xi_moments_csv,
        recovery_scenarios_csv=args.recovery_scenarios_csv,
        track_csv=args.track_csv, require_xi_moments=False)

    _diag_mmsi = E._infer_concrete_track_mmsi(
        args.track_csv, None, formal=(args.study_mode == "formal"))
    train_df = RP.load_samples(
        args.xi_train_samples,
        mmsi=(_diag_mmsi if args.study_mode == "formal" else "ALL"),
        formal=(args.study_mode == "formal"),
        expected_split=("train" if args.study_mode == "formal" else None))
    hs = sorted(train_df["h_min"].astype(int).unique())
    states = sorted(train_df["c_state"].astype(str).unique())
    xi_amb = RP.ambiguity_from_samples(
        train_df, hs, states, formal=(args.study_mode == "formal"))

    if str(getattr(xi_amb, "predictor", "unknown")) != str(args.recovery_predictor):
        raise SystemExit(
            f"ξ predictor={getattr(xi_amb,'predictor','unknown')!r} 与 --recovery-predictor={args.recovery_predictor!r} 不一致。")
    wamb = RM.weather_ambiguity_from_moments_csv(
        args.weather_moments_csv, RM.decision_horizons_of(xi_amb),
        formal=(args.study_mode == "formal"))

    p_base = M.Params()
    p_base.quick_inspection_capacity = int(args.quick_inspection_capacity)
    p_base.swap_station_capacity = int(args.swap_stations)
    p_base.soc_correction = str(args.soc_correction)
    p_base.soc_risk_allocation = str(args.soc_risk_allocation)
    p_base.time_recourse_mode = str(args.time_recourse)
    p_base.speed_adjustable = (p_base.time_recourse_mode == "wait_and_speed")
    p_base.validate_contract(formal=(args.study_mode == "formal"))

    pair_args = SimpleNamespace(
        pair_radius=args.pair_radius, e1_uavs=args.uavs,
        uav=str(args.uavs).split(",")[0].strip())
    pair_radius, pair_mode = E._resolve_pair_radius(pair_args, p_base, xi_amb, turbines)
    opts, reach, kind, T_eff, wx0 = E.build_launch_options(
        turbines, lat0lon0, args.track_csv, xi_amb, wx_df,
        args.window_min, args.dtau_min, pair_radius,
        hs_quantile=args.hs_quantile, track_start_min=args.track_start_min,
        allow_synth=False, infarm_radius_m=args.infarm_radius,
        predictor=args.recovery_predictor, weather_alignment=args.weather_alignment)

    print(f"instance: turbines={len(turbines)} reach={len(reach)} launch_opts={len(opts)} "
          f"horizons={RM.decision_horizons_of(xi_amb)} T={T_eff:.1f}min")
    print(f"pair_radius={pair_radius:.1f}m ({pair_mode}) track={kind}")
    print(f"xi_predictor={getattr(xi_amb,'predictor','unknown')} "
          f"weather_formal={getattr(wamb,'formal_eligible',False)}")

    e1df = _read_existing_e1(args.existing_e1_csv)
    e1audit = _existing_e1_audit(e1df)
    if not e1audit.empty:
        _print_header("A. EXISTING E1 CERTIFICATE / SINGLETON AUDIT")
        print(e1audit.to_string(index=False))
        _save(e1audit, outdir, "existing_e1_audit.csv")
        if "max_stops_observed" in e1df:
            print("existing max_stops_observed distribution:",
                  e1df["max_stops_observed"].value_counts(dropna=False).sort_index().to_dict())
    else:
        print("existing E1 csv not found; continuing with physical diagnostics only:", args.existing_e1_csv)

    _print_header("B. GEOMETRY")
    geom = _geometry_summary(reach, opts)
    print(json.dumps(geom, ensure_ascii=False, indent=2))

    all_scan_rows = []
    all_pool_ledger = []
    pool_summaries = []
    profile_summaries = []
    root_messages = []
    revalidated_rows = []

    for uk in [x.strip() for x in str(args.uavs).split(",") if x.strip()]:
        _print_header(f"C. UAV={uk} MULTI-STOP DIAGNOSTIC")
        p = M.apply_uav_profile(p_base, uk)
        deck_args = SimpleNamespace(t_swap_min=args.t_swap_min, t_launch_min=args.t_launch_min)
        t_swap, t_launch = E._uav_deck(deck_args, uk)
        floor = _profile_floor_summary(p, reach, max(RM.decision_horizons_of(xi_amb)))
        floor["uav"] = uk
        floor["uav_label"] = str(p.uav_label)
        profile_summaries.append(floor)
        print("profile floors:", json.dumps(floor, ensure_ascii=False, indent=2))

        print("\n[C1] build old-style heuristic multi-stop pool for candidate discovery only...")
        ps, ledger, revalidated, resource_feasible = _heuristic_pool_probe(
            uk, reach, opts, p, xi_amb, wamb, args, t_swap, t_launch)
        pool_summaries.append(ps)
        all_pool_ledger.extend(dict(uav=uk, **r) for r in ledger)
        print(json.dumps(ps, ensure_ascii=False, indent=2))

        for c in revalidated:
            revalidated_rows.append(dict(
                uav=uk, ordered_tids=">".join(_ordered_tids(c)),
                stops=len(_ordered_tids(c)), tau=float(c["tau"]), h=float(c["h"]),
                E_plan_Wh=float(c["E_plan_Wh"]), E_soc_required_Wh=float(c["E_soc_required_Wh"]),
                single_route_K1B1_resource_feasible=any(
                    BP._exact_route_signature(c) == BP._exact_route_signature(q)
                    for q in resource_feasible)))

        print("\n[C2] direct formal-physics scan (same route oracle as exact BPC)...")
        scan_rows, direct_cols, scan_complete, evals = _scan_multistop(
            uk, reach, opts, p, xi_amb, wamb, args, t_launch)
        all_scan_rows.extend(scan_rows)
        print("scan evaluations by stop:", evals)
        print("scan complete flags:", scan_complete)
        fsum = _failure_summary(scan_rows)
        if not fsum.empty:
            print(fsum.to_string(index=False))

        direct_resource = []
        for c in direct_cols[:128]:
            audit = BP._audit_integer_selection(
                [c], (0,), 1, 1, p, float(p.quick_inspection_min), float(t_swap),
                int(p.quick_inspection_capacity), int(p.swap_station_capacity), deadline=None)
            if audit.status is RA.ResourceAuditStatus.FEASIBLE:
                direct_resource.append(c)

        exrow = None
        if not e1audit.empty:
            q = e1audit[(e1audit["uav"] == uk) & (e1audit["role"] == "minimal-K1B1")]
            if not q.empty:
                exrow = q.iloc[-1].to_dict()
        msgs = _classify_root_cause(exrow, ps, direct_resource, scan_rows, scan_complete)
        root_messages.extend([f"{uk}: {m}" for m in msgs])
        for m in msgs:
            print("ROOT-CAUSE:", m)

    _print_header("D. AGGREGATED FAILURE SUMMARY")
    scan_df = pd.DataFrame(all_scan_rows)
    fail_df = _failure_summary(all_scan_rows)
    near_df = _near_misses(all_scan_rows, args.near_miss_count)
    pool_df = pd.DataFrame(pool_summaries)
    profile_df = pd.DataFrame(profile_summaries)
    reval_df = pd.DataFrame(revalidated_rows)

    if not fail_df.empty:
        print(fail_df.to_string(index=False))
    if not near_df.empty:
        print("\nnearest multi-stop misses:")
        cols = [c for c in ["uav","stops","ordered_tids","tau","h","primary_reason",
                            "margin_E_Wh","margin_T_s","route_airspeed_margin_ms",
                            "escort_airspeed_margin_ms","E_plan_Wh","E_soc_required_Wh"]
                if c in near_df.columns]
        print(near_df[cols].head(30).to_string(index=False))

    _print_header("E. FINAL ROOT-CAUSE CLASSIFICATION")
    for m in root_messages:
        print("*", m)

    # Static code-path diagnosis: this is intentionally printed even if physical
    # scan times out, because it is a property of the current formal algorithm.
    print("\nSTATIC CODE-PATH FACTS:")
    print("* formal E1 skips the prebuilt multi-stop route pool.")
    print("* step12 _initial_singleton_columns generates singleton routes only.")
    print("* multi-stop formal columns must therefore arrive through exact implicit DFS pricing or later revalidated seeds.")
    print("* exact pricing is exhaustive only when complete; a time-limited incumbent may remain singleton.")
    print("* seed_cols are heuristic only and are revalidated by the physical oracle, so a future multi-stop warm start can be added without changing certificate scope.")

    if all_pool_ledger:
        ledger_df = pd.json_normalize(all_pool_ledger, sep="_")
    else:
        ledger_df = pd.DataFrame()

    _save(profile_df, outdir, "profile_service_floors.csv")
    _save(pool_df, outdir, "heuristic_pool_summary.csv")
    _save(reval_df, outdir, "revalidated_multistop_seeds.csv")
    _save(scan_df, outdir, "direct_physical_scan.csv")
    _save(fail_df, outdir, "failure_summary.csv")
    _save(near_df, outdir, "near_miss_multistop_routes.csv")
    if not ledger_df.empty:
        _save(ledger_df, outdir, "heuristic_route_ledger.csv")

    summary = dict(
        diagnostic_contract=DIAGNOSTIC_CONTRACT,
        formal_model_unchanged=True,
        certificate_emitted=False,
        instance=dict(
            n_turbines=len(turbines), reach=len(reach), launch_options=len(opts),
            horizons=list(map(float, RM.decision_horizons_of(xi_amb))),
            T_eff_min=float(T_eff), pair_radius_m=float(pair_radius),
            pair_radius_mode=str(pair_mode), track_kind=str(kind),
            recovery_predictor=str(args.recovery_predictor),
            soc_correction=str(args.soc_correction),
            soc_risk_allocation=str(args.soc_risk_allocation),
            time_recourse=str(args.time_recourse),
            max_stops=int(args.max_stops)),
        geometry=geom,
        pool_summary=pool_summaries,
        root_cause_messages=root_messages,
        output_directory=str(outdir))
    (outdir / "diagnosis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nOUTPUT DIRECTORY:", outdir)
    print("请把 diagnosis_summary.json、failure_summary.csv、"
          "heuristic_pool_summary.csv、revalidated_multistop_seeds.csv 和终端完整输出发给我。")


if __name__ == "__main__":
    main()
