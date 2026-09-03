#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step19_diagnose_formal_bottlenecks.py

Diagnostic-only follow-up to Step18.  It never changes the formal model and
never emits an optimization certificate.

It re-evaluates only the closest 2-stop near-misses and prints:
  - exact full time/speed-recourse certificate decomposition,
  - pooled-ALL vs selected-vessel Xi sample diagnostics,
  - diagnostic counterfactuals (Xi-only, no-geo2d, Gaussian/nominal risk),
  - mission-budget-preserving eps_E/eps_T reallocations,
  - horizon-support and UAV-speed-envelope warnings,
  - all currently observed E1 certificate/validation anomalies.

Counterfactual rows are explicitly NON-FORMAL and must never be used as final
results.  They exist only to identify which modelling/data component is
responsible for current singleton-only solutions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import step9_model as M
import step10_model_routing as RM
import step12_branch_price as BP
import step13_experiment_model as E
import step15_replay as RP

HERE = Path(__file__).resolve().parent
DEFAULT_DIAG = HERE / "results" / "model_experiments" / "multistop_diagnostics"
DEFAULT_OUT = HERE / "results" / "model_experiments" / "formal_bottleneck_diagnostics"
CONTRACT = "formal-bottleneck-diagnostic-v1"
FORMAL_RISK_MODE = "vp_unimodal"


def _finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _route_diag(route, h, p, wx, xi_amb, wamb, risk_mode=FORMAL_RISK_MODE):
    rp = RM.risk_policy_for_mode(risk_mode)
    return RM.route_feasible_at_h(
        route, int(h), p, wx, xi_amb, weather_unc=wamb,
        chance_mode="drcc", budget_gamma=2.0, formal=False, risk_policy=rp)


def _extract(dd, variant, uav, tids, tau, h, formal_eligible=False):
    td = dd.get("time_decomposition") or {}
    gd = td.get("geo_detail") or {}
    return dict(
        uav=uav, ordered_tids=">".join(tids), tau=float(tau), h=float(h),
        variant=variant, diagnostic_only=(not formal_eligible),
        feasible=bool(dd.get("feasible", False)),
        reason=str(dd.get("primary_reason", dd.get("reason", "unknown"))),
        margin_E_Wh=_finite(dd.get("margin_E")),
        margin_T_equiv_s=_finite(dd.get("time_drcc_margin_s", dd.get("margin_T"))),
        nominal_time_margin_equiv_s=_finite(dd.get("nominal_time_margin_s")),
        time_tightening_equiv_s=_finite(dd.get("time_drcc_tightening_s")),
        return_time_budget_s=_finite(dd.get("return_time_budget_s")),
        return_time_budget_safe_s=_finite(dd.get("return_time_budget_safe_s")),
        nonreturn_weather_mean_shift_s=_finite(dd.get("nonreturn_weather_mean_shift_s")),
        nonreturn_weather_std_term_s=_finite(dd.get("nonreturn_weather_std_term_s")),
        nonreturn_weather_reserve_s=_finite(dd.get("nonreturn_weather_reserve_s")),
        eps_time_total=_finite(dd.get("eps_time_total")),
        eps_time_nonreturn_weather=_finite(dd.get("eps_time_nonreturn_weather")),
        eps_time_return_required_airspeed=_finite(dd.get("eps_time_return_required_airspeed")),
        eps_time_along=_finite(dd.get("eps_time_along")),
        eps_time_cross=_finite(dd.get("eps_time_cross")),
        nominal_required_airspeed_ms=_finite(dd.get("return_required_airspeed_nom_ms")),
        safe_required_airspeed_ms=_finite(dd.get("return_required_airspeed_safe_ms")),
        return_airspeed_margin_ms=_finite(dd.get("return_airspeed_margin_ms")),
        route_airspeed_margin_ms=_finite((dd.get("margins") or {}).get("route_airspeed_ms")),
        escort_airspeed_margin_ms=_finite(dd.get("escort_margin_ms")),
        E_plan_Wh=_finite(dd.get("E_plan_Wh")),
        E_soc_required_Wh=_finite(dd.get("E_soc_required_Wh")),
        xi_mu_e_m=_finite(dd.get("xi_mu_e_m")),
        xi_mu_n_m=_finite(dd.get("xi_mu_n_m")),
        xi_sigma_ee_m2=_finite(dd.get("xi_sigma_ee_m2")),
        xi_sigma_en_m2=_finite(dd.get("xi_sigma_en_m2")),
        xi_sigma_nn_m2=_finite(dd.get("xi_sigma_nn_m2")),
        combined_required_airspeed_mean_ms=json.dumps(
            td.get("combined_required_airspeed_mean_ms",
                   dd.get("time_decomposition_combined_required_airspeed_mean_ms")),
            ensure_ascii=False),
        geo_bound_ms=_finite(td.get("geo_detail_bound_m")),
        geo_std_along_ms=_finite(td.get("geo_detail_std_along_m")),
        geo_std_cross_ms=_finite(td.get("geo_detail_std_cross_m")),
        geo_kappa_along=_finite(td.get("geo_detail_kappa_along")),
        geo_kappa_cross=_finite(td.get("geo_detail_kappa_cross")),
        geo_eps_along=_finite(td.get("geo_detail_eps_along")),
        geo_eps_cross=_finite(td.get("geo_detail_eps_cross")),
        failure_flags=json.dumps(
            {k: bool(v) for k, v in (dd.get("failure_flags") or {}).items() if bool(v)},
            ensure_ascii=False, sort_keys=True),
    )


def _mmsi_audit(train_df, selected):
    out = {}
    if "mmsi" not in train_df.columns:
        return dict(mmsi_column=False, selected_mmsi=selected)
    vals = train_df["mmsi"].astype(str)
    uniq = sorted(vals.unique())
    out.update(mmsi_column=True, selected_mmsi=selected,
               unique_mmsi_count=len(uniq),
               selected_present=bool(selected in set(uniq)),
               sample_rows=int(len(train_df)))
    pooled = train_df[["xi_e_m", "xi_n_m"]].astype(float)
    cov = np.cov(pooled.to_numpy().T, ddof=1) if len(pooled) > 1 else np.full((2,2), np.nan)
    out["pooled_cov_trace_m2"] = float(np.trace(cov))
    if selected in set(uniq):
        d = train_df[vals == selected].copy()
        out["selected_rows"] = int(len(d))
        counts = d.groupby(["h_min", "c_state"]).size()
        out["selected_cell_count"] = int(len(counts))
        out["selected_cell_min_n"] = int(counts.min()) if len(counts) else 0
        out["selected_cell_max_n"] = int(counts.max()) if len(counts) else 0
        if len(d) > 1:
            covs = np.cov(d[["xi_e_m","xi_n_m"]].astype(float).to_numpy().T, ddof=1)
            out["selected_cov_trace_m2"] = float(np.trace(covs))
            out["selected_to_pooled_cov_trace_ratio"] = float(np.trace(covs) / np.trace(cov)) if np.trace(cov) > 0 else None
    return out


def parser():
    ap = argparse.ArgumentParser(description="Step19 formal bottleneck decomposition (diagnostic only)")
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
    ap.add_argument("--weather-alignment", choices=["timestamp","representative_quantile"], default="timestamp")
    ap.add_argument("--recovery-predictor", choices=["cv_noleak","true_track"], default="cv_noleak")
    ap.add_argument("--soc-correction", choices=["none","geo2d"], default="geo2d")
    ap.add_argument("--soc-risk-allocation", choices=["fixed","optimized"], default="optimized")
    ap.add_argument("--time-recourse", choices=["wait_only","wait_and_speed"], default="wait_and_speed")
    ap.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAG)
    ap.add_argument("--e1-csv", type=Path,
                    default=HERE/"results"/"model_experiments"/"E1_frontier"/"E1_frontier.csv")
    ap.add_argument("--top-per-uav", type=int, default=3)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    return ap


def main():
    args = parser().parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print("="*100)
    print("STEP19 / FORMAL BOTTLENECK DECOMPOSITION")
    print("contract:", CONTRACT)
    print("ALL counterfactual variants are diagnostic-only and cannot certify.")
    print("="*100)

    near_path = Path(args.diagnostics_dir)/"near_miss_multistop_routes.csv"
    direct_path = Path(args.diagnostics_dir)/"direct_physical_scan.csv"
    if not direct_path.is_file():
        raise SystemExit(f"missing Step18 output: {direct_path}")
    direct = pd.read_csv(direct_path, low_memory=False)
    for c in ("margin_E_Wh","margin_T_s","nominal_time_margin_s","time_drcc_tightening_s"):
        direct[c] = pd.to_numeric(direct[c], errors="coerce")

    turbines, wx_df, _, lat0lon0, _, _, _ = E.load_all(
        args.n_turbines, farm=args.farm, allow_synth=False,
        turbines_csv=args.turbines_csv, wind_csv=args.wind_csv, wave_csv=args.wave_csv,
        xi_moments_csv=args.xi_moments_csv,
        recovery_scenarios_csv=args.recovery_scenarios_csv,
        track_csv=args.track_csv, require_xi_moments=False)

    selected_mmsi = E._infer_concrete_track_mmsi(args.track_csv, None, formal=True)
    train_all = RP.load_samples(
        args.xi_train_samples, mmsi="ALL", formal=True, expected_split="train")
    train_df = RP.load_samples(
        args.xi_train_samples, mmsi=selected_mmsi, formal=True, expected_split="train")
    hs = sorted(train_df["h_min"].astype(int).unique())
    states = sorted(train_df["c_state"].astype(str).unique())
    xi_amb = RP.ambiguity_from_samples(train_df, hs, states, formal=True)
    wamb = RM.weather_ambiguity_from_moments_csv(
        args.weather_moments_csv, RM.decision_horizons_of(xi_amb), formal=True)

    p0 = M.Params()
    p0.soc_correction = args.soc_correction
    p0.soc_risk_allocation = args.soc_risk_allocation
    p0.time_recourse_mode = args.time_recourse
    p0.speed_adjustable = (args.time_recourse == "wait_and_speed")
    p0.validate_contract(formal=True)

    pair_args = SimpleNamespace(pair_radius=args.pair_radius, e1_uavs=args.uavs,
                                uav=str(args.uavs).split(",")[0].strip())
    pair_radius, pair_mode = E._resolve_pair_radius(pair_args, p0, xi_amb, turbines)
    opts, reach, kind, T_eff, _ = E.build_launch_options(
        turbines, lat0lon0, args.track_csv, xi_amb, wx_df,
        args.window_min, args.dtau_min, pair_radius,
        track_start_min=args.track_start_min, allow_synth=False,
        infarm_radius_m=args.infarm_radius, predictor=args.recovery_predictor,
        weather_alignment=args.weather_alignment)

    ma = _mmsi_audit(train_all, selected_mmsi)
    print("\n[MMSI / XI AUDIT]")
    print(json.dumps(ma, ensure_ascii=False, indent=2))

    tidmap = {str(getattr(t,"tid",t)): t for t in reach}
    variants = []
    chosen = []
    for uk in [x.strip() for x in args.uavs.split(",") if x.strip()]:
        g = direct[(direct["uav"].astype(str)==uk) & (direct["stops"]==2)].copy()
        g = g[np.isfinite(g["margin_T_s"]) & (g["margin_T_s"] > -1e20)]
        if g.empty:
            continue
        g["_joint"] = np.maximum(0,-g["margin_T_s"])/60.0 + np.maximum(0,-g["margin_E_Wh"])/10.0
        cands = []
        cands.append(g.sort_values("_joint").iloc[0])
        ge = g[g["margin_E_Wh"] >= 0]
        if not ge.empty:
            cands.append(ge.sort_values("margin_T_s", ascending=False).iloc[0])
        cands.append(g.sort_values("margin_T_s", ascending=False).iloc[0])
        seen=set()
        for r in cands:
            key=(str(r["ordered_tids"]),float(r["tau"]),float(r["h"]))
            if key in seen: continue
            seen.add(key); chosen.append((uk,r))

    for uk,r in chosen:
        tids = str(r["ordered_tids"]).split(">")
        seq = [tidmap[x] for x in tids]
        tau = float(r["tau"]); h=int(float(r["h"]))
        oi,opt = min(enumerate(opts), key=lambda q: abs(float(q[1].tau_min)-tau))
        if abs(float(opt.tau_min)-tau) > 1e-9:
            raise RuntimeError(f"cannot map tau={tau} to launch option")
        route = RM.Route(rid=-1, turbines=seq, ship=opt.ship)
        p = M.apply_uav_profile(p0, uk)

        tests = []
        tests.append(("formal_full_vp_geo2d", p, wamb, "vp_unimodal", True))

        px = copy.deepcopy(p)
        tests.append(("diag_xi_only_no_weather", px, None, "vp_unimodal", False))

        png = copy.deepcopy(p); png.soc_correction="none"
        tests.append(("diag_no_geo2d", png, wamb, "vp_unimodal", False))

        tests.append(("diag_gaussian", copy.deepcopy(p), wamb, "gaussian", False))
        tests.append(("diag_nominal_risk", copy.deepcopy(p), wamb, "nominal", False))

        # Preserve or fully use the declared 5% mission union budget.
        # Other active fixed events consume 0.020, so eps_E + eps_T may be <= 0.030.
        allocs = [(0.0125,0.0125), (0.0100,0.0150), (0.0075,0.0175),
                  (0.0050,0.0200), (0.0050,0.0250)]
        for eE,eT in allocs:
            pr = copy.deepcopy(p)
            pr.eps_E=float(eE); pr.eps_T=float(eT)
            tests.append((f"diag_budget_realloc_E{eE:.4f}_T{eT:.4f}",pr,wamb,"vp_unimodal",False))

        print(f"\n[CANDIDATE] {uk} {'>'.join(tids)} tau={tau} h={h}")
        for name,pv,wv,rm,formal_flag in tests:
            try:
                dd=_route_diag(route,h,pv,opt.wx,xi_amb,wv,rm)
                row=_extract(dd,name,uk,tids,tau,h,formal_flag)
            except Exception as exc:
                row=dict(uav=uk,ordered_tids=">".join(tids),tau=tau,h=h,
                         variant=name,diagnostic_only=True,feasible=False,
                         reason=f"ERROR:{type(exc).__name__}:{exc}")
            variants.append(row)
            print(name,
                  "feasible=",row.get("feasible"),
                  "E=",row.get("margin_E_Wh"),
                  "T(eq)=",row.get("margin_T_equiv_s"),
                  "v_safe=",row.get("safe_required_airspeed_ms"),
                  "epsT=",row.get("eps_time_total"),
                  "reason=",row.get("reason"))

    vdf=pd.DataFrame(variants)
    vdf.to_csv(out/"candidate_counterfactuals.csv",index=False,encoding="utf-8-sig")

    # E1 systemic audit
    e1issues=[]
    if Path(args.e1_csv).is_file():
        e1=pd.read_csv(args.e1_csv,low_memory=False)
        pos=e1[pd.to_numeric(e1.get("covered"),errors="coerce")>0].copy()
        if len(pos):
            if "emp_viol" in pos:
                vals=sorted(pd.to_numeric(pos["emp_viol"],errors="coerce").dropna().unique().tolist())
                if len(vals)==1:
                    e1issues.append(f"all {len(pos)} positive-coverage E1 rows share identical emp_viol={vals[0]}; audit replay aggregation.")
            if "max_stops_observed" in pos and int(pd.to_numeric(pos["max_stops_observed"],errors="coerce").max())==1:
                e1issues.append("all positive-coverage E1 incumbents are singleton.")
            if "coverage_global_certificate_available" in e1 and not bool(e1["coverage_global_certificate_available"].astype(str).str.lower().eq("true").any()):
                e1issues.append("no E1 row has a physical coverage global certificate.")
            if "coverage_upper_bound" in e1:
                ub=pd.to_numeric(e1["coverage_upper_bound"],errors="coerce").dropna()
                if len(ub) and ub.nunique()==1:
                    e1issues.append(f"coverage upper bound is constant at {ub.iloc[0]} across reported rows; bound is non-discriminating.")
    # Same speed envelope is deliberate but makes time feasibility identical by construction.
    speeds={}
    for uk in [x.strip() for x in args.uavs.split(",") if x.strip()]:
        pp=M.apply_uav_profile(p0,uk)
        speeds[uk]=dict(v_air_max=float(pp.v_air_max),v_cr=float(pp.v_cr),v_max=float(pp.v_max))
    if len({tuple(v.values()) for v in speeds.values()})==1:
        e1issues.append("S/M/L share the same cruise/max airspeed envelope by design; time-DRCC feasibility is therefore nearly identical across UAV types.")

    if max(hs) <= 30:
        e1issues.append(f"Xi formal horizon support stops at {max(hs)} min; no formal route may use a longer recovery horizon without regenerating data.")

    print("\n[SYSTEMIC ISSUES]")
    for x in e1issues:
        print("*",x)

    summary=dict(
        contract=CONTRACT, certificate_emitted=False,
        selected_track_mmsi=selected_mmsi, mmsi_audit=ma,
        horizons=hs, max_formal_horizon_min=max(hs),
        uav_speed_envelopes=speeds,
        systemic_issues=e1issues,
        evaluated_candidates=len(chosen),
        output_directory=str(out))
    (out/"step19_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nOUTPUT:",out)
    print("Send step19_summary.json + candidate_counterfactuals.csv + console output back to ChatGPT.")


if __name__=="__main__":
    main()
