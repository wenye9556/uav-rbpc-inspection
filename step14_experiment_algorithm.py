#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step14_experiment_algorithm.py — 最终资源模型算法实验。

A1 比较研究用贪心、研究用受限列池主问题与正式精确
Branch-Price-and-Cut + Logic-Based Benders。A2 比较正式按需列生成算法与受限列池
研究基线的运行时间和解质量。受限列池基线始终标记
``global_certificate_available=False``，不得把其 MIP Gap 当作隐式全路线模型 Gap。

数据默认 fail-closed；``--allow-synth`` 仅用于机制调试，不得形成真实业务结论。
"""
from __future__ import annotations

import argparse
import logging
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

import step9_model as M
import step10_model_routing as RM
import step11_algorithm_route_drcc as RA
import step12_branch_price as BP
import step13_experiment_model as S13
EU = M  # shared utilities are merged into step9_model to preserve package layout

log = logging.getLogger("expA")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
RESULTS = Path(__file__).resolve().parent / "results" / "algorithm_experiments"


def _save(df, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / name, index=False, encoding="utf-8-sig")
    log.info("写 %s (%d 行)", outdir / name, len(df))


def _validate_formal_base_parity(args):
    """Fail closed when formal A1/A2 diverges from Step13's finite-instance contract."""
    if str(getattr(args, "study_mode", "formal")).lower() != "formal":
        return
    required = {
        "weather_drcc": "on",
        "recovery_predictor": "cv_noleak",
        "pool_h": "pareto",
        "soc_correction": "geo2d",
        "deck_mode": "interval",
    }
    bad = []
    for name, expected in required.items():
        actual = str(getattr(args, name, expected))
        if actual != expected:
            bad.append(f"--{name.replace('_', '-')} {actual!r} (required {expected!r})")
    if bad:
        raise SystemExit("formal A1/A2 必须与 Step13 formal finite instance 同口径: " + "; ".join(bad))


def _setup(n, args, dtau=None):
    """Build the exact same formal base instance semantics used by Step13."""
    formal = (str(getattr(args, "study_mode", "formal")).lower() == "formal")
    _validate_formal_base_parity(args)
    if formal and args.allow_synth:
        raise SystemExit("formal A1/A2 禁止 --allow-synth。")
    if args.track_csv is None and args.track_mmsi:
        args.track_csv = str(
            Path(__file__).resolve().parent / "tracks" / f"track_{args.track_mmsi}.csv")

    turbines, wx_df, xi_amb, lat0lon0, sc_csv, src, track_csv = S13.load_all(
        n, farm=args.farm, allow_synth=args.allow_synth,
        turbines_csv=args.turbines_csv, wind_csv=args.wind_csv, wave_csv=args.wave_csv,
        xi_moments_csv=args.xi_moments_csv,
        recovery_scenarios_csv=args.recovery_scenarios_csv,
        track_csv=args.track_csv,
        require_xi_moments=(not formal and args.xi_train_samples is None))
    if args.track_csv:
        track_csv = Path(args.track_csv)

    selected_mmsi = S13._infer_concrete_track_mmsi(
        track_csv, getattr(args, "track_mmsi", None), formal=formal)
    if formal:
        if args.xi_train_samples is None:
            raise SystemExit("formal A1/A2 必须提供 --xi-train-samples；禁止使用 mmsi=ALL moments。")
        _train_df, xi_amb = S13._xi_ambiguity_from_train_samples(
            args.xi_train_samples, selected_mmsi, formal=True)
        src = (f"purged-train:{Path(args.xi_train_samples).name}:"
               f"mmsi={selected_mmsi}:{M.sha256_file(args.xi_train_samples)}")

    p = M.apply_uav_profile(M.Params(), args.uav)
    p.soc_correction = getattr(args, "soc_correction", "geo2d")
    p.soc_risk_allocation = getattr(args, "soc_risk_allocation", "optimized")
    p.time_recourse_mode = str(getattr(args, "time_recourse", "wait_and_speed"))
    p.speed_adjustable = (p.time_recourse_mode == "wait_and_speed")
    p.validate_contract(formal=formal)
    t_swap, t_launch = S13._uav_deck(args, args.uav)

    if args.weather_drcc == "on":
        if args.weather_moments_csv is not None:
            wamb = RM.weather_ambiguity_from_moments_csv(
                args.weather_moments_csv, RM.decision_horizons_of(xi_amb), formal=formal)
            S13._assert_weather_xi_train_binding(
                wamb, args.xi_train_samples, formal=formal)
        elif formal:
            raise SystemExit(
                "formal A1/A2 weather-drcc=on 必须提供 --weather-moments-csv。")
        elif args.allow_synth:
            wamb = RM.weather_ambiguity_from_series(
                wx_df, RM.decision_horizons_of(xi_amb), scale=1.0)
        else:
            raise SystemExit("算法实验 weather-drcc=on 缺少 --weather-moments-csv。")
    else:
        wamb = None

    if not hasattr(args, "e1_uavs"):
        args.e1_uavs = str(args.uav)
    _pr, _pr_mode = S13._resolve_pair_radius(args, M.Params(), xi_amb, turbines)
    args._pair_radius_m, args._pair_radius_mode, args._xi_source = _pr, _pr_mode, src
    opts, reach, kind, T_eff, wx0 = S13.build_launch_options(
        turbines, lat0lon0, track_csv, xi_amb, wx_df, args.window_min,
        dtau if dtau is not None else args.dtau_min, _pr,
        hs_quantile=args.hs_quantile, track_start_min=args.track_start_min,
        allow_synth=args.allow_synth,
        predictor=getattr(args, "recovery_predictor", "cv_noleak"),
        weather_alignment=getattr(args, "weather_alignment", "timestamp"),
        formal=formal, bound_track_mmsi=(selected_mmsi if formal else None))

    if formal:
        selected_window = str(wx0.get("selected_track_mmsi") or "").strip()
        if selected_window != str(selected_mmsi):
            raise SystemExit(
                "formal A1/A2 轨迹窗口 MMSI 与预绑定 MMSI 不一致: "
                f"window={selected_window!r}, expected={selected_mmsi!r}")
        if str(getattr(xi_amb, "selected_mmsi", "")) != str(selected_mmsi):
            raise SystemExit("formal A1/A2 Xi ambiguity 未绑定当前轨迹 MMSI。")
        if bool(getattr(xi_amb, "cross_vessel_pooling", False)):
            raise SystemExit("formal A1/A2 禁止 cross-vessel pooled Xi。")

    args._preselected_track_mmsi = str(selected_mmsi or "ALL")
    S13._record_formal_instance_provenance(
        args, mmsi=selected_mmsi, track_csv=track_csv,
        xi_train_samples=args.xi_train_samples,
        weather_moments_csv=args.weather_moments_csv, weather_uncertainty=wamb,
        launch_formal=formal)
    if formal:
        log.info(
            "formal A instance: mmsi=%s track_sha=%s xi_train_sha=%s weather_sha=%s "
            "weather_predictor_contract=%s launch_formal=True",
            args._formal_instance_mmsi,
            args._formal_instance_track_sha256,
            args._formal_instance_xi_train_sha256,
            args._formal_instance_weather_moments_sha256,
            args._formal_instance_weather_predictor_contract)
    args._last_wx0 = wx0
    return p, xi_amb, wamb, opts, reach, T_eff, kind, t_swap, t_launch


def _final_resource_kwargs(args, t_swap, t_launch, max_stops, wamb):
    return dict(deck_delta_min=args.deck_delta_min, t_swap_min=t_swap,
                landing_clear_min=args.landing_clear_min,
                quick_inspection_capacity=args.quick_inspection_capacity,
                swap_station_capacity=args.swap_stations,
                max_stops=max_stops, weather_unc=wamb,
                batteries=args.batteries, deck_mode=args.deck_mode,
                t_launch_min=t_launch, pool_h_mode=getattr(args, "pool_h", "pareto"),
                time_limit_s=args.time_limit_s,
                coverage_gap_target_abs=args.coverage_gap_target_abs,
                energy_gap_target_rel=args.energy_gap_target_rel,
                energy_gap_target_abs_Wh=args.energy_gap_target_abs_wh)



def _global_gap_fields(rr):
    """CSV-safe bounds and fail-closed normalized certificate evidence."""
    certificate_fields = S13._canonical_certificate_fields(rr)
    return dict(
        coverage_incumbent=rr.get("coverage_incumbent", rr.get("covered")),
        coverage_upper_bound=rr.get("coverage_upper_bound"),
        coverage_gap_abs=rr.get("coverage_gap_abs"),
        coverage_gap_pct=rr.get("coverage_gap_pct"),
        energy_incumbent_Wh=rr.get("energy_incumbent_Wh", rr.get("energy_Wh")),
        energy_lower_bound_Wh=rr.get("energy_lower_bound_Wh"),
        energy_gap_abs_Wh=rr.get("energy_gap_abs_Wh"),
        energy_gap_pct=rr.get("energy_gap_pct"),
        conditional_energy_gap_pct=rr.get("conditional_energy_gap_pct"),
        global_energy_gap_reason=rr.get("global_energy_gap_reason"),
        pricing_complete=rr.get("pricing_complete"),
        pricing_bound_available=rr.get("pricing_bound_available"),
        branching_complete=rr.get("branching_complete"),
        farkas_pricing_complete=rr.get("farkas_pricing_complete"),
        resource_audit_complete=rr.get("resource_audit_complete"),
        bound_scope=rr.get("bound_scope"), bound_source=rr.get("bound_source"),
        restricted_pool_gap_pct=rr.get("restricted_pool_gap_pct"),
        **certificate_fields)


def _solution_detail(rr):
    """Canonical, CSV-safe final-resource assignment detail for A1 cross-checking."""
    missions = []
    for c in rr.get("chosen", []) or []:
        route = c.get("route")
        ordered = list(route.turbine_ids()) if route is not None else list(c.get("ordered_tids", c.get("tids", ())))
        missions.append(dict(
            tau=round(float(c.get("tau", 0.0)), 9), h=round(float(c.get("h", 0.0)), 9),
            ordered_tids=[str(x) for x in ordered],
            E_plan_Wh=round(float(c.get("E_plan_Wh", c.get("E0", 0.0))), 9),
            E_soc_required_Wh=round(float(c.get("E_soc_required_Wh", 0.0)), 9),
            uav_id=c.get("uav_id"), battery_group=c.get("battery_group"),
            turnaround_before=c.get("turnaround_before"),
            post_service_mode=c.get("post_service_mode"),
            post_service_interval=c.get("post_service_interval")))
    missions.sort(key=lambda x: (x["tau"], x["h"], x["ordered_tids"]))
    detail = dict(missions=missions, battery_energy_used_Wh=rr.get("battery_energy_used_Wh", []),
                  battery_end_soc_pct=rr.get("battery_end_soc_pct", []),
                  quick_inspection_events=rr.get("quick_inspection_events", []),
                  swap_events=rr.get("swap_events", []),
                  resource_cuts=rr.get("resource_cuts", 0))
    return json.dumps(detail, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def A1_accuracy(args, outdir):
    """Compare three solvers on the *same final resource model*.

    The first two arms are research baselines on a prebuilt route pool.  The
    ``exact_branch_price_cut`` arm is the only formal global-certificate path.
    """
    methods = ("research_greedy", "research_restricted_pool", "exact_branch_price_cut")
    rows = []
    signature_params = M.apply_uav_profile(M.Params(), args.uav)
    signature_params.time_recourse_mode = str(getattr(args, "time_recourse", "wait_and_speed"))
    signature_params.speed_adjustable = (signature_params.time_recourse_mode == "wait_and_speed")
    _sig = dict(uav=args.uav, K=int(args.k), batteries=int(args.batteries),
                max_stops=int(args.max_stops), deck_mode=args.deck_mode,
                quick_inspection_capacity=int(args.quick_inspection_capacity),
                swap_stations=int(args.swap_stations),
                predictor=getattr(args, "recovery_predictor", "cv_noleak"),
                pool_h=getattr(args, "pool_h", "pareto"),
                soc_correction=getattr(args, "soc_correction", "geo2d"),
                soc_risk_allocation=getattr(args, "soc_risk_allocation", "optimized"),
                    time_recourse_mode=getattr(args, "time_recourse", "wait_and_speed"),
                    time_contract_id=RM.time_contract_for(signature_params),
                    speed_is_recourse=bool(getattr(signature_params, "speed_adjustable", False)),
                    return_speed_recourse_contract=(RM.SPEED_RECOURSE_CONTRACT if getattr(signature_params, "speed_adjustable", False) else None),
                geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
                pair_radius_request=str(args.pair_radius),
                window_min=float(args.window_min), dtau_min=float(args.dtau_min),
                pricing_mode=str(getattr(args, "pricing_mode", "exact-implicit-dfs")), result_contract=S13.RESULT_CONTRACT)
    rows, _done = S13._resume_load(outdir, "A1_accuracy.csv", ["n_turbines", "method"],
                                   _sig, getattr(args, "resume", "on"))
    for n in [int(x) for x in str(args.n_list).split(",") if x.strip()]:
        if all((str(n), m) in _done for m in methods):
            continue
        p, xi_amb, wamb, opts, reach, T_eff, kind, t_swap, t_launch = _setup(n, args)
        max_stops = int(args.max_stops)
        kw = _final_resource_kwargs(args, t_swap, t_launch, max_stops, wamb)
        base = dict(n_turbines=n, reach=len(reach), n_slots=len(opts), track=kind,
                    uav=args.uav, K=args.k, T_min=round(T_eff, 1),
                    max_stops=max_stops, deck_mode=args.deck_mode,
                    quick_inspection_capacity=args.quick_inspection_capacity,
                    swap_stations=args.swap_stations,
                    predictor=getattr(args, "recovery_predictor", "cv_noleak"),
                    pool_h=getattr(args, "pool_h", "pareto"),
                    soc_correction=getattr(args, "soc_correction", "geo2d"),
                    soc_risk_allocation=getattr(args, "soc_risk_allocation", "optimized"),
                    time_recourse_mode=getattr(args, "time_recourse", "wait_and_speed"),
                    time_contract_id=RM.time_contract_for(p),
                    speed_is_recourse=bool(getattr(p, "speed_adjustable", False)),
                    return_speed_recourse_contract=(RM.SPEED_RECOURSE_CONTRACT if getattr(p, "speed_adjustable", False) else None),
                    geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
                    pair_radius_m=round(float(getattr(args, "_pair_radius_m", -1.0)), 3),
                    dtau_min=float(args.dtau_min), result_contract=S13.RESULT_CONTRACT)
        t0 = time.time()
        cols = RA.build_route_columns(reach, opts, p, xi_amb, T_eff, args.deck_delta_min,
                                 max_stops, wamb, "drcc", 2.0, "vp_unimodal", 8.0,
                                 pool_h_mode=getattr(args, "pool_h", "pareto"))
        pool_generation_s = time.time() - t0
        t0 = time.time()
        greedy = RA.solve_resource_master(reach, opts, p, xi_amb, args.k, T_eff,
                                   cols_override=cols, solver="greedy", **kw)
        tg = time.time() - t0 + pool_generation_s
        t0 = time.time()
        restricted = RA.solve_resource_master(reach, opts, p, xi_amb, args.k, T_eff,
                                       cols_override=cols, solver="auto", **kw)
        tr = time.time() - t0 + pool_generation_s
        t0 = time.time()
        exact = BP.solve_fleet_anytime(
            reach, opts, p, xi_amb, args.k, T_eff,
            solver_mode="exact-branch-price-cut",
            pricing_mode=getattr(args, "pricing_mode", "exact-implicit-dfs"),
            solver="auto", **kw)
        te = time.time() - t0
        greedy.update(global_certificate_available=False,
                      global_route_space_certificate=False,
                      implicit_route_space_certified=False,
                      bound_scope="validated_route_pool",
                      algorithm="research-baseline-greedy")
        restricted.update(global_certificate_available=False,
                          global_route_space_certificate=False,
                          implicit_route_space_certified=False,
                          bound_scope="validated_route_pool",
                          algorithm="research-baseline-restricted-pool")
        results = {"research_greedy": (greedy, tg),
                   "research_restricted_pool": (restricted, tr),
                   "exact_branch_price_cut": (exact, te)}
        certified = bool(exact.get("lexicographic_optimal", False))
        if certified:
            best_cov = int(exact["covered"]); best_energy = float(exact["energy_Wh"])
        else:
            best_cov = max(int(rr[0].get("covered", 0)) for rr in results.values())
            best_energy = min(float(rr[0].get("energy_Wh", float("inf")))
                              for rr in results.values() if int(rr[0].get("covered", 0)) == best_cov)
        for meth in methods:
            rr, tt = results[meth]
            cov = int(rr.get("covered", 0)); energy = float(rr.get("energy_Wh", 0.0))
            egap = (100.0 * max(energy - best_energy, 0.0) / max(best_energy, 1e-9)
                    if cov == best_cov and best_energy < float("inf") else None)
            enum = rr.get("enumeration") or {}
            rows.append(dict(**base, method=meth, batteries=rr.get("batteries", args.batteries),
                             covered=cov, energy_Wh=energy,
                             mean_stops=rr.get("mean_stops"),
                             multi_stop_ratio=rr.get("multi_stop_ratio"),
                             time_s=round(float(tt), 3), pool_generation_s=round(pool_generation_s, 3),
                             best_known=best_cov, gap_to_best=best_cov-cov,
                             best_energy_at_best_coverage=(round(best_energy, 6)
                                                          if best_energy < float("inf") else None),
                             energy_gap_to_best_pct=(round(egap, 6) if egap is not None else None),
                             status=rr.get("status", "-"),
                             coverage_optimal=bool(rr.get("coverage_optimal", False)),
                             energy_optimal=bool(rr.get("energy_optimal", False)),
                             lexicographic_certified=bool(rr.get("lexicographic_certified", False)),
                             lexicographic_optimal=bool(
                                 rr.get("lexicographic_optimal", False)),
                             restricted_master_certificate=rr.get("restricted_master_certificate"),
                             resource_cuts=int(rr.get("resource_cuts", 0) or 0),
                             n_swaps=int(rr.get("n_swaps", 0) or 0),
                             n_quick_reuses=int(rr.get("n_quick_reuses", 0) or 0),
                             selected_routes_and_resources=_solution_detail(rr),
                             battery_end_soc_pct=json.dumps(rr.get("battery_end_soc_pct", [])),
                             sequence_upper_bound=enum.get("sequence_upper_bound"),
                             enumeration_complete=enum.get("complete"),
                             pool=rr.get("pool_size", enum.get("retained_columns", enum.get("deduplicated_columns", len(cols)))),
                             solver=str(rr.get("solver", ""))[:80],
                             **_global_gap_fields(rr)))
            _done.add((str(n), meth))
        _save(pd.DataFrame(rows), outdir, "A1_accuracy.csv")
        EU.write_run_manifest(
            outdir, "A1_accuracy", args,
            input_paths=[x for x in [getattr(args, "xi_train_samples", None),
                                      getattr(args, "track_csv", None),
                                      getattr(args, "weather_moments_csv", None)] if x is not None],
            extra={"completed_rows": len(rows), "result_contract": S13.RESULT_CONTRACT,
                   "final_resource_model": True, **S13._formal_instance_manifest_extra(args)})
    df = pd.DataFrame(rows); _save(df, outdir, "A1_accuracy.csv"); return df


def A2_speed(args, outdir):
    """Runtime/quality comparison within the final model.

    The formal arm generates columns on demand.  The comparison arm solves a
    prebuilt validated route pool and is always a research baseline.  Runtime ratios are
    reported only when the formal arm has a valid finite-discrete global certificate and
    the research baseline matches its lexicographic objective.
    """
    rows = []
    signature_params = M.apply_uav_profile(M.Params(), args.uav)
    signature_params.time_recourse_mode = str(getattr(args, "time_recourse", "wait_and_speed"))
    signature_params.speed_adjustable = (signature_params.time_recourse_mode == "wait_and_speed")
    _sig = dict(uav=args.uav, K=int(args.k), batteries=int(args.batteries),
                max_stops=int(args.max_stops), deck_mode=args.deck_mode,
                quick_inspection_capacity=int(args.quick_inspection_capacity),
                swap_stations=int(args.swap_stations), pricing_mode=str(getattr(args, "pricing_mode", "exact-implicit-dfs")),
                predictor=getattr(args, "recovery_predictor", "cv_noleak"),
                soc_correction=getattr(args, "soc_correction", "geo2d"),
                soc_risk_allocation=getattr(args, "soc_risk_allocation", "optimized"),
                    time_recourse_mode=getattr(args, "time_recourse", "wait_and_speed"),
                    time_contract_id=RM.time_contract_for(signature_params),
                    speed_is_recourse=bool(getattr(signature_params, "speed_adjustable", False)),
                    return_speed_recourse_contract=(RM.SPEED_RECOURSE_CONTRACT if getattr(signature_params, "speed_adjustable", False) else None),
                geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
                result_contract=S13.RESULT_CONTRACT)
    rows, _done = S13._resume_load(outdir, "A2_speed.csv", ["n_turbines", "dtau_min"],
                                   _sig, getattr(args, "resume", "on"))
    for n in [int(x) for x in str(args.a2_n).split(",") if x.strip()]:
        for dtau in [float(x) for x in str(args.a2_dtau).split(",") if x.strip()]:
            if (str(n), str(float(dtau))) in _done:
                continue
            p, xi_amb, wamb, opts, reach, T_eff, kind, t_swap, t_launch = _setup(n, args, dtau=dtau)
            max_stops = int(args.max_stops)
            kw = _final_resource_kwargs(args, t_swap, t_launch, max_stops, wamb)
            t0 = time.time()
            exact = BP.solve_fleet_anytime(
                reach, opts, p, xi_amb, args.k, T_eff,
                solver_mode="exact-branch-price-cut",
                pricing_mode=getattr(args, "pricing_mode", "exact-implicit-dfs"),
                solver="auto", **kw)
            t_exact = time.time() - t0
            t0 = time.time()
            seed = RA.build_route_columns(reach, opts, p, xi_amb, T_eff, args.deck_delta_min,
                                     max_stops, wamb, "drcc", 2.0, "vp_unimodal", 8.0,
                                     pool_h_mode=getattr(args, "pool_h", "pareto"))
            restricted = BP.solve_fleet_anytime(
                reach, opts, p, xi_amb, args.k, T_eff,
                solver_mode="research-baseline", solver="auto", seed_cols=seed, **kw)
            restricted["global_certificate_available"] = False
            restricted["global_route_space_certificate"] = False
            restricted["implicit_route_space_certified"] = False
            t_restricted = time.time() - t0
            cert = bool(exact.get("lexicographic_optimal", False))
            match_cov = bool(cert and int(exact.get("covered", -1)) == int(restricted.get("covered", -2)))
            match_lex = bool(match_cov and abs(float(exact.get("energy_Wh", 0.0)) -
                                               float(restricted.get("energy_Wh", 0.0))) <= 0.11)
            ratio = (t_restricted / max(t_exact, 1e-9)) if match_lex else None
            enum = exact.get("enumeration") or {}
            rows.append(dict(n_turbines=n, reach=len(reach), dtau_min=dtau,
                             n_slots=len(opts), uav=args.uav, K=args.k,
                             batteries=args.batteries, max_stops=max_stops,
                             deck_mode=args.deck_mode, track=kind,
                             predictor=getattr(args, "recovery_predictor", "cv_noleak"),
                             soc_correction=getattr(args, "soc_correction", "geo2d"),
                             soc_risk_allocation=getattr(args, "soc_risk_allocation", "optimized"),
                             time_recourse_mode=getattr(args, "time_recourse", "wait_and_speed"),
                             time_contract_id=RM.time_contract_for(p),
                             speed_is_recourse=bool(getattr(p, "speed_adjustable", False)),
                             return_speed_recourse_contract=(
                                 RM.SPEED_RECOURSE_CONTRACT
                                 if getattr(p, "speed_adjustable", False) else None),
                             geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
                             result_contract=S13.RESULT_CONTRACT,
                             exact_algorithm=exact.get("algorithm"),
                             exact_generated_columns=exact.get("generated_columns"),
                             exact_archive_size=exact.get("generated_column_archive_size"),
                             exact_pricing_calls=exact.get("pricing_calls"),
                             exact_processed_nodes=exact.get("processed_nodes"),
                             exact_resource_cuts=int(exact.get("resource_cuts", 0) or 0),
                             exact_coverage_optimal=bool(exact.get("coverage_optimal", False)),
                             exact_energy_optimal=bool(exact.get("energy_optimal", False)),
                             exact_status=exact.get("status"),
                             exact_time_s=round(t_exact, 3),
                             exact_covered=exact.get("covered"),
                             exact_energy_Wh=exact.get("energy_Wh"),
                             restricted_pool_size=len(seed),
                             restricted_pool_time_s=round(t_restricted, 3),
                             restricted_pool_covered=restricted.get("covered"),
                             restricted_pool_energy_Wh=restricted.get("energy_Wh"),
                             restricted_pool_status=restricted.get("status"),
                             restricted_pool_resource_cuts=int(
                                 restricted.get("resource_cuts", 0) or 0),
                             restricted_pool_certificate=False,
                             lexicographic_optimal=cert,
                             match_coverage_optimal=match_cov,
                             match_lexicographic_optimal=match_lex,
                             match_optimal=match_lex, diagnostic_only=not match_lex,
                             runtime_ratio=(round(ratio, 6) if ratio is not None else None),
                             speedup_factor=(round(ratio, 6)
                                             if ratio is not None and ratio > 1 else None),
                             slowdown_factor=(round(1.0 / ratio, 6)
                                              if ratio is not None and ratio < 1 else None),
                             comparison_label=("speedup" if ratio is not None and ratio > 1 else
                                               ("slowdown" if ratio is not None else "not-comparable")),
                             speedup=(round(ratio, 6) if ratio is not None else None),
                             # Backward aliases consumed by existing figures.  Their
                             # semantics are now explicit: extensive=restricted pool,
                             # ours=on-demand exact BPC.
                             ext_pool=len(seed), ext_evals=None,
                             ext_resource_cuts=int(restricted.get("resource_cuts", 0) or 0),
                             ext_coverage_optimal=False, ext_energy_optimal=False,
                             ext_status=restricted.get("status"),
                             t_enum_s=round(t_restricted, 3),
                             ext_covered=restricted.get("covered"),
                             ext_energy_Wh=restricted.get("energy_Wh"),
                             t_gurobi_total_s=round(t_restricted, 3),
                             ours_covered=exact.get("covered"),
                             ours_energy_Wh=exact.get("energy_Wh"),
                             ours_status=exact.get("status"),
                             t_ours_s=round(t_exact, 3),
                             ours_pool=exact.get("generated_column_archive_size", 0),
                             ours_resource_cuts=int(exact.get("resource_cuts", 0) or 0),
                             ours_restricted_master_certificate=None,
                             **{f"exact_{k}": v for k, v in _global_gap_fields(exact).items()},
                             **{f"restricted_{k}": v for k, v in _global_gap_fields(restricted).items()}))
            _done.add((str(n), str(float(dtau))))
            _save(pd.DataFrame(rows), outdir, "A2_speed.csv")
            EU.write_run_manifest(
                outdir, "A2_speed", args,
                input_paths=[x for x in [getattr(args, "xi_train_samples", None),
                                          getattr(args, "track_csv", None),
                                          getattr(args, "weather_moments_csv", None)] if x is not None],
                extra={"completed_rows": len(rows), "result_contract": S13.RESULT_CONTRACT,
                       "final_resource_model": True, **S13._formal_instance_manifest_extra(args)})
    df = pd.DataFrame(rows); _save(df, outdir, "A2_speed.csv"); return df


def main():
    ap = argparse.ArgumentParser(description=f"{S13.PROJECT_NAME} A1/A2 算法实验")
    ap.add_argument("--exp", default="all", choices=["all", "A1_accuracy", "A2_speed"])
    ap.add_argument("--study-mode", choices=["formal", "mechanism"], default="formal",
                    help="formal 必须与 Step13 使用同一单船 train/weather/launch 信息合同。")
    ap.add_argument("--n-list", type=str, default="6,8,10")
    ap.add_argument("--a2-n", type=str, default="6,8")
    ap.add_argument("--a2-dtau", type=str, default="15,10,5")
    ap.add_argument("--farm", default="Rodsand_II", choices=["Rodsand_II", "Nysted", "Anholt"])
    ap.add_argument("--pair-radius", type=str, default="auto",
                    help="'auto'=UAV 物理外包络(正式口径); 数字=显式米数(仅敏感性分析)")
    ap.add_argument("--max-stops", type=int, default=4,
                    help="与正式模型一致的停靠上限；不得用 max_stops=2 代表完整模型。")
    ap.add_argument("--window-min", type=float, default=360.0)
    ap.add_argument("--dtau-min", type=float, default=5.0)
    ap.add_argument("--deck-delta-min", type=float, default=2.5)
    ap.add_argument("--deck-mode", default="interval", choices=["interval", "slot"],
                    help="更新: interval=区间甲板占用(新默认); slot=旧瞬时槽(消融)")
    ap.add_argument("--t-swap-min", type=float, default=None, help="None=按 UAV 档位默认")
    ap.add_argument("--t-launch-min", type=float, default=None, help="None=按 UAV 档位默认")
    ap.add_argument("--landing-clear-min", type=float, default=1.0)
    ap.add_argument("--quick-inspection-capacity", type=int, default=1)
    ap.add_argument("--swap-stations", type=int, default=1)
    ap.add_argument("--uav", default="auto", choices=["auto"] + sorted(M.UAV_PROFILES),
                    help="UAV 档位(E1 选型后回填; 默认 S=DJI M30T)")
    ap.add_argument("--k", type=int, default=None,
                    help="--uav auto(默认)时由 E1 选型自动回填; 显式 --uav 时默认 3")
    ap.add_argument("--batteries", type=int, default=None)
    ap.add_argument("--selection-metric", default="safe_per_inventory_kWh",
                    choices=["safe_per_inventory_kWh", "per_battery", "energy_per_safe", "max_safe"],
                    help="自动消费 E1 选型时的跨机型指标；默认使用统一物理资源单位库存kWh。")
    ap.add_argument("--hs-quantile", type=float, default=0.5)
    ap.add_argument("--weather-drcc", default="on", choices=["on", "off"])
    ap.add_argument("--weather-alignment", choices=["timestamp", "representative_quantile"],
                    default="timestamp",
                    help="算法实验正式默认按 AIS UTC 时间同步天气；分位情景须显式指定。")
    ap.add_argument("--turbines-csv", type=Path, default=None)
    ap.add_argument("--wind-csv", type=Path, default=None)
    ap.add_argument("--wave-csv", type=Path, default=None)
    ap.add_argument("--xi-moments-csv", type=Path, default=None)
    ap.add_argument("--xi-train-samples", type=Path, default=None,
                    help="formal A1/A2 必需；用于按 concrete MMSI 重建 Xi ambiguity，禁止 mmsi=ALL。")
    ap.add_argument("--weather-moments-csv", type=Path, default=None,
                    help="step7 同步生成的真实 no-leak weather moments；weather-drcc=on 的正式算法实验必需。")
    ap.add_argument("--recovery-scenarios-csv", type=Path, default=None)
    ap.add_argument("--track-csv", default=None)
    ap.add_argument("--track-mmsi", default=None)
    ap.add_argument("--track-start-min", type=float, default=None)
    ap.add_argument("--allow-synth", action="store_true")
    ap.add_argument("--time-limit-s", type=float, default=None)
    ap.add_argument("--coverage-gap-target-abs", type=int, default=0)
    ap.add_argument("--energy-gap-target-rel", type=float, default=0.0)
    ap.add_argument("--energy-gap-target-abs-wh", type=float, default=1e-6)
    ap.add_argument("--solver-mode", choices=["exact-branch-price-cut"],
                    default="exact-branch-price-cut")
    ap.add_argument("--pricing-mode",
                    choices=["r-bpc", "exact-implicit-dfs", "exact-mip"],
                    default="exact-implicit-dfs")
    ap.add_argument("--recovery-predictor", choices=["cv_noleak", "true_track"],
                    default="cv_noleak", dest="recovery_predictor",
                    help="更新: 与 step13 同口径 —— cv_noleak=无泄漏后向窗 CV(正式); "
                         "true_track=真实未来航迹(泄漏消融)")
    ap.add_argument("--pool-h", choices=["pareto", "first"], default="pareto", dest="pool_h",
                    help="更新: 列池 h 保留策略(pareto=非支配前沿/正式; first=旧口径消融)")
    ap.add_argument("--soc-correction", choices=["none", "geo2d"], default="geo2d",
                    help="与 step13 同口径(更新 连带修复: 此前 A 实验缺此旗标会与 E 实验口径错位)。"
                         "geo2d=正式口径(默认); none=仅消融对照。")
    ap.add_argument("--soc-risk-allocation", choices=["fixed", "optimized"], default="optimized",
                    help="与 step13 同一 geo2d Bonferroni 份额合同；optimized 不改变总 ε。")
    ap.add_argument("--time-recourse", choices=["wait_only", "wait_and_speed"], default="wait_and_speed",
                    help="与 step13 同一固定接地时间 recourse 合同。")
    ap.add_argument("--resume", choices=["on", "off"], default="on",
                    help="断点续跑(更新): on=跳过已完成 (n,method)/(n,dtau) 键续跑; off=整跑覆盖")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    formal = (args.study_mode == "formal")
    _validate_formal_base_parity(args)
    if formal and args.allow_synth:
        raise SystemExit("formal A1/A2 禁止 --allow-synth。")
    if formal:
        required = {"--turbines-csv": args.turbines_csv, "--wind-csv": args.wind_csv,
                    "--wave-csv": args.wave_csv, "--xi-train-samples": args.xi_train_samples}
        if args.weather_drcc == "on":
            required["--weather-moments-csv"] = args.weather_moments_csv
        missing = [name for name, value in required.items()
                   if value is None or not Path(value).is_file()]
        if args.track_csv is None and args.track_mmsi is None:
            missing.append("--track-csv/--track-mmsi")
        if args.track_start_min is None:
            missing.append("--track-start-min")
        if missing:
            raise SystemExit("formal A1/A2 必须显式提供输入: " + ", ".join(missing))
    elif not args.allow_synth:
        required = {"--turbines-csv": args.turbines_csv, "--wind-csv": args.wind_csv,
                    "--wave-csv": args.wave_csv, "--xi-moments-csv": args.xi_moments_csv}
        if args.weather_drcc == "on":
            required["--weather-moments-csv"] = args.weather_moments_csv
        missing = [name for name, value in required.items()
                   if value is None or not Path(value).is_file()]
        if args.track_csv is None and args.track_mmsi is None:
            missing.append("--track-csv/--track-mmsi")
        if missing:
            raise SystemExit("算法实验必须通过本地路径显式提供输入: " + ", ".join(missing))
    S13._resolve_e2_config(args)                   # A 实验与 E2 同规则自动消费 E1 选型
    if args.batteries is None:
        args.batteries = 2 * int(args.k)

    if args.exp in ("all", "A1_accuracy"):
        print("\n[A1_accuracy] research baselines vs exact Branch-Price-and-Cut (uav=%s, stops=%d, 甲板=%s)"
              % (args.uav, args.max_stops, args.deck_mode))
        df = A1_accuracy(args, RESULTS / "A1_accuracy")
        print(df[["n_turbines", "method", "covered", "best_known", "gap_to_best",
                  "status", "coverage_optimal", "energy_optimal", "time_s"]].to_string(index=False))
    if args.exp in ("all", "A2_speed"):
        print("\n[A2_speed] research restricted pool vs exact Branch-Price-and-Cut")
        df = A2_speed(args, RESULTS / "A2_speed")
        print(df[["n_turbines", "dtau_min", "ext_pool", "ext_status", "t_gurobi_total_s",
                  "ours_covered", "t_ours_s", "match_lexicographic_optimal",
                  "runtime_ratio", "speedup_factor", "slowdown_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
