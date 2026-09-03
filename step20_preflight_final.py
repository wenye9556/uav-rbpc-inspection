#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step20_preflight_final.py

Zero-optimization formal-publication preflight.

This step never solves E1/E2 and never consumes the final test.  It validates
the data/model/protocol contracts that must hold *before* a costly final run:
single-vessel Xi binding, train/validation/test separation, purge/horizon
support, formal weather provenance, exact mission risk budget, and the
one-time-final-test freeze protocol.

Exit code:
  0  all mandatory checks PASS
  2  one or more mandatory checks FAIL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import pandas as pd

import step9_model as M
import step10_model_routing as RM
import step13_experiment_model as E
import step15_replay as RP

CONTRACT = "formal-publication-preflight-v3-coherent-weather-power-audit-metadata-only-test"


def _sha(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(rows, name, ok, detail="", *, mandatory=True):
    rows.append(dict(check=name, status=("PASS" if ok else ("FAIL" if mandatory else "WARN")),
                     mandatory=bool(mandatory), detail=str(detail)))
    tag = "PASS" if ok else ("FAIL" if mandatory else "WARN")
    print(f"[{tag}] {name}: {detail}")
    return bool(ok)


def parser():
    ap = argparse.ArgumentParser(description="Formal final-run preflight; no optimization/test consumption")
    ap.add_argument("--track-csv", type=Path, required=True)
    ap.add_argument("--track-mmsi", default=None)
    ap.add_argument("--xi-train-samples", type=Path, required=True)
    ap.add_argument("--validation-samples", type=Path, required=True)
    ap.add_argument("--final-test-samples", type=Path, required=True)
    ap.add_argument("--weather-moments-csv", type=Path, required=True)
    ap.add_argument("--holdout-purge-min", type=float, required=True)
    ap.add_argument("--soc-correction", choices=["none","geo2d"], default="geo2d")
    ap.add_argument("--soc-risk-allocation", choices=["fixed","optimized"], default="optimized")
    ap.add_argument("--time-recourse", choices=["wait_only","wait_and_speed"], default="wait_and_speed")
    ap.add_argument("--recovery-predictor", choices=["cv_noleak","true_track"], default="cv_noleak")
    ap.add_argument("--max-sorties-power", type=int, default=8,
                    help="仅元数据统计功效审计：报告 m=1..N 条 sortie 时零失败 UCB 所需样本量。")
    ap.add_argument("--results-root", type=Path,
                    default=Path(__file__).resolve().parent/"results"/"model_experiments")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent/"results"/"model_experiments"/"final_preflight.json")
    return ap


def main():
    args = parser().parse_args()
    rows = []
    print("="*96)
    print("STEP20 / FORMAL PUBLICATION PREFLIGHT")
    print("contract:", CONTRACT)
    print("This step does NOT optimize and does NOT read final-test outcomes.")
    print("="*96)

    for name, p in [
        ("track_csv", args.track_csv),
        ("xi_train_samples", args.xi_train_samples),
        ("validation_samples", args.validation_samples),
        ("final_test_samples", args.final_test_samples),
        ("weather_moments_csv", args.weather_moments_csv),
    ]:
        if not _check(rows, f"file:{name}", Path(p).is_file(), str(p)):
            continue

    try:
        selected = E._infer_concrete_track_mmsi(
            args.track_csv, args.track_mmsi, formal=True)
        _check(rows, "single_concrete_track_mmsi", bool(selected), selected)
    except Exception as exc:
        selected = ""
        _check(rows, "single_concrete_track_mmsi", False, f"{type(exc).__name__}:{exc}")

    dfs = {}
    if selected:
        for split, p in [("train", args.xi_train_samples),
                         ("validation", args.validation_samples),
                         ("test", args.final_test_samples)]:
            try:
                d = (RP.load_sample_metadata(p, mmsi=selected, formal=True, expected_split=split)
                     if split == "test" else
                     RP.load_samples(p, mmsi=selected, formal=True, expected_split=split))
                dfs[split] = d
                concrete = sorted(set(d["mmsi"].astype(str)))
                _check(rows, f"{split}:mmsi_exact", concrete == [selected],
                       f"rows={len(d)} mmsi={concrete}")
                _check(rows, f"{split}:nonempty", len(d) > 0, f"rows={len(d)}")
            except Exception as exc:
                _check(rows, f"{split}:formal_sample_contract", False,
                       f"{type(exc).__name__}:{exc}")

    hashes = []
    for p in (args.xi_train_samples, args.validation_samples, args.final_test_samples):
        if Path(p).is_file():
            hashes.append(_sha(p))
    _check(rows, "train_validation_test_hashes_distinct",
           len(hashes) == 3 and len(set(hashes)) == 3,
           f"hashes={hashes}")

    xi_amb = None
    if "train" in dfs:
        try:
            hs = sorted(dfs["train"]["h_min"].astype(int).unique())
            states = sorted(dfs["train"]["c_state"].astype(str).unique())
            xi_amb = RP.ambiguity_from_samples(dfs["train"], hs, states, formal=True)
            _check(rows, "formal_xi_single_vessel",
                   str(getattr(xi_amb, "selected_mmsi", "")) == selected
                   and not bool(getattr(xi_amb, "cross_vessel_pooling", True)),
                   f"xi_mmsi={getattr(xi_amb,'selected_mmsi',None)} "
                   f"cross_pool={getattr(xi_amb,'cross_vessel_pooling',None)} cells={len(xi_amb.cells)}")
        except Exception as exc:
            _check(rows, "formal_xi_single_vessel", False,
                   f"{type(exc).__name__}:{exc}")

    if len(dfs) == 3:
        hsets = {k: sorted(v["h_min"].astype(int).unique().tolist()) for k,v in dfs.items()}
        max_h = max(hsets["train"]) if hsets["train"] else math.inf
        _check(rows, "purge_covers_max_train_horizon",
               float(args.holdout_purge_min) >= float(max_h),
               f"purge={args.holdout_purge_min}, max_h={max_h}")
        _check(rows, "validation_contains_train_horizons",
               set(hsets["train"]).issubset(hsets["validation"]),
               str(hsets))
        _check(rows, "test_contains_train_horizons",
               set(hsets["train"]).issubset(hsets["test"]),
               str(hsets))
        try:
            RP.validate_holdout_independence(
                dfs["train"], dfs["validation"], dfs["test"],
                purge_min=float(args.holdout_purge_min),
                require_real_weather=True, require_real_recovery_state=True)
            _check(rows, "holdout_temporal_disjointness_and_purge", True,
                   "train/validation/test intervals are purged/disjoint; test schema/provenance pass; final-test outcomes not materialized")
        except Exception as exc:
            _check(rows, "holdout_temporal_disjointness_and_purge", False,
                   f"{type(exc).__name__}:{exc}")

    if xi_amb is not None:
        try:
            wamb = RM.weather_ambiguity_from_moments_csv(
                args.weather_moments_csv, RM.decision_horizons_of(xi_amb), formal=True)
            _check(rows, "formal_weather_contract", True,
                   f"horizons={RM.decision_horizons_of(xi_amb)} "
                   f"source={getattr(wamb,'source_path',None)}")
            actual_train_sha = _sha(args.xi_train_samples)
            recorded_train_sha = str(getattr(wamb, "xi_train_source_sha256", "") or "")
            _check(rows, "weather_moments_bound_to_actual_xi_train",
                   recorded_train_sha.lower() == actual_train_sha.lower(),
                   f"recorded={recorded_train_sha} actual={actual_train_sha}")
        except Exception as exc:
            _check(rows, "formal_weather_contract", False,
                   f"{type(exc).__name__}:{exc}")

    p = M.Params()
    p.soc_correction = args.soc_correction
    p.soc_risk_allocation = args.soc_risk_allocation
    p.time_recourse_mode = args.time_recourse
    p.speed_adjustable = (args.time_recourse == "wait_and_speed")
    try:
        p.validate_contract(formal=True)
        risk_vals = [p.eps_E,p.eps_T,p.eps_cap,p.eps_gate,p.eps_air,p.eps_dock,p.eps_escort]
        exact_sum = sum((Fraction.from_float(float(x)) for x in risk_vals), Fraction(0))
        exact_budget = Fraction.from_float(float(p.mission_failure_budget))
        _check(rows, "mission_risk_budget_exact_binary64",
               exact_sum <= exact_budget,
               f"sum={float(exact_sum):.17g} budget={float(exact_budget):.17g}")
    except Exception as exc:
        _check(rows, "formal_params_contract", False,
               f"{type(exc).__name__}:{exc}")

    # Metadata-only statistical-power audit.  This never reads validation/test
    # failure outcomes; it only reports whether a cell could pass the current
    # Bonferroni Hoeffding-Azuma gate even under zero observed failures.
    power_report = []
    if "validation" in dfs and "test" in dfs:
        allocation_budget = float(RM.mission_eps_budget(p, True))
        max_sorties = max(1, int(args.max_sorties_power))
        for m in range(1, max_sorties + 1):
            alpha = 0.05 / float(m)
            required_n = int(math.ceil(math.log(1.0 / alpha) /
                                       (2.0 * allocation_budget * allocation_budget)))
            vg = dfs["validation"].groupby(["h_min", "c_state"]).size()
            tg = dfs["test"].groupby(["h_min", "c_state"]).size()
            keys = sorted(set(vg.index) | set(tg.index), key=lambda z: (float(z[0]), str(z[1])))
            supported_validation = 0
            supported_test = 0
            for h, state in keys:
                nv = int(vg.get((h, state), 0))
                nt = int(tg.get((h, state), 0))
                sv = bool(nv >= required_n)
                st = bool(nt >= required_n)
                supported_validation += int(sv)
                supported_test += int(st)
                power_report.append(dict(
                    sorties=int(m), h_min=float(h), c_state=str(state),
                    zero_failure_required_n=int(required_n),
                    validation_n=nv, validation_zero_failure_support=sv,
                    final_test_metadata_n=nt, final_test_zero_failure_support=st))
            _check(
                rows, f"statistical_power_metadata:m={m}", True,
                f"zero-failure required_n={required_n}; "
                f"validation supported cells={supported_validation}/{len(keys)}; "
                f"test-metadata supported cells={supported_test}/{len(keys)}",
                mandatory=False)

    _check(rows, "formal_recovery_predictor_cv_noleak",
           args.recovery_predictor == "cv_noleak", args.recovery_predictor)
    _check(rows, "formal_soc_geo2d", args.soc_correction == "geo2d", args.soc_correction)
    _check(rows, "formal_time_recourse_wait_and_speed",
           args.time_recourse == "wait_and_speed", args.time_recourse)

    rr = Path(args.results_root)
    e1_final = rr/"E1_frontier"/"E1_final_test.csv"
    e2_final = rr/"E2_robust_comparison"/"E2_final_test.csv"
    _check(rows, "no_legacy_E1_final_test_consumption",
           not e1_final.exists(),
           ("absent" if not e1_final.exists()
            else f"FOUND {e1_final}; v9 forbids formal E1 test consumption"))
    _check(rows, "E2_final_test_state",
           True,
           ("not yet consumed" if not e2_final.exists()
            else f"existing frozen final audit present: {e2_final}; do not change model/selection after this"),
           mandatory=False)

    passed = all(r["status"] != "FAIL" for r in rows)
    rec = dict(contract=CONTRACT, pass_all_mandatory=bool(passed),
               selected_mmsi=selected or None,
               checks=rows, statistical_power=power_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPRECHECK:", "PASS" if passed else "FAIL")
    print("report:", args.out)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
