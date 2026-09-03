#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step13_experiment_model.py — 最终模型实验入口。

正式模型: 船沿起飞时可获得的 AIS/GNSS 预测轨迹穿行风场；列为 (τ,π,h)。
L1 最大化 6 小时窗内可靠完成的不同风机数量，L2 最小化
E_flight+E_escort+E_dock。给定列池的主问题显式分配 UAV 与实体电池组，允许剩余
SOC 精确复用；同电池续用只占快速检查工位，实际换电才占非着陆区换电工位，
着陆区仅占用到清场。正式运行使用 train 矩、validation 选型、独立 joint test 审计，
并以逐架次 Bonferroni 同时单侧置信上界判定任务级 95% 可靠性。最终有限模型证书
由 step12.solve_fleet_anytime 的按需精确定价、完备分支和精确资源审计闭合时给出。

模型实验(更新 作者定案: E1 三轴主实验 + E2 鲁棒方法对照; E2/A 配置依赖 E1 选型):
  E1_frontier   UAV 种类(S/M/L) × 机数 K(更新: 1..8) × 电池 B(0..8 基础网格 + 饱和自动延伸)
                三轴前沿(每 UAV 共享列池); 停靠上界逐 UAV auto(更新, 消除 L 档删失),
                每行带 stops_cap/stops_cap_hit; 明细逐 UAV 各存(含逐列裕度/绑定约束)
  E1_select     更新: 读 E1_frontier.csv 出 (uav,K,B) 三口径选型表(knee/per_battery/energy)
                + 打印 E2/A 回填命令(落地 更新 遗留"膝点自动选型脚本")
  E2_robust     分布鲁棒(vp, 本文) vs {nominal, gaussian, SAA, box, budget_Γ2, cantelli}
                × 起始窗 Hs 分位; 逐列 out-of-sample 回放(t3 重尾); adaptive 已移除(dormant);
                更新: stops_cap 与 E1 同源解析(--stops-cap), 回填口径不再错位

旧 exp1~8 / dro_stress / multiweather / co-timing 已按作者"直接去掉"决策于 更新 整体删除
(机制或被 吸收为模型本身, 或由 E1/E2/E3 取代); 其历史 CSV 与本套件数字不可混排。
算法实验见 step14(A1 池完备性/精确性、A2 规模; 待算法改进轮重述)。

结果目录: results/model_experiments/{E1_frontier, E2_robust_comparison}/
数据: data/turbines_*.csv, weather/weather_*.csv, tracks/xi_moments_caseB.csv,
      tracks/ship_track*.csv(真实 AIS; 无则合成穿场航迹自动放慢铺满窗)。

用法:
  python step13_experiment_model.py --exp full_suite                  # E1/E2 + A1/A2 机制全流程
  python step13_experiment_model.py --exp E1_frontier --fleet-ks 1,2,3,4
  python step13_experiment_model.py --exp E2_robust --k 3 --e2-quantiles 0.2,0.5,0.8
  常用: --window-min 360 --dtau-min 5 --deck-delta-min 2.5 --t-swap-min 4 --replay-n 400
  口径(更新 定案): --pair-radius auto --soc-correction geo2d 皆为默认 = 正式口径;
  断点续跑: --resume on(默认) —— 中断后原命令重跑即续, 口径签名不一致会被拒绝混排

依赖: numpy pandas scipy(milp; 缺则贪心回退并如实标注)
对应 doc_model.md(节)、doc_experiments.md(更新 重写版)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import step9_model as M
import step10_model_routing as RM
import step11_algorithm_route_drcc as RA
import step12_branch_price as BP
EU = M  # shared utilities are merged into step9_model to preserve package layout

log = logging.getLogger("exp_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS = Path(__file__).resolve().parent / "results" / "model_experiments"
PROJECT_NAME = "shipborne-uav-wind-turbine-inspection"
RESULT_CONTRACT = "fleet-anytime-result-v17.1-coherent-weather-formal-algorithm-parity"
FORMAL_EXPERIMENT_SCHEDULER_CONTRACT = (
    "formal-e1-e2-v11-global-battery-purged-disjoint-confirmatory-martingale-"
    "launch-asof-weather-single-vessel-coherent-weather-algorithm-parity")


def _infer_concrete_track_mmsi(track_csv, explicit_mmsi=None, *, formal=False):
    """Resolve one concrete vessel identity before any Xi/holdout statistics are loaded.

    Formal sample-driven experiments are single-vessel experiments.  Filename
    ``track_<mmsi>.csv`` and an explicit ``--track-mmsi`` are accepted only when
    they agree.  As a fallback, a unique MMSI column in the track file may bind
    the identity.  Ambiguous/missing identity is fail-closed in formal mode.
    """
    explicit = str(explicit_mmsi or "").strip()

    # Mechanism mode may intentionally receive multiple track_*.csv candidates
    # from load_all(); build_launch_options() is responsible for selecting one
    # by in-farm dwell time.  Do not coerce that candidate list into Path here.
    if isinstance(track_csv, (list, tuple)):
        candidates = [x for x in track_csv if x is not None]
        if formal:
            if len(candidates) != 1:
                raise SystemExit(
                    "正式实验必须在统计量加载前显式绑定唯一 AIS 航迹；"
                    "禁止传入多个 track_*.csv 候选。")
            track_csv = candidates[0]
        else:
            # In mechanism mode the concrete MMSI is resolved after
            # build_launch_options() selects the actual track.
            return explicit

    path = Path(track_csv) if track_csv is not None else None
    from_name = ""
    if path is not None:
        m = re.fullmatch(r"track_(\d+)", path.stem, flags=re.IGNORECASE)
        if m:
            from_name = m.group(1)
    from_data = ""
    if path is not None and path.is_file():
        try:
            probe = pd.read_csv(path, nrows=2000)
            for col in ("mmsi", "MMSI"):
                if col in probe.columns:
                    vals = sorted({str(x).strip() for x in probe[col].dropna().astype(str)
                                   if str(x).strip() not in {"", "nan", "None"}})
                    if len(vals) == 1:
                        from_data = vals[0]
                    elif len(vals) > 1 and formal:
                        raise SystemExit(
                            f"正式轨迹文件必须绑定唯一 MMSI；{path.name} 的 {col} 包含 {vals[:8]!r}")
                    break
        except SystemExit:
            raise
        except Exception:
            from_data = ""
    vals = [v for v in (explicit, from_name, from_data) if v]
    if vals and len(set(vals)) != 1:
        raise SystemExit(f"轨迹 MMSI provenance 冲突: explicit/name/data={vals!r}")
    resolved = vals[0] if vals else ""
    if formal and not resolved:
        raise SystemExit(
            "正式实验必须在加载 Xi train/validation/test 前绑定一个具体 MMSI；"
            "请使用 --track-mmsi 或 track_<mmsi>.csv。禁止 mmsi=ALL。")
    return resolved


def _xi_ambiguity_from_train_samples(path, mmsi, *, formal: bool):
    """Build Xi ambiguity from the exact train file used by a formal instance.

    This helper is shared by Step13 and Step14 so algorithm experiments cannot
    silently fall back to the pooled ``mmsi=ALL`` moments file.
    """
    import step15_replay as RP
    pth = Path(path)
    if not pth.is_file():
        raise SystemExit(f"Xi train 样本不存在: {pth}")
    selected = str(mmsi or "").strip()
    if formal and (not selected or selected.upper() == "ALL"):
        raise SystemExit("formal Xi train 必须绑定具体 MMSI，禁止 mmsi=ALL。")
    sample_mmsi = selected if formal else "ALL"
    train_df = RP.load_samples(
        pth, mmsi=sample_mmsi, formal=bool(formal),
        expected_split=("train" if formal else None))
    hs = sorted(train_df["h_min"].astype(int).unique())
    states = sorted(train_df["c_state"].astype(str).unique())
    xi_amb = RP.ambiguity_from_samples(train_df, hs, states, formal=bool(formal))
    if not xi_amb.cells:
        raise SystemExit("--xi-train-samples 无法形成任何可用 (h,c) Xi ambiguity cell。")
    if formal:
        if str(getattr(xi_amb, "selected_mmsi", "")) != selected:
            raise SystemExit(
                "formal Xi ambiguity 未绑定请求 MMSI: "
                f"xi={getattr(xi_amb, 'selected_mmsi', None)!r}, requested={selected!r}")
        if bool(getattr(xi_amb, "cross_vessel_pooling", False)):
            raise SystemExit("formal Xi ambiguity 检测到 cross_vessel_pooling=True。")
    return train_df, xi_amb


def _assert_weather_xi_train_binding(wamb, xi_train_samples, *, formal: bool):
    """Bind formal weather moments to the *actual* Xi train file passed on the CLI."""
    if not formal or wamb is None:
        return
    if xi_train_samples is None:
        raise SystemExit("formal weather ambiguity 需要 --xi-train-samples 以验证 provenance。")
    actual = str(EU.sha256_file(Path(xi_train_samples)) or "")
    recorded = str(getattr(wamb, "xi_train_source_sha256", "") or "")
    if not actual or recorded.lower() != actual.lower():
        raise SystemExit(
            "formal weather moments 与本次 Xi train 文件 provenance 不一致: "
            f"recorded={recorded!r}, actual={actual!r}")


# =============================================================================
# 数据装载 + 代表性出动构造(与原 step15 一致, 复用回退逻辑)
# =============================================================================
def load_all(n_turbines, farm="Rodsand_II", allow_synth=True, *,
             turbines_csv=None, wind_csv=None, wave_csv=None,
             xi_moments_csv=None, recovery_scenarios_csv=None,
             track_csv=None, require_xi_moments=True):
    here = Path(__file__).resolve().parent
    def _resolve(explicit, candidates, label):
        if explicit is not None:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise SystemExit(f"{label}不存在: {path}")
            return path
        return M._first_existing(candidates)

    turb_csv = _resolve(turbines_csv, [here / "data" / f"turbines_{farm}_clean.csv",
                                  Path(f"/mnt/user-data/uploads/turbines_{farm}_clean.csv"),
                                  here / "data" / "turbines_Rodsand_II_clean.csv",
                                  Path("/mnt/user-data/uploads/turbines_Rodsand_II_clean.csv")], "风机 CSV")
    xi_csv = _resolve(xi_moments_csv, [here / "tracks" / "xi_moments_caseB.csv",
                                here / "xi_moments_caseB.csv",
                                Path("/mnt/user-data/uploads/xi_moments_caseB.csv")], "ξ 矩 CSV")
    wind_csv = _resolve(wind_csv, [here / "weather" / f"weather_{farm}.csv",
                                  here / "weather" / "weather_Rodsand_II.csv",
                                  Path("/mnt/user-data/uploads/weather_Rodsand_II.csv")], "风场 CSV")
    wave_csv = _resolve(wave_csv, [here / "weather" / f"waves_{farm}.csv",
                                  here / "weather" / "waves_Rodsand_II.csv",
                                  Path("/mnt/user-data/uploads/waves_Rodsand_II.csv")], "浪场 CSV")
    sc_csv = _resolve(recovery_scenarios_csv, [here / "tracks" / "recovery_scenarios.csv",
                                here / "recovery_scenarios.csv",
                                Path("/mnt/user-data/uploads/recovery_scenarios.csv")], "回收场景 CSV")
    _legacy = _resolve(track_csv, [here / "tracks" / f"ship_track_{farm}.csv",
                                 here / "tracks" / "ship_track.csv",
                                 here / "tracks" / "ais_track.csv",
                                 here / "tracks" / "mothership_track.csv",
                                 Path("/mnt/user-data/uploads/ship_track.csv"),
                                 Path("/mnt/user-data/uploads/ais_track.csv")], "母船航迹 CSV")
    missing_core = []
    if turb_csv is None:
        missing_core.append("风机文件 data/turbines_<farm>_clean.csv")
    if xi_csv is None and require_xi_moments:
        missing_core.append("误差矩文件 tracks/xi_moments_caseB.csv")
    if wind_csv is None:
        missing_core.append("风场文件 weather/weather_<farm>.csv")
    if wave_csv is None:
        missing_core.append("浪场文件 weather/waves_<farm>.csv")
    if missing_core and not allow_synth:
        raise SystemExit(
            "正式实验缺少本地输入，已拒绝静默使用占位数据:\n  - "
            + "\n  - ".join(missing_core)
            + "\n请把本地数据放到上述目录；仅调试时显式加入 --allow-synth。")

    # 更新 修复: 真实产物名是 step6 的 track_<mmsi>.csv(如 track_219028973.csv) ——
    # 此前清单一个都匹配不上, 导致全部实验静默回退合成航迹(见 历史变更记录 事故复盘)。
    # 返回值可为【候选列表】: build_launch_options 会按"场内驻留时长"自动选船并大声报告。
    if _legacy is not None:
        track_csv = _legacy
    else:
        _glob = sorted((here / "tracks").glob("track_*.csv")) if (here / "tracks").is_dir() else []
        track_csv = (_glob if len(_glob) > 1 else (_glob[0] if _glob else None))

    turbines = M.load_turbines(turb_csv, farm=farm) if turb_csv else \
        [M.Turbine(f"DEMO_{i}", np.array([11.55 + 0.004 * (i % 12), 54.52 + 0.004 * (i // 12)]),
                   68.5, 115.0) for i in range(30)]
    if n_turbines:
        turbines = turbines[:n_turbines]
    lat0, lon0 = turbines[0].lonlat[1], turbines[0].lonlat[0]
    for t in turbines:
        t.local = M.latlon_to_local_m(t.lonlat[1], t.lonlat[0], lat0, lon0)

    wx_df = M.load_weather(wave_csv, wind_csv)
    if xi_csv:
        xi_amb = M.XiAmbiguity.from_csv(xi_csv, mmsi="ALL")
        src = f"真实 ξ 矩 {xi_csv.name}"
    else:
        xi_amb = RM._demo_xi_realistic([5, 10, 15, 20, 30], ["直航", "转弯", "低速", "动力定位"])
        src = "占位真实量级 ξ 夹具(无 xi_moments_caseB.csv)"
        log.warning("无 ξ 文件, 使用真实量级合成夹具。")
    return turbines, wx_df, xi_amb, (lat0, lon0), sc_csv, src, track_csv


def _record_formal_instance_provenance(args, *, mmsi, track_csv, xi_train_samples,
                                       weather_moments_csv, weather_uncertainty,
                                       launch_formal: bool):
    """Record finite-instance audit fields shared by Step13 and Step14 formal runs.

    ``step9_model._jsonable`` intentionally omits underscore-prefixed Namespace
    attributes, so formal manifests copy these values explicitly via ``extra``.
    This is provenance hardening only; it does not alter optimization semantics.
    """
    args._formal_instance_mmsi = str(mmsi or "ALL")
    args._formal_instance_track_sha256 = (
        EU.sha256_file(track_csv) if track_csv is not None and Path(track_csv).is_file() else "")
    args._formal_instance_xi_train_sha256 = (
        EU.sha256_file(xi_train_samples)
        if xi_train_samples is not None and Path(xi_train_samples).is_file() else "")
    args._formal_instance_weather_moments_sha256 = (
        EU.sha256_file(weather_moments_csv)
        if weather_moments_csv is not None and Path(weather_moments_csv).is_file() else "")
    args._formal_instance_weather_predictor_contract = (
        str(getattr(weather_uncertainty, "predictor_contract", ""))
        if weather_uncertainty is not None else "off")
    args._formal_instance_launch_formal = bool(launch_formal)


def _formal_instance_manifest_extra(args):
    """Return private audit fields that must be explicitly persisted in manifests."""
    keys = (
        "_formal_instance_mmsi",
        "_formal_instance_track_sha256",
        "_formal_instance_xi_train_sha256",
        "_formal_instance_weather_moments_sha256",
        "_formal_instance_weather_predictor_contract",
        "_formal_instance_launch_formal",
    )
    return {k: getattr(args, k, None) for k in keys}


def _anytime_solver_kwargs(args):
    """One solver-budget contract shared by all experiment entry points."""
    return dict(
        time_limit_s=getattr(args, "time_limit_s", None),
        coverage_gap_target_abs=int(getattr(args, "coverage_gap_target_abs", 0)),
        energy_gap_target_rel=float(getattr(args, "energy_gap_target_rel", 0.0)),
        energy_gap_target_abs_Wh=float(getattr(args, "energy_gap_target_abs_wh", 1e-6)),
        pricing_mode=str(getattr(args, "pricing_mode", "r-bpc")),
        archive_diagnostic_time_limit_s=float(
            getattr(args, "archive_diagnostic_time_limit_s", 30.0)),
        archive_shadow_diagnostic_time_limit_s=float(
            getattr(args, "archive_shadow_diagnostic_time_limit_s", 30.0)),
        archive_clique_diagnostic_time_limit_s=float(
            getattr(args, "archive_clique_diagnostic_time_limit_s", 30.0)),
        archive_primal_recovery=(
            str(getattr(args, "archive_primal_recovery", "off")).lower() == "on"),
        archive_primal_recovery_time_limit_s=float(
            getattr(args, "archive_primal_recovery_time_limit_s", 2.0)),
        fullspace_target_diagnostic_time_limit_s=float(
            getattr(args, "fullspace_target_diagnostic_time_limit_s", 0.0)),
    )


def _formal_ondemand_pricing(args):
    """True when the experiment must not prebuild a multi-stop route pool."""
    return (
        str(getattr(args, "study_mode", "formal")) == "formal"
        and str(getattr(args, "solver_mode", "exact-branch-price-cut")).replace("_", "-")
            == "exact-branch-price-cut"
    )


_CERTIFICATE_COMPAT_FIELDS = (
    "global_certificate_available",
    "global_route_space_certificate",
    "implicit_route_space_certified",
)


def _strict_certificate_bool(value):
    """Parse persisted certificate evidence without Python truthiness traps.

    Accepted values are actual booleans (including NumPy bool) and finite numeric
    0/1 values.  Strings such as ``"False"``/``"0"``, NaN, None and arbitrary
    objects are invalid rather than truthy.  Invalid evidence always fails closed.
    """
    if isinstance(value, (bool, np.bool_)):
        return True, bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if int(value) in (0, 1):
            return True, bool(int(value))
        return False, False
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isfinite(f) and f in (0.0, 1.0):
            return True, bool(int(f))
        return False, False
    return False, False


def _normalized_certificate_evidence(result):
    """Return ``(certificate, conflict, invalid)`` from one canonical source."""
    if result is None:
        return False, False, False

    canonical_present = "global_certificate_available" in result
    canonical_valid, canonical = _strict_certificate_bool(
        result.get("global_certificate_available")) if canonical_present else (False, False)
    invalid = bool(canonical_present and not canonical_valid)
    # Missing or invalid canonical evidence can never be promoted by aliases.
    canonical_for_compare = bool(canonical) if canonical_valid else False
    conflict = False
    for name in _CERTIFICATE_COMPAT_FIELDS:
        if name not in result:
            continue
        valid, value = _strict_certificate_bool(result.get(name))
        if not valid:
            invalid = True
            continue
        if not canonical_present or not canonical_valid or bool(value) != canonical_for_compare:
            # Legacy-only positives and any disagreement are migration conflicts.
            if name != "global_certificate_available":
                conflict = True
    cert = bool(canonical_present and canonical_valid and canonical and not conflict and not invalid)
    return cert, bool(conflict), bool(invalid)


def _normalized_certificate_state(result):
    """Backward-compatible ``(certificate, conflict)`` accessor."""
    cert, conflict, _invalid = _normalized_certificate_evidence(result)
    return cert, conflict


def _canonical_certificate_fields(result):
    """Normalized solver certificate fields for reports and persisted CSVs."""
    cert, conflict, invalid = _normalized_certificate_evidence(result)
    return dict(
        global_certificate_available=cert,
        global_route_space_certificate=cert,
        implicit_route_space_certified=cert,
        certificate_field_conflict=conflict,
        certificate_field_invalid=invalid)


def _global_certificate_flag(result):
    """Read the normalized complete finite-route-space certificate."""
    return _normalized_certificate_evidence(result)[0]


def _e1_certificate_provenance_fields(result):
    """Persist the solver proof-contract identity on every E1 frontier row.

    E1_frontier can certify the resource knee inline via ``_cell(full_lex=True)``
    without entering ``E1_knee_refine``.  Both paths must therefore carry the
    same certificate provenance; otherwise a genuinely certified knee cannot
    pass the later formal freeze provenance check.
    """
    result = result or {}
    return dict(
        result_certificate_contract=result.get("result_certificate_contract"),
        formal_proof_contract=result.get("formal_proof_contract"),
        proof_contract_sha256=result.get("proof_contract_sha256"),
    )


def _e1_global_certificate_with_provenance(result):
    """Trust an E1 full-lex certificate only with the current proof contracts.

    This also makes old V17.1.4 frontier rows that lost provenance on the inline
    ``E1_frontier`` full-lex path automatically fall back to
    ``needs_lexicographic_knee_certification`` on resume, so the knee is
    re-certified rather than silently remaining unfreezable.
    """
    if result is None or not _global_certificate_flag(result):
        return False
    return bool(
        str(result.get("result_certificate_contract", "")) == str(BP.RESULT_CERTIFICATE_CONTRACT)
        and str(result.get("formal_proof_contract", "")) == str(BP.FORMAL_PROOF_CONTRACT)
        and str(result.get("proof_contract_sha256", "")) == str(BP.FORMAL_PROOF_CONTRACT_SHA256)
    )


def _coverage_certificate_flag(result):
    """Physical Stage-1 certificate, fail-closed across result-contract versions.

    A full lexicographic physical certificate implies the coverage certificate.
    New coverage-only solves expose the weaker certificate explicitly; a missing
    field is never inferred from ``coverage_optimal`` alone.
    """
    if result is None:
        return False
    if "coverage_global_certificate_available" in result:
        valid, value = _strict_certificate_bool(
            result.get("coverage_global_certificate_available"))
        return bool(valid and value)
    return bool(_global_certificate_flag(result))


def _e1_frontier_solver_kwargs(args):
    """Fast formal E1 cells: exact Stage-1 only, with a shorter discovery clock.

    This never upgrades an unfinished solve to exact.  The returned incumbent and
    rigorous coverage upper bound are safe anytime evidence.  Full lexicographic
    optimization is deferred to the small number of knee candidates.
    """
    kw = _anytime_solver_kwargs(args)
    if (str(getattr(args, "study_mode", "formal")) == "formal"
            and _formal_ondemand_pricing(args)):
        frontier_limit = getattr(args, "e1_frontier_time_limit_s", 120.0)
        if frontier_limit is not None:
            frontier_limit = float(frontier_limit)
            if frontier_limit <= 0.0:
                frontier_limit = kw.get("time_limit_s")
            elif kw.get("time_limit_s") is not None:
                frontier_limit = min(frontier_limit, float(kw["time_limit_s"]))
        kw["time_limit_s"] = frontier_limit
        kw["solve_scope"] = "coverage-only"
    return kw


def _e1_certify_solver_kwargs(args):
    """Long-clock full lexicographic solve used only for formal knee candidates."""
    kw = _anytime_solver_kwargs(args)
    cert_limit = getattr(args, "e1_certify_time_limit_s", None)
    if cert_limit is not None:
        cert_limit = float(cert_limit)
        if cert_limit > 0.0:
            kw["time_limit_s"] = cert_limit
    kw["solve_scope"] = "lexicographic"
    return kw


def _e1_coverage_certify_solver_kwargs(args):
    """Long-clock Stage-1 solve for only the coverage-bound cells that block selection."""
    kw = _anytime_solver_kwargs(args)
    cert_limit = getattr(args, "e1_certify_time_limit_s", None)
    if cert_limit is not None:
        cert_limit = float(cert_limit)
        if cert_limit > 0.0:
            kw["time_limit_s"] = cert_limit
    kw["solve_scope"] = "coverage-only"
    return kw



def _e1_target_solver_kwargs(args, target):
    """Long-clock exact fixed-coverage decision; user gaps cannot certify."""
    kw = _anytime_solver_kwargs(args)
    cert_limit = getattr(args, "e1_certify_time_limit_s", None)
    if cert_limit is not None:
        cert_limit = float(cert_limit)
        if cert_limit > 0.0:
            kw["time_limit_s"] = cert_limit
    kw["coverage_gap_target_abs"] = 0
    kw["energy_gap_target_rel"] = 0.0
    kw["energy_gap_target_abs_Wh"] = 0.0
    kw["solve_scope"] = "coverage-target"
    kw["coverage_target"] = int(target)
    return kw


def _apply_target_decision_to_frontier(df, uav, K, B, target, result):
    """Apply only a certified target YES/NO to one rigorous Stage-1 interval."""
    mask = ((df["uav"].astype(str) == str(uav))
            & (pd.to_numeric(df["K"], errors="coerce") == int(K))
            & (pd.to_numeric(df["batteries"], errors="coerce") == int(B)))
    idxs = list(df.index[mask])
    if len(idxs) != 1:
        raise RuntimeError(f"target refine cell must be unique: {(uav, K, B)}")
    idx = idxs[0]
    before = _e1_raw_coverage_interval_record(df.loc[idx])
    if before is None:
        raise RuntimeError(f"target refine cell has no rigorous interval: {(uav, K, B)}")
    lb, ub = map(int, before)
    valid, certified = _strict_certificate_bool(result.get("target_decision_certified", False))
    if not (valid and certified):
        # An unresolved target solve has no effect on the rigorous interval.
        if "coverage_incumbent_refined" not in df.columns or pd.isna(df.loc[idx, "coverage_incumbent_refined"]):
            df.loc[idx, "coverage_incumbent_refined"] = lb
        if "coverage_upper_bound_refined" not in df.columns or pd.isna(df.loc[idx, "coverage_upper_bound_refined"]):
            df.loc[idx, "coverage_upper_bound_refined"] = ub
        return False

    decision = str(result.get("target_decision", "")).strip().upper()
    if decision == "FEASIBLE":
        ok, proven = _strict_certificate_bool(result.get("target_feasible_proven", False))
        if not (ok and proven):
            raise RuntimeError("certified TARGET_FEASIBLE lacks physical witness proof")
        if int(target) > ub:
            raise RuntimeError("target witness contradicts existing rigorous upper bound")
        new_lb, new_ub = max(lb, int(target)), ub
    elif decision == "INFEASIBLE":
        ok, proven = _strict_certificate_bool(result.get("target_infeasible_proven", False))
        if not (ok and proven):
            raise RuntimeError("certified TARGET_INFEASIBLE lacks full-space proof")
        if lb >= int(target):
            raise RuntimeError("target infeasibility contradicts existing rigorous incumbent")
        new_lb, new_ub = lb, min(ub, int(target) - 1)
    else:
        raise RuntimeError(f"certified target decision has invalid label {decision!r}")
    if not (0 <= int(new_lb) <= int(new_ub)):
        raise RuntimeError("target refinement produced an inconsistent rigorous interval")
    df.loc[idx, "coverage_incumbent_refined"] = int(new_lb)
    df.loc[idx, "coverage_upper_bound_refined"] = int(new_ub)
    df.loc[idx, "target_coverage_last"] = int(target)
    df.loc[idx, "target_decision_last"] = decision
    df.loc[idx, "target_decision_certified"] = True
    df.loc[idx, "target_certificate_type"] = result.get("target_certificate_type")
    df.loc[idx, "target_result_certificate_contract"] = result.get("result_certificate_contract")
    df.loc[idx, "target_formal_proof_contract"] = result.get("formal_proof_contract")
    df.loc[idx, "target_proof_contract_sha256"] = result.get("proof_contract_sha256")
    df.loc[idx, "target_runtime_s"] = result.get("runtime_s")
    df.loc[idx, "target_master_backend"] = result.get("target_master_backend")
    df.loc[idx, "target_master_solves"] = result.get("target_master_solves")
    df.loc[idx, "target_global_battery_relaxation_status"] = result.get(
        "target_global_battery_relaxation_status")
    df.loc[idx, "target_global_battery_min_required"] = result.get(
        "target_global_battery_min_required")
    df.loc[idx, "target_global_battery_dp_states"] = result.get(
        "target_global_battery_dp_states")
    df.loc[idx, "target_global_battery_one_pack_masks"] = result.get(
        "target_global_battery_one_pack_masks")
    return True


def _e1_target_blockers(df, uav, selection_row, order="BK"):
    """Return at most the immediate monotone predecessor that still permits target T."""
    status = str(selection_row.get("selection_status", ""))
    if status != "uncertified_resource_knee":
        return []
    raw_t = selection_row.get("coverage_threshold")
    if raw_t is None or pd.isna(raw_t):
        return []
    target = int(raw_t)
    sub = df[df["uav"].astype(str) == str(uav)].copy()
    if sub.empty:
        return []
    Ks = sorted({int(x) for x in pd.to_numeric(sub["K"], errors="coerce").dropna()})
    Bs = sorted({int(x) for x in pd.to_numeric(sub["batteries"], errors="coerce").dropna()})
    if not Ks or not Bs:
        return []

    def iv(k, b):
        q = _e1_monotone_coverage_interval(sub, int(k), int(b))
        return q

    order = str(order).upper()
    if order == "BK":
        kmax = max(Ks)
        feasible_b = [b for b in Bs if iv(kmax, b) is not None and iv(kmax, b)[0] >= target]
        if not feasible_b:
            return []
        bstar = min(feasible_b)
        prev_bs = [b for b in Bs if b < bstar]
        if prev_bs:
            bprev = max(prev_bs)
            q = iv(kmax, bprev)
            if q is not None and q[1] >= target:
                return [(kmax, bprev, target, "B-predecessor")]
        feasible_k = [k for k in Ks if iv(k, bstar) is not None and iv(k, bstar)[0] >= target]
        if not feasible_k:
            return []
        kstar = min(feasible_k)
        prev_ks = [k for k in Ks if k < kstar]
        if prev_ks:
            kprev = max(prev_ks)
            q = iv(kprev, bstar)
            if q is not None and q[1] >= target:
                return [(kprev, bstar, target, "K-predecessor")]
        return []
    if order == "KB":
        bmax = max(Bs)
        feasible_k = [k for k in Ks if iv(k, bmax) is not None and iv(k, bmax)[0] >= target]
        if not feasible_k:
            return []
        kstar = min(feasible_k)
        prev_ks = [k for k in Ks if k < kstar]
        if prev_ks:
            kprev = max(prev_ks)
            q = iv(kprev, bmax)
            if q is not None and q[1] >= target:
                return [(kprev, bmax, target, "K-predecessor")]
        feasible_b = [b for b in Bs if iv(kstar, b) is not None and iv(kstar, b)[0] >= target]
        if not feasible_b:
            return []
        bstar = min(feasible_b)
        prev_bs = [b for b in Bs if b < bstar]
        if prev_bs:
            bprev = max(prev_bs)
            q = iv(kstar, bprev)
            if q is not None and q[1] >= target:
                return [(kstar, bprev, target, "B-predecessor")]
        return []
    raise ValueError("knee order must be BK or KB")


def _e1_build_formal_warmstart(reach, opts, p_u, xi_amb, wamb, args, T_eff, cap_u):
    """Heuristic candidate finder only; step12 revalidates every returned seed."""
    warm_s = max(0.0, float(getattr(args, "formal_warmstart_seconds", 60.0)))
    if warm_s <= 0.0:
        return None
    try:
        cols = RA.build_route_columns(
            reach, opts, p_u, xi_amb, T_eff, args.deck_delta_min,
            cap_u, wamb, "drcc", 2.0, "vp_unimodal", 8.0,
            pool_h_mode=getattr(args, "pool_h", "pareto"),
            diagnostics_sink=None, deadline=time.monotonic() + warm_s)
        return list(cols) if cols else None
    except TimeoutError:
        return None
    except Exception as exc:
        log.warning("formal target warmstart candidate finder failed for %s: %s",
                    getattr(p_u, "uav_key", "?"), type(exc).__name__)
        return None


def _validate_e1_frontier_for_target_refine(df, args, current_resume_sha):
    """Accept compatible prior frontier bounds only when physical/data identity is unchanged."""
    if df is None or df.empty:
        raise SystemExit("E1_knee_refine requires a nonempty E1_frontier.csv")
    if "study_mode" in df.columns:
        modes = {str(x) for x in df["study_mode"].dropna().unique()}
        if modes != {"formal"}:
            raise SystemExit(f"E1_knee_refine only accepts formal frontier, got {modes}")
    # Formal refinement may never accumulate proof across result/scheduler
    # contracts.  Older v9/v10/v11/v12 frontiers are diagnostics only.
    allowed = {RESULT_CONTRACT}
    if "result_contract" not in df.columns:
        raise SystemExit("E1_knee_refine frontier lacks result_contract")
    contracts = {str(x) for x in df["result_contract"].dropna().unique()}
    if not contracts or not contracts <= allowed:
        raise SystemExit(f"E1_knee_refine rejects frontier result contracts {contracts}")
    expected = {
        "physical_numeric_contract": RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
        "route_identity_contract": BP.ROUTE_IDENTITY_CONTRACT,
        "model_semantics_contract": BP.MODEL_SEMANTICS_CONTRACT,
        "formal_experiment_scheduler_contract": FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
        "resume_input_sha256": str(current_resume_sha),
        "xi_mmsi": str(getattr(args, "_resolved_xi_mmsi", "")),
        "validation_samples_hash": (EU.sha256_file(getattr(args, "validation_samples", None)) or "none"),
        "xi_train_samples_hash": (EU.sha256_file(getattr(args, "xi_train_samples", None)) or "none"),
        "final_test_samples_hash": (EU.sha256_file(getattr(args, "final_test_samples", None)) or "none"),
    }
    for col, want in expected.items():
        if col not in df.columns:
            raise SystemExit(f"E1_knee_refine frontier lacks provenance column {col}")
        vals = [str(x) for x in df[col].dropna().unique()]
        if len(vals) != 1 or vals[0] != str(want):
            raise SystemExit(
                f"E1_knee_refine provenance mismatch {col}: file={vals!r} current={want!r}")
    return True


def _e1_final_test_consumption_allowed(args) -> bool:
    """Formal E1 never consumes the independent final-test holdout."""
    return bool(
        str(getattr(args, "study_mode", "mechanism")).lower() != "formal"
        and getattr(args, "final_test_samples", None) is not None)


def _formal_statistic_value(value):
    """Keep gate-critical statistics at full binary64 precision; round only display copies."""
    return None if value is None else float(value)


def _formal_validation_selection_gate(replay_result):
    """Formal selection keeps the predeclared internal 0.045 allocation gate."""
    return replay_result.get("allocation_budget_holds")


def _e1_raw_coverage_interval_record(rec):
    """Return the rigorous Stage-1 interval, including certified target refinements.

    Pandas creates NaN in untouched rows after the first targeted refinement.
    NaN therefore means "no refinement" and must fall back to the original
    rigorous Stage-1 bound rather than erasing evidence.
    """
    try:
        raw_lb_ref = rec.get("coverage_incumbent_refined", None)
        raw_lb = (rec.get("coverage_incumbent", rec.get("covered", 0))
                  if raw_lb_ref is None or pd.isna(raw_lb_ref) else raw_lb_ref)
        lb = int(raw_lb or 0)
        raw_ub_ref = rec.get("coverage_upper_bound_refined", None)
        raw_ub = (rec.get("coverage_upper_bound", None)
                  if raw_ub_ref is None or pd.isna(raw_ub_ref) else raw_ub_ref)
        ub = int(rec.get("coverable_note", lb)
                 if raw_ub is None or pd.isna(raw_ub) else raw_ub)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return (lb, ub) if 0 <= lb <= ub else None


def _e1_monotone_coverage_interval(df, K, B):
    """Rigorous 2-D resource-monotone envelope for formal E1 coverage bounds.

    If K'<=K and B'<=B then C*(K',B')<=C*(K,B); if K'>=K and B'>=B
    then C*(K,B)<=C*(K',B').  Therefore

        max L(K',B') <= C*(K,B) <= min U(K',B')

    over the corresponding southwest/northeast resource orthants. Missing rows
    never create evidence. This helper only tightens bounds; it does not mark
    any optimization as closed.
    """
    K, B = int(K), int(B)
    lows, ups = [], []
    for _, rec in df.iterrows():
        q = _e1_raw_coverage_interval_record(rec)
        if q is None:
            continue
        try:
            kk, bb = int(rec["K"]), int(rec["batteries"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if kk <= K and bb <= B:
            lows.append(int(q[0]))
        if kk >= K and bb >= B:
            ups.append(int(q[1]))
    if not lows or not ups:
        return None
    lb, ub = max(lows), min(ups)
    return (lb, ub) if 0 <= lb <= ub else None


def _e1_bound_strictly_improved(before, after):
    """Whether a certification run produced rigorous interval information gain."""
    return bool(
        before is not None and after is not None
        and (int(after[0]) > int(before[0]) or int(after[1]) < int(before[1])))


def _e1_plateau_long_refinement_allowed(e1_b_auto):
    """Fixed-grid formal runs must not waste long solves trying to extend a B tail."""
    return str(e1_b_auto).lower() == "on"


def _formal_resource_knee_generic_refinement_allowed(selection_row):
    """Whether generic long max-coverage refinement can still add formal evidence.

    Once the hard coverable cap certifies the plateau, the threshold T is fixed.
    In that state an exact predecessor target decision is strictly more relevant
    than re-solving generic max-coverage and v12 must route directly to it.
    """
    if selection_row is None:
        return False
    if str(selection_row.get("selection_status", "")) != "uncertified_resource_knee":
        return False
    return str(selection_row.get("saturation_proof", "")) != "hard-coverable-cap"


def _e2_solver_kwargs(args, q, quantiles):
    """Formal E2: short diagnostic clocks below q_max, long exact clock at q_max.

    Only the harshest configured weather quantile is eligible for final-test
    freezing, so lower-q rows are diagnostics and may remain anytime results.
    Harshest-q formal rows must close the full lexicographic certificate before
    ``run_status=ok``.  This changes scheduling only, never certificate meaning.
    """
    kw = _anytime_solver_kwargs(args)
    if (str(getattr(args, "study_mode", "formal")) == "formal"
            and _formal_ondemand_pricing(args)):
        qmax = max(float(x) for x in quantiles)
        harshest = float(q) == float(qmax)
        limit = (getattr(args, "e2_certify_time_limit_s", None) if harshest
                 else getattr(args, "e2_discovery_time_limit_s", 120.0))
        if limit is not None:
            limit = float(limit)
            if limit <= 0.0:
                limit = kw.get("time_limit_s")
            elif (not harshest) and kw.get("time_limit_s") is not None:
                limit = min(limit, float(kw["time_limit_s"]))
            kw["time_limit_s"] = limit
    kw["solve_scope"] = "lexicographic"
    return kw


def _implicit_route_space_certificate(result):
    """Compatibility accessor; always identical to the canonical certificate."""
    return _normalized_certificate_evidence(result)[0]


def _certificate_field_conflict(result):
    """True when present certificate aliases disagree with the canonical field."""
    return _normalized_certificate_evidence(result)[1]


def _certificate_field_invalid(result):
    """True when any persisted certificate field has an invalid non-boolean value."""
    return _normalized_certificate_evidence(result)[2]


def _route_pool_metadata(formal_ondemand, prebuilt_columns, solver_result):
    """Truthful route-space metadata for one E1 cell."""
    if formal_ondemand:
        count = int(solver_result.get(
            "generated_column_archive_size",
            solver_result.get("pool_size", 0)) or 0)
        if bool(solver_result.get("route_space_materialized", False)):
            return "complete_materialized_physical_route_space", count
        return "on_demand_implicit_route_space", count
    return (("nonempty" if len(prebuilt_columns) else "empty_all_routes_infeasible"),
            int(len(prebuilt_columns)))


def _zero_coverage_reason(formal_ondemand, prebuilt_columns, solver_result, batteries):
    """Classify genuine zero coverage without calling on-demand BPC an empty pool."""
    covered = int(solver_result.get(
        "covered", solver_result.get("coverage_incumbent", 0)) or 0)
    if covered > 0:
        return None
    if int(batteries) == 0:
        return "zero_battery_anchor"
    if formal_ondemand:
        if (bool(solver_result.get("coverage_optimal", False))
                and int(solver_result.get("coverage_upper_bound", 1) or 0) == 0):
            return ("complete_materialized_route_space_zero_coverage_proven"
                    if bool(solver_result.get("route_space_materialized", False))
                    else "implicit_route_space_zero_coverage_proven")
        return ("complete_materialized_solver_returned_zero_coverage"
                if bool(solver_result.get("route_space_materialized", False))
                else "on_demand_solver_returned_zero_coverage")
    if not prebuilt_columns:
        return "empty_route_pool"
    return "resource_assignment_zero"


def _e1_route_space_is_empty(sub):
    """Only a genuinely empty prebuilt research pool counts as empty."""
    if "route_pool_status" in sub.columns:
        statuses = {str(v) for v in sub["route_pool_status"].dropna()
                    if str(v).strip()}
        if "on_demand_implicit_route_space" in statuses:
            return False
        if statuses:
            return statuses == {"empty_all_routes_infeasible"}
    return bool("route_pool_count" in sub.columns
                and int(pd.to_numeric(sub["route_pool_count"], errors="coerce")
                        .fillna(0).max()) == 0)


def attach_per_turbine_weather(turbines, time_str):
    """把逐风机天气挂到风机对象 .wx_local(model.md §15 消费端)。匹配到与 time_str 最近的小时。"""
    here = Path(__file__).resolve().parent
    wind_pt = here / "weather" / "weather_per_turbine_Rodsand_II.csv"
    wave_pt = here / "weather" / "waves_per_turbine_Rodsand_II.csv"
    if not wind_pt.exists() and not wave_pt.exists():
        log.warning("未找到逐风机天气文件, 退回 farm 级天气。")
        return 0
    try:
        t0 = pd.to_datetime(time_str)
    except Exception:
        t0 = None

    def _slice_nearest(csv):
        if not csv.exists():
            return {}, None
        df = pd.read_csv(csv); df["time"] = pd.to_datetime(df["time"])
        tt = t0 if t0 is not None else df["time"].iloc[len(df) // 2]
        uniq = df["time"].drop_duplicates()
        tsel = uniq.iloc[(uniq - tt).abs().argsort().iloc[0]]
        sub = df[df["time"] == tsel]
        return {str(r["turbine_id"]): r for _, r in sub.iterrows()}, str(tsel)

    wmap, twind = _slice_nearest(wind_pt) if wind_pt.exists() else ({}, None)
    vmap, twave = _slice_nearest(wave_pt) if wave_pt.exists() else ({}, None)
    n = 0
    for t in turbines:
        tid = str(t.tid); loc = {}
        if tid in wmap:
            u = float(wmap[tid].get("u10", np.nan)); v = float(wmap[tid].get("v10", np.nan))
            if not (math.isnan(u) or math.isnan(v)):
                loc["wind10"] = float(math.hypot(u, v))
                loc["wind_dir_from"] = float((270.0 - math.degrees(math.atan2(v, u))) % 360.0)
        if tid in vmap:
            r = vmap[tid]
            if "VHM0" in r and not pd.isna(r["VHM0"]): loc["Hs"] = float(r["VHM0"])
            if "VTM02" in r and not pd.isna(r["VTM02"]): loc["Tp"] = float(r["VTM02"])
            if "VMDR" in r and not pd.isna(r["VMDR"]): loc["wave_dir"] = float(r["VMDR"])
        if loc:
            t.wx_local = loc; n += 1
    log.info("逐风机天气已挂载 %d/%d 台 (风@%s, 浪@%s)", n, len(turbines), twind, twave)
    return n


def pick_weather(wx_df, hs_quantile=None):
    """取一个代表性天气样点; hs_quantile 给定则取 Hs 接近该分位的小时。"""
    valid = wx_df.dropna(subset=["wind10_ms"]) if "wind10_ms" in wx_df.columns else wx_df
    base = valid if len(valid) else wx_df
    if hs_quantile is not None and "Hs_m" in base.columns and base["Hs_m"].notna().any():
        target = float(base["Hs_m"].quantile(hs_quantile))
        row = base.iloc[(base["Hs_m"] - target).abs().argsort().iloc[0]]
    else:
        row = base.iloc[len(base) // 2]

    def _f(v, d):
        try:
            return float(v) if not pd.isna(v) else d
        except Exception:
            return d
    return dict(wind10=_f(row.get("wind10_ms"), 6.7), wind_dir_from=_f(row.get("wind_dir_from_deg"), 230.0),
                Hs=_f(row.get("Hs_m"), 0.5), Tp=_f(row.get("wave_Tm_s"), 2.1),
                wave_dir=_f(row.get("wave_dir_deg"), 200.0), ship_heading=90.0, time=str(row.name))


def build_ship(turbines, lat0lon0, sc_csv, xi_amb, pair_radius, recovery_state="动力定位"):
    """构造代表性起飞事件的船位预测 + 就近可达风机集。

    重要(无泄漏): ShipPrediction.c_state = c(τ) 是【起飞/决策时刻 τ 可观测】的船舶状态,
    用作 ξ 模糊集索引 𝒫_{h,c(τ)} —— 与 step7 按预测起点状态归组 ξ 矩一致, 不使用回收时刻
    真实状态 c(τ+h)(那会泄漏未来)。回收处能否着舰由【着舰门】(海况)判定, 与 ξ 索引解耦。
    参数 recovery_state 在此【作为选取代表性起飞事件的偏好状态】: 默认挑选【起飞时处于受控
    低运动态(动力定位/低速)】的事件 —— 这类时刻船位可预测性好(ξ 小), 是合理的代表性作业窗;
    并非用回收时刻状态索引 ξ。"""
    lat0, lon0 = lat0lon0
    horizons = sorted(xi_amb.horizons)
    states_in_amb = {c for (_, c) in xi_amb.cells}
    calm_pref = [recovery_state, "动力定位", "低速"]
    launch_state = next((s for s in calm_pref if s in states_in_amb), recovery_state)

    if sc_csv and sc_csv.is_file():
        sc = pd.read_csv(sc_csv)
        if "c_state" in sc.columns:
            cand = sc[sc["c_state"] == launch_state]
            if cand.empty:
                cand = sc[sc["c_state"].isin(["动力定位", "低速"])]
            cand = cand if not cand.empty else sc
        else:
            cand = sc
        r0 = cand.iloc[len(cand) // 2]
        P_launch = M.latlon_to_local_m(float(r0["launch_lat"]), float(r0["launch_lon"]), lat0, lon0)
        P_pred = M.latlon_to_local_m(float(r0["pred_recover_lat"]), float(r0["pred_recover_lon"]), lat0, lon0)
        h0 = max(float(r0["h_min"]), 1.0)
        v_ship = (P_pred - P_launch) / (h0 * 60.0)
        c_state = str(r0.get("c_state", launch_state))
        ship = RM.ShipPrediction.from_cv(P_launch, v_ship, horizons, c_state)
        geom = f"半真实几何(真实起飞点 {sc_csv.name}, 起飞时刻状态 c(τ)={c_state}, CV 外推到细 h)"
    else:
        centroid = np.mean([t.local for t in turbines], axis=0)
        P_launch = centroid + np.array([-700.0, -500.0])
        ship = RM.ShipPrediction.from_cv(P_launch, np.array([1.2, 0.9]), horizons, launch_state)
        geom = f"占位几何(合成质心起飞 + 低速航迹, 起飞时刻状态 c(τ)={launch_state})"
    reach = [t for t in turbines if np.linalg.norm(t.local - ship.P_launch) <= pair_radius]
    if not reach:
        reach = turbines
    return ship, reach, geom


def _synth_transit_track(turbines, n_pts=60, speed_mps=3.0, margin_m=400.0):
    """合成穿场航迹(无真实 AIS 文件时回退): 沿风机布局 PCA 主轴匀速穿过整个风场。返回 ShipTrack。"""
    P = np.array([t.local for t in turbines], float); ctr = P.mean(axis=0); Q = P - ctr
    try:
        _u, _s, _vt = np.linalg.svd(Q, full_matrices=False); axis = _vt[0]
    except Exception:
        axis = np.array([1.0, 0.0])
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    proj = Q @ axis; half = float(max(abs(proj.min()), abs(proj.max())))
    start = ctr - axis * (half + margin_m); end = ctr + axis * (half + margin_m)
    total_m = float(np.linalg.norm(end - start)); dur = total_m / max(speed_mps, 1e-6)
    ts = np.linspace(0.0, dur, n_pts)
    pts = np.array([start + (end - start) * (t / max(dur, 1e-9)) for t in ts])
    return M.ShipTrack(ts, pts)


def _save(df, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("写 %s (%d 行)", path, len(df))
    return path


# =============================================================================
# A1 验证: out-of-sample 违反率(把"名义/高斯不可行会违反约束"变成实证)
# =============================================================================
def _dr_feasible_singletons(turbines, ship, p, wx, amb, max_stops=1):
    """该模糊集 `amb` 下【可作单台 DR 可行路由】的风机 tid 集合(用于求两方案的公共可行子集)。"""
    feas = set()
    for t in turbines:
        r = RM.Route(rid=-1, turbines=[t], ship=ship)
        d = RM.route_drcc_feasible(r, p, wx, amb, objective="min_h")
        if d["feasible"]:
            feas.add(t.tid)
    return feas


def _nominal_ambiguity(xi_amb):
    """构造"名义模糊集": Σ 置零、仅保留 μ。这样 SOC 项 κ√(aΣa)=0, 退化为名义点 ξ=μ 可行。"""
    cells = {}
    for (h, c), cell in xi_amb.cells.items():
        cells[(h, c)] = M.XiCell(h_min=cell.h_min, c_state=cell.c_state, n=cell.n,
                                 mu=cell.mu, Sigma=np.zeros((2, 2)),
                                 support_radius=0.0, p95_norm=0.0, rms_norm=0.0)
    return M.XiAmbiguity(cells, list(xi_amb.horizons))


def _gauss_kappa_patch(eps):
    """高斯 κ=Φ⁻¹(1-ε)(用 scipy 或近似)。"""
    try:
        from scipy.stats import norm
        return float(norm.ppf(1 - eps))
    except Exception:
        # 近似(Acklam): ε=.05→1.645, .025→1.96, .01→2.326
        table = {0.20: 0.842, 0.10: 1.282, 0.05: 1.645, 0.025: 1.960, 0.01: 2.326}
        return table.get(round(eps, 3), 1.645)


def _merged_state_ambiguity(xi_amb):
    r"""把 (h,c) 模糊集按 h 合并(忽略状态 c): 对每个 h, 用【该 h 下所有状态样本汇合】成的单一矩
    (μ,Σ), 复制给所有状态键。模拟"不分状态"的粗模糊集, 对比状态分组 c 的价值。

    **更新 审计修复(关键)**: 旧实现"取该 h 下样本量最大的状态作代表"——当【作业实际使用的
    状态】恰为最大样本状态时(真实数据里 `动力定位` 往往既样本最多又是代表作业态), 合并矩 ≡ 分状态矩,
    消融退化为【恒等无效】(上传结果中 by_state 与 merged 逐位相同即此因)。现改为按【全状态样本汇合】
    的正确"状态无关"矩(总方差律 / law of total covariance):
      $N=\sum_c n_c$,  $\mu=\frac{1}{N}\sum_c n_c\mu_c$,
      $\Sigma=\frac{1}{N}\sum_c n_c(\Sigma_c+\mu_c\mu_c^\top)-\mu\mu^\top$  (含状态间均值离散度)。
    汇合矩含【状态间差异】, 一般比"作业态(动力定位, 通常最可预测 ⇒ Σ 最小)"的分状态矩更松 ⇒
    消融能稳定显示"忽略状态会放大模糊集、更保守(覆盖更少)", 与数据中哪个状态样本最多无关。
    支撑半径/p95/rms 取各状态最大(保守包络)。"""
    import numpy as _np
    cells = {}
    horizons = sorted(xi_amb.horizons)
    states = sorted({c for (_, c) in xi_amb.cells})
    for h in horizons:
        cand = [(hh, cc) for (hh, cc) in xi_amb.cells if hh == h]
        if not cand:
            continue
        ns = _np.array([float(max(xi_amb.cells[k].n, 0)) for k in cand])
        if ns.sum() <= 0:
            ns = _np.ones(len(cand))
        w = ns / ns.sum()
        mus = [_np.asarray(xi_amb.cells[k].mu, float).reshape(-1) for k in cand]
        Sigs = [_np.asarray(xi_amb.cells[k].Sigma, float) for k in cand]
        mu_pool = sum(wi * mi for wi, mi in zip(w, mus))
        # 二阶原点矩汇合 + 总方差律(含状态间均值离散)
        second = sum(wi * (Si + _np.outer(mi, mi)) for wi, Si, mi in zip(w, Sigs, mus))
        Sig_pool = second - _np.outer(mu_pool, mu_pool)
        Sig_pool = 0.5 * (Sig_pool + Sig_pool.T)            # 数值对称化
        n_pool = int(ns.sum())
        sr = max(float(xi_amb.cells[k].support_radius) for k in cand)
        p95 = max(float(getattr(xi_amb.cells[k], "p95_norm", 0.0)) for k in cand)
        rms = max(float(getattr(xi_amb.cells[k], "rms_norm", 0.0)) for k in cand)
        for c in states:
            cells[(h, c)] = M.XiCell(h_min=h, c_state=c, n=n_pool, mu=mu_pool,
                                     Sigma=Sig_pool, support_radius=sr,
                                     p95_norm=p95, rms_norm=rms)
    return M.XiAmbiguity(cells, horizons)


def _median_state_ambiguity(xi_amb):
    r"""【温和的 state-agnostic 对照】对每个 h, 用各状态【协方差的样本量加权平均】Σ_avg=Σ_c w_c Σ_c
    (μ 取加权均值), 但**不含**状态间均值离散项。
    与 `_merged_state_ambiguity`(全汇合 Σ_pool = Σ_avg + 状态间均值离散, 最保守)相比, Σ_avg 恒 ≤ Σ_pool;
    又因平均含较大 ξ 的状态, 一般 ≥ 作业态(动力定位)的 Σ ⇒ 排序 by_state ≤ cov_avg ≤ pool_all,
    给出"忽略状态"代价的【温和中段】, 使消融呈区间而非单一极端。"""
    import numpy as _np
    cells = {}
    horizons = sorted(xi_amb.horizons)
    states = sorted({c for (_, c) in xi_amb.cells})
    for h in horizons:
        cand = [(hh, cc) for (hh, cc) in xi_amb.cells if hh == h]
        if not cand:
            continue
        ns = _np.array([float(max(xi_amb.cells[k].n, 0)) for k in cand])
        if ns.sum() <= 0:
            ns = _np.ones(len(cand))
        w = ns / ns.sum()
        mus = [_np.asarray(xi_amb.cells[k].mu, float).reshape(-1) for k in cand]
        Sigs = [_np.asarray(xi_amb.cells[k].Sigma, float) for k in cand]
        mu_avg = sum(wi * mi for wi, mi in zip(w, mus))
        Sig_avg = sum(wi * Si for wi, Si in zip(w, Sigs))        # 仅状态内协方差均值, 无状态间离散
        Sig_avg = 0.5 * (Sig_avg + Sig_avg.T)
        n_pool = int(ns.sum())
        sr = max(float(xi_amb.cells[k].support_radius) for k in cand)
        p95 = max(float(getattr(xi_amb.cells[k], "p95_norm", 0.0)) for k in cand)
        rms = max(float(getattr(xi_amb.cells[k], "rms_norm", 0.0)) for k in cand)
        for c in states:
            cells[(h, c)] = M.XiCell(h_min=h, c_state=c, n=n_pool, mu=mu_avg, Sigma=Sig_avg,
                                     support_radius=sr, p95_norm=p95, rms_norm=rms)
    return M.XiAmbiguity(cells, horizons)



# =============================================================================
# 时空耦合机队模型实验套件（仅三个模型实验）
#   E1_frontier  UAV×K×B 三轴前沿(更新)   E2_robust  鲁棒方法对照(7 判据×多窗, 去 adaptive)
#   旧 exp1~8/dro/multiweather/co-timing 已按作者"直接去掉"决策整体删除(其机制或被 吸收为
#   模型本身(τ,ω,h 列/时变天气), 或由 E1/E2/E3 取代); 历史 CSV 与本套件不可混排。
# =============================================================================

def _wx_row_dict(row):
    """wx_df 单行 → 求解用天气 dict(与 pick_weather 同键同默认)。"""
    def _f(v, d):
        try:
            return float(v) if not pd.isna(v) else d
        except Exception:
            return d
    return dict(wind10=_f(row.get("wind10_ms"), 6.7), wind_dir_from=_f(row.get("wind_dir_from_deg"), 230.0),
                Hs=_f(row.get("Hs_m"), 0.5), Tp=_f(row.get("wave_Tm_s"), 2.1),
                wave_dir=_f(row.get("wave_dir_deg"), 200.0), ship_heading=90.0, time=str(row.name))


def _wx_series(wx_df, hs_quantile=0.5, absolute_start=None, alignment="representative_quantile",
               past_only_anchor=False):
    """Return a complete UTC weather timeline, its scenario anchor and alignment metadata.

    Wind/wave rows with any required missing value are excluded explicitly and counted.  Timestamp
    mode anchors the mission at its absolute UTC start; representative mode anchors at the selected
    Hs quantile but still advances by true elapsed time rather than integer-hour row offsets.
    """
    required = ["wind10_ms", "wind_dir_from_deg", "Hs_m", "wave_Tm_s", "wave_dir_deg"]
    missing = [c for c in required if c not in wx_df.columns]
    if missing:
        raise ValueError(f"天气表缺少必要列: {missing}")
    base = wx_df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(base.index, utc=True, errors="raise"))
    if idx.has_duplicates:
        dup = idx[idx.duplicated()].unique()[:5]
        raise ValueError(f"天气表存在重复 UTC 时间戳，例如 {list(map(str, dup))}")
    base.index = idx
    base = base.sort_index()
    for col in required:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    complete_mask = base[required].notna().all(axis=1)
    incomplete = int((~complete_mask).sum())
    base = base.loc[complete_mask].copy()
    if base.empty:
        raise ValueError("天气表没有风浪字段均完整的 UTC 行。")

    mode = str(alignment).strip().lower()
    if mode == "timestamp":
        if absolute_start is None:
            raise ValueError("weather_alignment=timestamp 需要带可解析 UTC 时间的 AIS 航迹")
        target = pd.Timestamp(absolute_start)
        target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
        if past_only_anchor:
            # Formal information-set anchor: nearest is unsafe because it may
            # silently pick a reanalysis row after the decision time.
            pos = int(base.index.searchsorted(target, side="right")) - 1
        else:
            pos = int(base.index.get_indexer([target], method="nearest")[0])
        if pos < 0:
            raise ValueError("天气时间轴在任务 UTC 起点之前没有可用完整观测。")
        delta_s = float((base.index[pos] - target).total_seconds())
        if past_only_anchor and delta_s > 1e-9:
            raise AssertionError("formal weather anchor used future information")
        err_min = abs(delta_s) / 60.0
        if err_min > 90.0:
            raise ValueError(f"任务 UTC 起点与最近允许天气观测相差 {err_min:.1f} min，超过 90 min。")
        start = pos
        meta = dict(weather_alignment_mode="timestamp", weather_target_time=str(target),
                    weather_start_time=str(base.index[start]),
                    weather_match_error_min=round(err_min, 3), hs_quantile=None)
    elif mode == "representative_quantile":
        target_hs = float(base["Hs_m"].quantile(hs_quantile))
        start = int((base["Hs_m"] - target_hs).abs().to_numpy().argmin())
        meta = dict(weather_alignment_mode="representative_quantile_scenario",
                    weather_target_time=None, weather_start_time=str(base.index[start]),
                    weather_match_error_min=None, hs_quantile=float(hs_quantile),
                    target_Hs=round(target_hs, 6))
    else:
        raise ValueError(f"未知 weather alignment: {alignment}")
    meta.update(weather_rows_total=int(len(wx_df)), weather_rows_complete=int(len(base)),
                weather_rows_incomplete=int(incomplete), weather_timezone="UTC")
    return base, start, meta


def _launch_asof_weather_forecaster(rows, weather_anchor, mission_origin_sec,
                                    max_gap_min=90.0):
    """Build the formal launch-asof weather forecast used by route columns.

    The implementation intentionally reuses the exact backward-linear predictor
    that Step7 used to create the weather residuals/moments. Every target at
    ``issue_sec`` is therefore a function only of observations at/before issue.
    Realized future reanalysis remains available to replay, never to planning.
    """
    import step7_compute_xi as S7
    base = rows.copy().sort_index()
    times = S7.to_epoch_seconds_utc(base.index).astype(float)
    wind_speed = pd.to_numeric(base["wind10_ms"], errors="raise").to_numpy(float)
    wind_dir = pd.to_numeric(base["wind_dir_from_deg"], errors="raise").to_numpy(float)
    hs = pd.to_numeric(base["Hs_m"], errors="raise").to_numpy(float)
    tp = pd.to_numeric(base["wave_Tm_s"], errors="raise").to_numpy(float)
    wave_dir = pd.to_numeric(base["wave_dir_deg"], errors="raise").to_numpy(float)
    wind_vec = S7._wind_vec(wind_speed, wind_dir)
    anchor = pd.Timestamp(weather_anchor)
    anchor = anchor.tz_localize("UTC") if anchor.tzinfo is None else anchor.tz_convert("UTC")
    max_gap_s = 60.0 * float(max_gap_min)

    def _forecast(issue_sec, target_sec):
        issue_abs = anchor + pd.to_timedelta(float(issue_sec) - float(mission_origin_sec), unit="s")
        target_abs = anchor + pd.to_timedelta(float(target_sec) - float(mission_origin_sec), unit="s")
        issue_epoch = float(S7.to_epoch_seconds_utc([issue_abs])[0])
        target_epoch = float(S7.to_epoch_seconds_utc([target_abs])[0])
        fc, reason = S7._forecast_weather_coherent(
            issue_epoch, target_epoch, times, wind_vec, wind_speed, hs, max_gap_s)
        if fc is None:
            raise ValueError(
                f"formal coherent weather forecast unavailable at issue={issue_abs}: {reason}; "
                "future weather fallback is forbidden")
        source_max = float(fc["observation_epoch"])
        if source_max > issue_epoch + 1e-9:
            raise AssertionError("formal weather forecast consumed a post-issue observation")
        j = int(np.searchsorted(times, issue_epoch, side="right")) - 1
        if j < 0:
            raise ValueError("formal weather forecast has no causal wave-state row")
        vec = np.asarray(fc["wind_vec"], float).reshape(2)
        pspd = float(fc["wind_speed"])
        phs = float(fc["Hs"])
        if pspd <= 1e-12:
            from_deg = 0.0  # direction is physically irrelevant at zero speed
        else:
            to_deg = (math.degrees(math.atan2(float(vec[0]), float(vec[1]))) + 360.0) % 360.0
            from_deg = (to_deg + 180.0) % 360.0
        if abs(float(np.linalg.norm(vec)) - pspd) > 1e-10 * max(1.0, pspd):
            raise AssertionError("formal weather vector/scalar nominal lost coherence")
        return dict(
            wind10=pspd, wind_dir_from=float(from_deg),
            Hs=phs, Tp=float(tp[j]), wave_dir=float(wave_dir[j]),
            ship_heading=90.0, time=str(issue_abs),
            weather_issue_time=str(issue_abs), weather_target_time=str(target_abs),
            weather_observation_epoch=float(source_max),
            weather_source_max_epoch=float(source_max), weather_issue_epoch=float(issue_epoch),
            weather_predictor="weather_speed_primary_coherent_noleak",
            weather_predictor_contract=RM.WEATHER_PREDICTOR_CONTRACTS[
                "weather_speed_primary_coherent_noleak"],
            weather_information_scope="launch-asof-target-forecast",
            weather_default_used=False)

    return _forecast


def _formal_past_quantile_weather_provider(rows, cutoff_abs, hs_quantile=0.5):
    """Past-only constant scenario provider for the formal E2 quantile axis."""
    import step7_compute_xi as S7
    cutoff = pd.Timestamp(cutoff_abs)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    base = rows.copy().sort_index()
    base = base.loc[base.index <= cutoff].copy()
    if base.empty:
        raise ValueError("formal representative-weather scenario has no pre-cutoff history")
    target_hs = float(base["Hs_m"].quantile(float(hs_quantile)))
    pos = int((base["Hs_m"] - target_hs).abs().to_numpy().argmin())
    row = base.iloc[pos]
    source_time = base.index[pos]
    source_epoch = float(S7.to_epoch_seconds_utc([source_time])[0])
    cutoff_epoch = float(S7.to_epoch_seconds_utc([cutoff])[0])
    if source_epoch > cutoff_epoch + 1e-9:
        raise AssertionError("formal Hs quantile scenario used future weather")
    scenario = _wx_row_dict(row)
    scenario.update(
        time=str(source_time), weather_source_max_epoch=source_epoch,
        weather_issue_epoch=cutoff_epoch, weather_predictor="past_history_quantile_scenario",
        weather_predictor_contract="causal-past-history-quantile-scenario-v1",
        weather_information_scope="past-history-scenario-not-confirmatory-forecast",
        weather_default_used=False)

    def _provider(issue_sec, target_sec):
        out = dict(scenario)
        out["weather_issue_offset_sec"] = float(issue_sec)
        out["weather_target_offset_sec"] = float(target_sec)
        return out

    meta = dict(weather_alignment_mode="representative_quantile_past_only_scenario",
                weather_target_time=None, weather_start_time=str(source_time),
                weather_match_error_min=None, hs_quantile=float(hs_quantile),
                target_Hs=round(target_hs, 6), weather_rows_complete=int(len(base)),
                weather_rows_total=int(len(rows)), weather_rows_incomplete=0,
                weather_timezone="UTC", weather_source_max_epoch=source_epoch)
    return _provider, scenario, meta


def _load_track_ds(csv_path, lat0, lon0, min_dt_s=10.0, max_pts=60000):
    """更新: 读大 AIS(几十 MB/多日)并降采样到 ≥min_dt_s 间隔、≤max_pts 点。"""
    tr = M.load_ship_track(csv_path, lat0, lon0)
    if tr is None or len(tr) < 2:
        return None
    t, P = np.asarray(tr.t, float), np.asarray(tr.P, float)
    med = float(np.median(np.diff(t))) if len(t) > 1 else min_dt_s
    stride = max(int(np.ceil(min_dt_s / max(med, 1e-6))), int(np.ceil(len(t) / max_pts)), 1)
    if stride > 1:
        t, P = t[::stride], P[::stride]
        log.info("航迹降采样 ×%d: %d 点(中位间隔 %.1fs)", stride, len(t), float(np.median(np.diff(t))))
    abs_start = getattr(tr, "absolute_start", None)
    if abs_start is not None:
        abs_start = pd.Timestamp(abs_start) + pd.to_timedelta(float(t[0]), unit="s")
    return M.ShipTrack(t - t[0], P, absolute_start=abs_start,
                       time_source=getattr(tr, "time_source", "relative"))


def _resume_context_sha256(**items) -> str:
    """Binary64-exact deterministic fingerprint of data/model inputs used for resume.

    This is deliberately independent of CSV display rounding.  It binds the
    actual in-memory instance inputs so a changed turbine/weather/Xi/parameter
    state cannot inherit a previous completed checkpoint merely because the
    high-level experiment labels are unchanged.
    """
    def fp(value):
        if isinstance(value, pd.DataFrame):
            return (
                "dataframe", tuple(str(c) for c in value.columns),
                tuple(str(t) for t in value.dtypes),
                tuple(tuple(fp(v) for v in row)
                      for row in value.itertuples(index=False, name=None)))
        if isinstance(value, pd.Timestamp):
            return ("timestamp", value.isoformat())
        if isinstance(value, np.datetime64):
            return ("datetime64", str(value.astype("datetime64[ns]")))
        if isinstance(value, Path):
            return ("path", str(value), EU.sha256_file(value) or "missing")
        return BP._state_fp(value)

    payload = {str(k): fp(v) for k, v in sorted(items.items())}
    payload["__source_tree_sha256__"] = M.source_tree_sha256(Path(__file__).resolve().parent)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resume_load(outdir: Path, fname: str, key_cols: list, sig: dict, resume: str,
                 completed_status_col: str | None = None,
                 completed_values=("ok", "completed")):
    """读取结果 CSV 作为检查点，并严格验证完整口径签名。

    ``resume=off`` 或文件不存在时返回空检查点。已有文件无法读取、缺少键列、
    同一签名列内部混入多个值、签名与当前运行不一致或键重复时均立即拒绝，
    防止把不同模型口径或损坏文件静默混排。
    """
    path = Path(outdir) / fname
    if str(resume).lower() == "off" or not path.is_file():
        return [], set()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        raise SystemExit(f"[resume] {fname} 读取失败，拒绝将损坏文件视为空检查点: {e}") from e
    if not len(df):
        return [], set()

    missing_keys = [k for k in key_cols if k not in df.columns]
    if missing_keys:
        raise SystemExit(f"[resume] {fname} 缺少检查点键列 {missing_keys}，拒绝续跑。")

    def _same(a, b):
        if isinstance(b, bool) or isinstance(a, (bool, np.bool_)):
            return str(a).strip().lower() == str(b).strip().lower()
        if isinstance(b, (int, float)):
            try:
                af = float(a); bf = float(b)
                if not (math.isfinite(af) and math.isfinite(bf)):
                    return False
                return af.hex() == bf.hex()
            except (TypeError, ValueError):
                return False
        return str(a) == str(b)

    bad = []
    for k, v in sig.items():
        if k not in df.columns:
            bad.append(f"{k}(旧文件缺列)")
            continue
        series = df[k]
        if v is None:
            # CSV has no native null scalar: an expected Python ``None`` is
            # round-tripped by pandas as an empty field/NaN.  This is a valid,
            # uniform signature value, not a damaged checkpoint.  Mixed null
            # and non-null rows still fail closed below.
            nonnull = series.dropna().tolist()
            if nonnull:
                uniq = list(dict.fromkeys(str(x) for x in nonnull))[:5]
                bad.append(f"{k}: 文件值={uniq!r} 现=None")
            continue
        vals = series.dropna().tolist()
        if len(vals) != len(df):
            bad.append(f"{k}(存在空值)")
            continue
        if not all(_same(x, v) for x in vals):
            uniq = list(dict.fromkeys(str(x) for x in vals))[:5]
            bad.append(f"{k}: 文件值={uniq!r} 现={v!r}")
    if bad:
        raise SystemExit(
            f"[resume] {fname} 与当前运行口径不一致，拒绝混排:\n  " + "\n  ".join(bad)
            + "\n  解决: --resume off 覆盖重跑；或移走旧文件/换输出目录。")

    dup = df.duplicated(subset=key_cols, keep=False)
    if bool(dup.any()):
        sample = df.loc[dup, key_cols].head(5).to_dict("records")
        raise SystemExit(f"[resume] {fname} 存在重复检查点键，拒绝续跑: {sample}")
    done_df = df
    if completed_status_col is not None:
        if completed_status_col not in df.columns:
            raise SystemExit(f"[resume] {fname} 缺少状态列 {completed_status_col}，拒绝续跑。")
        allowed = {str(x).strip().lower() for x in completed_values}
        done_df = df[df[completed_status_col].astype(str).str.strip().str.lower().isin(allowed)]
    done = set()
    required_cert_hashes = ("model_contract_sha256", "parameter_contract_sha256",
                            "instance_contract_sha256", "algorithm_contract_sha256")
    for _, row in done_df.iterrows():
        record = row.to_dict()
        if _global_certificate_flag(record):
            missing = [k for k in required_cert_hashes
                       if k not in record or pd.isna(record[k]) or not str(record[k]).strip()]
            if missing:
                raise SystemExit(
                    f"[resume] {fname} 含正全局证书但缺少 {missing}，拒绝继承旧证书。")
            if str(record.get("physical_numeric_contract", "")) != RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT:
                raise SystemExit(
                    f"[resume] {fname} 正证书 physical_numeric_contract 与当前严格合同不一致。")
            if str(record.get("route_identity_contract", "")) != BP.ROUTE_IDENTITY_CONTRACT:
                raise SystemExit(
                    f"[resume] {fname} 正证书 route_identity_contract 与当前严格合同不一致。")
            if str(record.get("model_semantics_contract", "")) != BP.MODEL_SEMANTICS_CONTRACT:
                raise SystemExit(
                    f"[resume] {fname} 正证书 model_semantics_contract 与当前严格合同不一致。")
        done.add(tuple(str(record[k]) for k in key_cols))
    log.info("[resume] %s: 载入 %d 行，将跳过 %d 个已完成键 %s",
             fname, len(df), len(done), key_cols)
    return df.to_dict("records"), done


def _resolve_pair_radius(args, p_base, xi_amb, turbines):
    """【更新 问题1】解析 --pair-radius:
      'auto'(新默认) = max over 本次涉及的 UAV 档位 of step9.max_flight_radius_m(...).R_max_m
                       —— 半径不再是无依据的 8km 常数, 而是"该批 UAV 里最能飞的那档、在
                       决策 h 上限下的静风外包络最大作业半径"(取 max: 窗/reach 是共享外沿,
                       弱档的不可达由列级 DRCC 兜底, 不会引入乐观解);
      数字字符串      = 显式覆盖(米), 仅供敏感性分析(旧 8km 口径已于 2026-07-06 定案废弃,
                       旧 结果作废存档, 全部实验按新口径重跑)。
    返回 (radius_m, mode_str); 并把推导表打进日志。"""
    s = str(args.pair_radius).strip().lower()
    h_max = float(max(RM.decision_horizons_of(xi_amb)))
    # Δz_insp 按当前风场几何(全场同机型), 与 route_nominal_ET 同口径
    dz = max(float(turbines[0].H_tip) - float(p_base.z_cruise), 0.0) if turbines else 55.0
    if s != "auto":
        r = float(s)
        log.info("pair_radius=%.0fm(显式覆盖, 敏感性分析用; 正式口径=auto)", r)
        return r, f"explicit({r:.0f}m)"
    uks = sorted({u.strip() for u in str(getattr(args, "e1_uavs", "")).split(",") if u.strip()}
                 | {str(args.uav).strip()})
    rows = []
    for uk in uks:
        try:
            q = M.apply_uav_profile(p_base, uk)
        except SystemExit:
            continue
        d = M.max_flight_radius_m(q, h_max, dz_insp_m=dz)
        rows.append((uk, d))
        log.info("pair_radius[auto] %s: R_max=%.0fm (能量限 %.0fm / 时间限 %.0fm; "
                 "E_avail=%.1fWh, h_max=%.0fmin, Δz=%.0fm)", uk, d["R_max_m"],
                 d["R_energy_m"], d["R_time_m"], d["E_avail_Wh"], h_max, dz)
    if not rows:
        # 更新: 8km 常数彻底废弃 —— 无有效档位时回退【基线档 M350 的物理外包络】
        d0 = M.max_flight_radius_m(M.Params(), h_max, dz_insp_m=dz)
        log.warning("pair_radius[auto]: 无有效 UAV 档位, 回退基线档物理外包络 %.0fm", d0["R_max_m"])
        return float(d0["R_max_m"]), f"fallback(base,{d0['R_max_m']:.0f}m)"
    uk_best, d_best = max(rows, key=lambda t: t[1]["R_max_m"])
    r = d_best["R_max_m"]
    log.info("pair_radius[auto] = %.0fm(取 %s 档外包络; 正式口径)", r, uk_best)
    return float(r), f"auto({uk_best},{r:.0f}m,h_max={h_max:.0f})"


def _infarm_segments(track, turbines, pair_radius):
    """返回 [(t_in, t_out, dwell_s), ...](船距任一风机 ≤pair_radius 的连续时段)。"""
    TP = np.array([t.local for t in turbines], float)
    dmin = np.array([np.min(np.linalg.norm(TP - p, axis=1)) for p in track.P])
    inside = dmin <= pair_radius
    segs, s = [], None
    for i, f in enumerate(inside):
        if f and s is None:
            s = i
        if s is not None and (not f or i == len(inside) - 1):
            e = i if f else i - 1
            segs.append((float(track.t[s]), float(track.t[e]), float(track.t[e] - track.t[s])))
            s = None
    return segs


def _classify_state_noleak(track, t_sec, dt_s=30.0, win_s=120.0,
                           dp_kn=0.3, low_kn=1.0, dp_disp_m=30.0, turn_thr=6.0):
    """更新 修复(审计 P1-状态分类器同源): 起飞时隙的 c(τ) 四态分类, 与 step7 估 ξ 矩的
    分类器【同名同阈值同语义】—— 动力定位(速度<0.3kn 且 120s 位移<30m)/低速(<1.0kn)/
    转弯(转率≥6°/min)/直航。旧口径只用 |v|<0.5m/s 二分类 ⇒ 起飞时隙的 c(τ) 与 ξ_amb 的
    分格状态空间不同源, "低速/转弯"档的矩从未被正确选中。
    无泄漏: 全部量取自【后向窗】(仅 t≤τ 的航迹样本) —— 速度 = [τ−dt, τ] 位移/时长;
    位移 = |P(τ)−P(τ−win)|; 转率 = [τ−2dt,τ−dt] 与 [τ−dt,τ] 两段位移方向差 / dt。
    与 step7 的差别仅在估计窗方向(step7 事后标注可用瞬时 SOG; 这里必须后向), 阈值逐位一致。"""
    KN = 0.514444
    t0 = float(t_sec)
    ts = float(track.t[0])
    tA = max(t0 - float(dt_s), ts)
    P0 = np.asarray(track.pos(t0), float)
    PA = np.asarray(track.pos(tA), float)
    el = max(t0 - tA, 1e-6)
    sp_kn = float(np.linalg.norm(P0 - PA)) / el / KN
    PB = np.asarray(track.pos(max(t0 - float(win_s), ts)), float)
    disp = float(np.linalg.norm(P0 - PB))
    if sp_kn < dp_kn and disp < dp_disp_m:
        return "动力定位"
    if sp_kn < low_kn:
        return "低速"
    tC = max(t0 - 2.0 * float(dt_s), ts)
    PC = np.asarray(track.pos(tC), float)
    v1, v2 = PA - PC, P0 - PA
    turn = 0.0
    if np.linalg.norm(v1) > 1.0 and np.linalg.norm(v2) > 1.0:   # 位移过短方向不可信 → 视为直航
        dh = math.degrees(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        dh = abs((dh + 180.0) % 360.0 - 180.0)
        turn = dh / (float(dt_s) / 60.0)
    return "转弯" if turn >= turn_thr else "直航"


def build_launch_options(turbines, lat0lon0, track_csv, xi_amb, wx_df, T_min, dtau_min,
                      pair_radius, hs_quantile=0.5, transit_speed_mps=3.0,
                      track_start_min=None, allow_synth=True, infarm_radius_m=None,
                      predictor="cv_noleak", weather_alignment="representative_quantile",
                      formal=False, bound_track_mmsi=None):
    """起飞时隙构造: 真实 AIS 航迹(缺则合成穿场)× Δτ 网格 × 逐 τ 天气/状态。
    返回 (launch_opts, reach, track_kind, T_eff_min, wx0)。窗按 min(T, 航迹时长) 截断并告知。

    更新(事故复盘·真航迹全零列池): 更新 起 pair_radius=auto=【L 档外包络】(本例 21.2km),
    而选船评分/进场窗起点此前直接复用 pair_radius 作"在场"判据 —— 21km 意义下【泊港/锚地也算在场】,
    整条航迹合并成 1 个场内段 ⇒ 窗起点=第 0 min(船离风机 ~13km+), 列级 DRCC 如实剪光一切
    (单台名义往返已 >30min, 可行 h 被推到 40-60, Σ_h∝h^1.38 + 等待能耗 ⇒ 时间/能量 SOC 双负),
    reach 却=84/90(它只查直线距离≤外包络)。修复: "在场判据"与"可达外包络"语义解耦 ——
    infarm_radius_m(--infarm-radius, 默认 3km=距最近风机≤3km 视为在风场作业, SCN)仅用于
    选船评分/窗起点/警告; reach/列级判据仍用 pair_radius(外包络语义不变)。"""
    lat0, lon0 = lat0lon0
    # v17: pair_radius is not a certified outer bound when the recovery ship moves.
    # Formal mode therefore never lets it influence either the in-farm window
    # diagnostic or the turbine reach set.
    r_config = float(infarm_radius_m) if infarm_radius_m is not None else INFARM_RADIUS_M_DEFAULT
    r_op = float(r_config) if formal else min(float(pair_radius), float(r_config))
    cands = track_csv if isinstance(track_csv, (list, tuple)) else ([track_csv] if track_csv else [])
    if formal and len(cands) != 1:
        raise SystemExit("正式实验必须显式绑定唯一 AIS 航迹；禁止根据未来驻留时间自动选船。")
    if formal and track_start_min is None:
        raise SystemExit("正式实验必须显式提供 --track-start-min；禁止扫描未来 AIS 自动选择任务窗。")
    track, kind, picked = None, None, None
    scored = []
    for c in cands:
        tr = _load_track_ds(c, lat0, lon0)
        if tr is None:
            continue
        # Formal runs already require a unique explicitly bound track and an
        # explicit mission start.  Do not inspect the future trajectory merely
        # to score/select the vessel or locate a window: even a diagnostic that
        # changes control flow would reintroduce look-ahead selection bias.
        dwell = float("nan") if formal else sum(
            d for _, _, d in _infarm_segments(tr, turbines, r_op))
        scored.append((dwell, c, tr))
    if scored:
        if not formal:
            scored.sort(key=lambda x: -x[0])
            for dwell_i, c, _tr in scored:
                log.info("候选航迹 %s: 场内驻留合计 %.0f min(在场判据 r_op=%.0fm)",
                         Path(c).name, dwell_i / 60.0, r_op)
        dwell, picked, track = scored[0]
        if len(scored) > 1:
            log.warning("多条 track_*.csv, 按【场内驻留最长】自动选 %s(用 --track-csv 显式指定可覆盖)。",
                        Path(picked).name)
        if (not formal) and dwell <= 0:
            log.warning("⚠ 选中航迹全程距风机 > %.0fkm —— 船不在本风场?请核对 --farm / MMSI。", r_op / 1000)
        # Formal: explicit --track-start-min, hence no future scan. Mechanism:
        # legacy automatic in-farm window selection remains available.
        segs = [] if formal else _infarm_segments(track, turbines, r_op)
        if track_start_min is not None:
            t0_off = float(track_start_min) * 60.0
        elif segs:
            want = min(T_min * 60.0, max(d for *_, d in segs))
            t0_off = next(a for a, _, d in segs if d >= want - 1e-6)
            log.info("进场窗自动起点: 航迹第 %.1f min(场内段 %d 个, 在场判据 r_op=%.0fm, "
                     "用 --track-start-min 覆盖)", t0_off / 60.0, len(segs), r_op)
        elif len(track) >= 2:
            # 更新: 全程都不进 r_op —— 退而取【距最近风机最近】的时刻开窗(响亮告知, 拒绝静默取 0)
            TP = np.array([t.local for t in turbines], float)
            dmin = np.array([np.min(np.linalg.norm(TP - q, axis=1)) for q in track.P])
            j = int(np.argmin(dmin))
            t0_off = float(track.t[j])
            log.warning("⚠ 航迹全程距最近风机 > r_op=%.0fm —— 无真正作业段。退而在【最近点】"
                        "(第 %.0f min, 距 %.0f m)开窗; 列池可能仍为空, 请核对船只/时段或调 --infarm-radius。",
                        r_op, t0_off / 60.0, float(dmin[j]))
        else:
            t0_off = 0.0
        # Preserve enough pre-window history for the first-slot backward velocity/state classifier.
        # The mission clock remains anchored at exactly t0_off; history is context, not extra mission time.
        history_buffer_s = 300.0
        slice_start_off = max(float(t0_off) - history_buffer_s, float(track.t[0]))
        i0 = max(int(np.searchsorted(track.t, slice_start_off, side="left")) - 1, 0)
        i1 = int(np.searchsorted(track.t, t0_off + T_min * 60.0 + 60.0 * 60.0, side="right"))
        i1 = min(max(i1, i0 + 1), len(track) - 1)
        slice_t0_original = float(track.t[i0])
        mission_origin_sec = float(t0_off) - slice_t0_original
        slice_abs = getattr(track, "absolute_start", None)
        if slice_abs is not None:
            slice_abs = pd.Timestamp(slice_abs) + pd.to_timedelta(slice_t0_original, unit="s")
        track = M.ShipTrack(track.t[i0:i1 + 1] - slice_t0_original, track.P[i0:i1 + 1],
                            absolute_start=slice_abs, time_source=getattr(track, "time_source", "relative"))
        track.mission_origin_sec = mission_origin_sec
        kind = (f"真实AIS({Path(picked).name}, 窗起点@{t0_off/60:.1f}min, "
                f"前置历史{mission_origin_sec/60:.1f}min, {len(track)}点)")
    if track is not None and len(track) >= 2:
        pass
    else:
        track = _synth_transit_track(turbines, speed_mps=transit_speed_mps)
        # 更新 修复: 合成航迹须铺满窗 T —— 否则窗被截成航迹时长(沙箱实测 36min ⇒ 时隙仅 1-2 个)。
        #   按 span/T 自动放慢船速重建; 巧合红利: 放慢后 |v|<0.5m/s ⇒ c(τ)=动力定位(σ_ξ=186m 可解),
        #   而 3m/s 的"直航"在合成 ξ 夹具(σ=727m@h5)下 ε=0.05 物理无解——那是夹具的诚实行为, 非 bug。
        if track.duration_sec() < T_min * 60.0:
            slow = max(transit_speed_mps * track.duration_sec() / (T_min * 60.0), 0.05)
            track = _synth_transit_track(turbines, speed_mps=slow)
            kind = f"合成穿场航迹(无AIS回退, 自动放慢至 {slow:.2f}m/s 铺满窗)"
        else:
            kind = "合成穿场航迹(无AIS文件回退)"
        mission_origin_sec = 0.0
        track.mission_origin_sec = 0.0
        if not allow_synth:
            raise SystemExit(
                "\n[数据哨兵] 未找到真实 AIS 航迹, 已按规则拒绝在合成航迹上出正式结果。\n"
                f"  搜索位置: <脚本目录>/tracks/ 下的 ship_track_*.csv / track_*.csv\n"
                f"  本次候选: {[str(c) for c in cands] or '(空)'}\n"
                "  修复: 把 step6 产出的 track_<mmsi>.csv 放入 tracks/, 或用 --track-csv 显式指定;"
                " 仅调试可加 --allow-synth。")
    horizons = RM.decision_horizons_of(xi_amb)
    h_min_grid = min(horizons)
    mission_origin_sec = float(getattr(track, "mission_origin_sec", 0.0))
    dur = max(track.duration_sec() - mission_origin_sec, 0.0)
    T_eff = min(T_min * 60.0, dur)
    if T_eff < T_min * 60.0 - 1e-6:
        log.warning("航迹时长 %.0fmin < 窗 T=%.0fmin, 窗按航迹截断(船离场即窗止)。", dur/60, T_min)
    # 时隙上限: 只要还能塞下【最短】可行 h(τ+h≤T 由列级过滤兜底) —— 旧版误用 h_max 砍掉了晚起飞。
    t_last = max(T_eff - h_min_grid * 60.0, 0.0)
    mission_slot_offsets = [t for t in np.arange(0.0, t_last + 1e-6, dtau_min * 60.0)]
    slot_times = [mission_origin_sec + t for t in mission_slot_offsets]
    effective_alignment = weather_alignment
    if (str(weather_alignment).lower() == "timestamp"
            and getattr(track, "absolute_start", None) is None):
        if allow_synth:
            log.warning("航迹无绝对 UTC 时间；仅因 --allow-synth 已开启，天气同步降级为代表性分位情景。")
            effective_alignment = "representative_quantile"
        else:
            raise SystemExit("[数据哨兵] 正式 timestamp 天气同步需要 AIS CSV 中可解析的 UTC 时间列；"
                             "请修复时间列，或显式 --weather-alignment representative_quantile "
                             "并将结果解释为情景实验。")
    mission_abs_start = None
    if getattr(track, "absolute_start", None) is not None:
        mission_abs_start = (pd.Timestamp(track.absolute_start)
                             + pd.to_timedelta(mission_origin_sec, unit="s"))
    rows, start, wx_meta = _wx_series(
        wx_df, hs_quantile, absolute_start=mission_abs_start,
        alignment=effective_alignment, past_only_anchor=bool(formal))
    weather_anchor = (mission_abs_start if str(effective_alignment).lower() == "timestamp"
                      else rows.index[start])

    def wx_of_t(t_sec):
        # Mechanism/backward-compatible path. Formal route construction below
        # never passes this realized-target accessor into Step10.
        elapsed_sec = float(t_sec) - mission_origin_sec
        target = pd.Timestamp(weather_anchor) + pd.to_timedelta(elapsed_sec, unit="s")
        pos = int(rows.index.get_indexer([target], method="nearest")[0])
        if pos < 0:
            raise ValueError(f"天气时间轴无法匹配 {target}")
        gap_min = abs(float((rows.index[pos] - target).total_seconds())) / 60.0
        if gap_min > 90.0:
            raise ValueError(f"天气匹配间隔 {gap_min:.1f} min 超过 90 min: {target}")
        out = _wx_row_dict(rows.iloc[pos])
        out["time"] = str(rows.index[pos])
        out["weather_target_time"] = str(target)
        out["weather_match_error_min"] = round(gap_min, 3)
        out["weather_default_used"] = False
        return out

    wx_forecast_of = None
    if formal:
        if mission_abs_start is None:
            raise SystemExit("正式天气信息集需要可解析的 AIS UTC 任务起点。")
        if str(effective_alignment).lower() == "timestamp":
            wx_forecast_of = _launch_asof_weather_forecaster(
                rows, mission_abs_start, mission_origin_sec)
            wx_base_formal = wx_forecast_of(mission_origin_sec, mission_origin_sec)
            wx_meta["weather_information_scope"] = "launch-asof-target-forecast"
        else:
            wx_forecast_of, wx_base_formal, wx_meta_q = _formal_past_quantile_weather_provider(
                rows, mission_abs_start, hs_quantile=hs_quantile)
            wx_meta.update(wx_meta_q)
            wx_meta["weather_information_scope"] = "past-history-scenario-not-confirmatory-forecast"
    else:
        wx_base_formal = wx_of_t(mission_origin_sec)

    def c_of_t(t_sec):
        return _classify_state_noleak(track, t_sec)

    opts = RM.build_launch_grid_from_track(
        track, slot_times, horizons,
        wx_base=wx_base_formal,
        wx_of_t=(None if formal else wx_of_t),
        wx_forecast_of=wx_forecast_of,
        c_state_of_t=c_of_t, predictor=predictor,
        mission_origin_sec=mission_origin_sec)
    # The symmetric launch-distance radius is not a theorem-level outer bound
    # with a moving recovery ship. Formal exactness therefore keeps every
    # turbine and lets the column-level physical/DRCC oracle decide feasibility.
    reach = (list(turbines) if formal else
             [t for t in turbines
              if min(float(np.linalg.norm(t.local - o.ship.P_launch)) for o in opts) <= pair_radius])
    log.info("时隙: %d 个(Δτ=%.0fmin, 窗=%.0fmin, %s); reach=%d/%d 台",
             len(opts), dtau_min, T_eff/60, kind, len(reach), len(turbines))
    # ---- 更新 诊断: 窗内船-最近风机距离(standoff)与 c(τ) 分布 —— reach 只是外包络,
    #      standoff 才决定列级 DRCC 生死(真航迹全零列池事故的可观测前兆, 必打) ----
    if opts:
        TP = np.array([t.local for t in turbines], float)
        d_stand = np.array([float(np.min(np.linalg.norm(TP - o.ship.P_launch, axis=1)))
                            for o in opts])
        from collections import Counter
        cnt = Counter(o.ship.c_state for o in opts)
        log.info("窗内 standoff(船→最近风机): 中位 %.0fm / 最小 %.0fm / 最大 %.0fm; "
                 "c(τ) 四态: %s(在场判据 r_op=%.0fm; 预测器=%s)",
                 float(np.median(d_stand)), float(d_stand.min()), float(d_stand.max()),
                 dict(cnt), r_op, predictor)
        if float(np.median(d_stand)) > r_op:
            log.warning("⚠ 窗内 standoff 中位 %.0fm > r_op=%.0fm —— 窗口大概率对到了船【泊港/"
                        "锚地/转场】时段, 列级 DRCC 会剪光所有列(列池=0, 但 reach 照常通过)。"
                        "修复: --track-start-min <作业时段起点分钟> 显式指定, 或调 --infarm-radius, "
                        "或 --track-csv 换船; 修正后用 --resume off 重跑(窗口径已变, 旧检查点作废)。",
                        float(np.median(d_stand)), r_op)
    wx0 = (dict(wx_base_formal) if formal else wx_of_t(mission_origin_sec))
    wx0.update(wx_meta)
    wx0["static_master_information_scope"] = (
        "launch-asof-columns-in-static-master" if formal else "mechanism-realized-target-series")
    # A static master that simultaneously chooses future launch epochs uses the
    # launch-time AIS state in each column. This is honest launch-asof modeling,
    # but it is not a proof of mission-start multistage nonanticipativity.
    wx0["operational_nonanticipativity_certified"] = False
    wx0["pair_radius_pruning_used"] = bool(not formal)
    wx0["pair_radius_pruning_formal_outer_bound_certified"] = False
    wx0["pair_radius_scope"] = ("diagnostic-only-formal-moving-ship" if formal
                                else "mechanism-static-symmetric-filter")
    wx0["mission_origin_sec"] = round(float(mission_origin_sec), 6)
    wx0["mission_absolute_start"] = (str(mission_abs_start) if mission_abs_start is not None else None)
    wx0["track_time_source"] = getattr(track, "time_source", "relative")
    wx0["track_absolute_start"] = (str(getattr(track, "absolute_start", None))
                                    if getattr(track, "absolute_start", None) is not None else None)
    wx0["selected_track_csv"] = (str(picked) if picked is not None else None)
    filename_mmsi = (Path(picked).stem.replace("track_", "")
                     if picked is not None else None)
    if formal and bound_track_mmsi not in (None, "", "ALL"):
        wx0["selected_track_mmsi"] = str(bound_track_mmsi)
    else:
        # Preserve the legacy/mechanism filename-derived label exactly; only
        # formal runs may replace it with the already-validated explicit MMSI.
        wx0["selected_track_mmsi"] = filename_mmsi
    for opt in opts:
        opt.ship._track_mmsi = wx0["selected_track_mmsi"]
        opt.ship._track_csv = wx0["selected_track_csv"]
    return opts, reach, kind, T_eff / 60.0, wx0


# 更新: 口径升版 —— 无泄漏回收预测(cv_noleak)+四态无泄漏状态分类+回放含对接/天气/门复判
# + τ=离舰时刻的甲板/占机语义 + pareto-h 列池 + Σ 插值修复；不同结果合同不可混排。
# resume 签名会拒绝旧检查点(须整批重跑)。

# 更新: "在场作业"判据半径默认值(米, SCN) —— 距最近风机 ≤3km 视为船在风场作业。
#   仅用于【选船评分 / 进场窗自动起点 / 不在场警告】三处; reach 可达集与列级 DRCC 仍用
#   pair_radius(UAV 物理外包络, 更新 语义不变)。两者语义不同, 严禁再互相复用
#   (更新 事故: 外包络 21.2km 被当在场判据 ⇒ 窗对到泊港时段 ⇒ E1 列池全零)。
INFARM_RADIUS_M_DEFAULT = 3000.0


def _replay_columns(chosen, p, xi_amb, eps, n_per=400, seed=7, wamb=None,
                    validation_mode="synthetic_stress", real_samples_csv=None,
                    weather_sample_mode="synthetic",
                    holdout_disjointness_verified=False,
                    holdout_independence_verified=None):
    """逐列 out-of-sample 回放: 每列按【它自己的 (τ 船, τ 天气, h, c(τ))】回放。
    更新: 返回诊断 dict —— 静默剔除是 更新 前 E2 'holds' 失真的根因, 现在必须显式记账:
      per            逐列违反率(None=该列回放缺样本)
      emp            架次加权经验违反率(仅计有样本列)
      safe_tids      回放可靠(viol ≤ mission_eps_budget; weather off 时=2ε)列覆盖的风机集合
      n_replayed / n_missing   有样本列数 / 缺样本列数(>0 时 emp 与 holds 只覆盖部分计划!)
      max_col_viol   逐列违反率最大值(计划级 holds 的短板判据)
    更新 修复(审计 P0-回放一致性), 三处口径变化(经 step15 新协议):
      ① ξ 样本改按【本实例 xi_amb 的逐 (h,c) 矩】t3 矩匹配生成(旧口径硬编码另一组
         base5 方差, 回放分布与规划歧义集不同源 ⇒ 违反率既不可比也不可控);
      ② 回放判据与规划完全对齐: E+E_dock>B_use、T+t_dock>60h、着舰门在实现浪/风下复判、
         空速包络在实现风下复判(wamb 给出该 h 的风/浪二阶矩, t3 重尾扰动);
      ③ 逐列 realized 审计: cv_noleak 预测器下每列有一次【真实】预测误差实现
         ξ_real = P_track(τ+h) − P̂(τ+h), 用它做单次真回放(n_realized/n_realized_viol),
         这是合成矩匹配样本之外的 out-of-sample 真值检验。"""
    import step15_replay as RP
    if holdout_independence_verified is not None:
        holdout_disjointness_verified = bool(
            holdout_disjointness_verified or holdout_independence_verified)
    if not chosen:
        return dict(per=[], emp=None, safe_tids=set(), n_replayed=0, n_missing=0,
                    max_col_viol=None, n_test_total=0, n_viol_total=0,
                    ci95_low=None, ci95_high=None, upper95=None,
                    n_realized=0, n_realized_viol=0,
                    realized_ci95_low=None, realized_ci95_high=None,
                    realized_upper95=None,
                    validation_type=("real-validation" if validation_mode in ("real_validation", "real_holdout")
                                     else ("real-joint-final-test" if validation_mode == "real_joint_final_test"
                                           else ("real-xi-final-test+synthetic-weather" if validation_mode == "real_xi_final_test"
                                                 else "synthetic-moment-matched-t3-stress"))),
                    allocation_budget=float(RM.mission_eps_budget(p, wamb is not None)),
                    mission_requirement_budget=0.05,
                    allocation_budget_holds=None, mission_requirement_holds=None,
                    all_routes_allocation_holds=False, all_routes_mission_holds=False,
                    validation_gate_contract=(
                        "selection-gate-internal-allocation-budget-v14-exact-qmax-once-only;per-sortie-bonferroni-retained;event-fingerprints-audit-only;"
                        "mission-0.05-reported-separately"),
                    route_validation_records=[],
                    disjoint_xi_holdout=bool(validation_mode in ("real_xi_final_test", "real_joint_final_test")),
                    disjoint_weather_holdout=bool(validation_mode == "real_joint_final_test"),
                    disjoint_real_holdout=bool(validation_mode == "real_joint_final_test"),
                    independent_xi_holdout=False, independent_weather_holdout=False,
                    independent_real_holdout=False)
    states = sorted({c["ship"].c_state for c in chosen})
    base_h = sorted(int(h) for h in xi_amb.horizons)
    mode = str(validation_mode).strip().lower()
    if mode == "real_holdout":
        # 旧名仅作为 validation 别名保留；它可以参与调参，因此绝不再标记为 confirmatory final test。
        mode = "real_validation"
    disjoint_xi_holdout = False
    disjoint_weather_holdout = False
    disjoint_real_holdout = False
    if mode in ("real_validation", "real_xi_final_test", "real_joint_final_test"):
        if real_samples_csv is None or not Path(real_samples_csv).is_file():
            raise FileNotFoundError(f"validation_mode={mode} 需要真实样本 CSV")
        _replay_mmsi = str(getattr(xi_amb, "selected_mmsi", "ALL"))
        if mode in ("real_validation", "real_xi_final_test", "real_joint_final_test") and _replay_mmsi.upper() == "ALL":
            raise ValueError("正式真实回放禁止 mmsi=ALL；必须绑定当前轨迹的具体 MMSI。")
        _formal_replay = bool(getattr(xi_amb, "formal_validated", False))
        _expected_split = ("validation" if mode == "real_validation" else "test")
        test_df = RP.load_samples(
            Path(real_samples_csv), mmsi=_replay_mmsi,
            formal=_formal_replay,
            expected_split=(_expected_split if _formal_replay else None))
        needed_cells = [(int(c["h"]), str(c["ship"].c_state)) for c in chosen]
        missing_cells = RP.required_cells_present(test_df, needed_cells)
        if missing_cells:
            # 真实留出禁止用插值矩或合成样本补格；缺格会在逐列回放中显式成为 n_missing。
            log.warning("真实回放缺少 %d 个 (h,c) 格: %s", len(missing_cells), missing_cells[:12])
        if mode == "real_validation":
            validation_type = "real-validation-not-final"
            independent_xi_holdout = independent_weather_holdout = independent_real_holdout = False
        elif mode == "real_xi_final_test":
            validation_type = "real-xi-final-test+synthetic-weather"
            disjoint_xi_holdout = bool(holdout_disjointness_verified)
            independent_xi_holdout = independent_weather_holdout = independent_real_holdout = False
        else:
            validation_type = "real-joint-final-test"
            disjoint_xi_holdout = bool(holdout_disjointness_verified)
            disjoint_weather_holdout = bool(holdout_disjointness_verified)
            disjoint_real_holdout = bool(holdout_disjointness_verified)
            independent_xi_holdout = independent_weather_holdout = independent_real_holdout = False
            weather_sample_mode = "real"
    elif mode == "synthetic_stress":
        test_df = RP._dist_samples(base_h, states, dist="t3", n_per=n_per, seed=seed, xi_amb=xi_amb)
        # 决策细格 h 不在统计粗格时，仅合成压力测试可以按内插矩补格；真实留出绝不补造样本。
        miss_h = sorted({int(c["h"]) for c in chosen} - set(base_h))
        for hh in miss_h:
            test_df = pd.concat([test_df, RP._dist_samples([hh], states, dist="t3", n_per=n_per,
                                                           seed=seed + 1000 + hh, xi_amb=xi_amb)],
                                ignore_index=True)
        validation_type = "synthetic-moment-matched-t3-stress"
        independent_xi_holdout = independent_weather_holdout = independent_real_holdout = False
    else:
        raise ValueError(f"未知 validation_mode={validation_mode}")
    per, safe_tids, n_tot, v_tot = [], set(), 0, 0
    route_reports = []
    route_validation_records = []
    n_rep = n_miss = 0
    n_real = n_real_v = 0
    vmax = None
    simultaneous_confidence = 1.0 - 0.05 / max(len(chosen), 1)
    for ci, c in enumerate(chosen):
        # Formal real-weather replay must receive the complete WeatherAmbiguity object so
        # step15 can validate source/train/predictor provenance before resolving this
        # sortie's exact h cell.  Keep the resolved cell only for local diagnostics/budget
        # bookkeeping; passing the cell itself would discard the formal provenance envelope.
        wu_cell = (wamb.get(int(c["h"])) if wamb is not None else None)
        rep = RP.replay_routes([(c["route"], int(c["h"]))], c["ship"], p, c["wx"], test_df,
                               weather_unc=wamb, weather_dist="t3",
                               weather_seed=seed + 20000 + 17 * ci,
                               include_dock=True, recheck_gate=True,
                               weather_sample_mode=weather_sample_mode,
                               recovery_state_sample_mode=("real" if mode in (
                                   "real_validation", "real_xi_final_test", "real_joint_final_test")
                                   else "planned"),
                               holdout_disjointness_verified=bool(holdout_disjointness_verified),
                               confirmatory=bool(mode == "real_joint_final_test"))
        route_reports.append(rep)
        _nv = rep.get("n_viol_any")
        _pr0 = ((rep.get("per_route") or [{}])[0]
                if isinstance(rep.get("per_route"), list) else {})
        route_validation_records.append(dict(
            tau=float(c.get("tau", 0.0)),
            h=float(c.get("h", 0.0)),
            ordered_tids=list(map(str, c.get("ordered_tids", c.get("tids", ())))),
            n_test_total=int(rep.get("n_test_total", 0) or 0),
            n_viol_any=(None if _nv is None else int(_nv)),
            viol_rate_any=rep.get("viol_rate_any"),
            validation_complete=bool(rep.get("validation_complete", False)),
            holdout_cell_sha256=_pr0.get("holdout_cell_sha256"),
            observed_failure_mask_sha256=_pr0.get("observed_failure_mask_sha256"),
            validation_event_fingerprint=_pr0.get("validation_event_fingerprint"),
            validation_event_fingerprint_scope=_pr0.get(
                "validation_event_fingerprint_scope",
                "audit-only-no-bonferroni-relaxation"),
        ))
        v = rep["viol_rate_any"]; nt = rep["n_test_total"]
        per.append(v)
        if v is None or nt == 0 or not rep.get("validation_complete", False):
            n_miss += 1
            continue
        n_rep += 1
        n_tot += nt
        v_tot += int(rep.get("n_viol_any", round(float(v) * nt)))
        vmax = v if vmax is None else max(vmax, v)
        # 更新(审计修复#8-联合预算): 逐列"回放可靠"阈 = RM.mission_eps_budget ——
        # weather on 时 = ε_E+ε_T+ε_cap+ε_gate+ε_air(更新 补浪门 ε_cap; 旧 2ε 让 gate/speed
        # 两类违反没有任何名义配额, holds 判据与规划保证不构成同一命题); weather off = 2ε 不变。
        _nviol = int(rep.get("n_viol_any", round(float(v) * nt)))
        _failures = [1] * _nviol + [0] * (int(nt) - _nviol)
        _upper = EU.martingale_conditional_risk_upper_bound(
            _failures, confidence=simultaneous_confidence)
        if (_upper is not None and
                _upper <= RM.mission_eps_budget(p, wu_cell is not None or weather_sample_mode == "real")):
            safe_tids.update(c["tids"])
        # ---- realized 审计(名义天气下的单次真实 ξ; 无航迹/合成场景自动为 0 样本) ----
        xi_r = None
        try:
            xi_r = c["ship"].xi_realized(float(c["h"]))
        except Exception:
            xi_r = None
        if xi_r is not None and np.all(np.isfinite(xi_r)):
            n_real += 1
            # 更新(审计修复#8): realized 审计与回放协议 同口径 —— 门/对接天气用
            # 实现回收点最近风机本地天气(recovery_gate_wx), t_dock 先算并传入等待窗。
            P_rr = c["ship"].predicted_at(float(c["h"])) + np.asarray(xi_r, float)
            gwx = RM.recovery_gate_wx(c["route"], c["wx"], P_rr)
            gHs = float(gwx.get("Hs", 0.5) if gwx.get("Hs") is not None else 0.5)
            gTp = float(gwx.get("Tp", 2.1)); gwv = float(gwx.get("wave_dir", 200.0))
            ghd = float(c["wx"].get("ship_heading", 0.0))
            w10n = gwx.get("wind10", 6.7)
            w10n = 6.7 if (w10n is None or (isinstance(w10n, float) and math.isnan(w10n))) else float(w10n)
            mo = M.deck_motion(gHs, gTp, gwv - ghd, p)
            t_d, E_d = M.dock_reserve(p, mo, w10n)
            det = RM.route_energy_time(c["route"], int(c["h"]), np.asarray(xi_r, float),
                                       p, c["wx"], detail=True, t_dock_s=t_d)
            Lg, _ = M.landing_gate(gHs, gTp, gwv, ghd, w10n, p)
            realized_time = RM.realized_fixed_touchdown_time(
                60.0 * float(c["h"]), float(det["T"]) + float(t_d))
            if (det["E"] + E_d > p.B_use or realized_time["time_violation"]
                    or not det["speed_feasible"] or Lg == 0):
                n_real_v += 1
    emp = round(v_tot / n_tot, 4) if n_tot else None
    # Pooled interval is descriptive only because different sorties may reuse the same holdout rows.
    ci_lo, ci_hi = EU.binomial_interval(int(v_tot), int(n_tot), confidence=0.95)
    pooled_up = EU.binomial_upper_bound(int(v_tot), int(n_tot), confidence=0.95)
    allocation_budget = float(RM.mission_eps_budget(
        p, wamb is not None or weather_sample_mode == "real"))
    mission_requirement_budget = 0.05
    route_uppers = []
    route_allocation_holds = []
    route_mission_holds = []
    for rec, rep in zip(route_validation_records, route_reports):
        nt = int(rep.get("n_test_total", 0) or 0)
        nv = rep.get("n_viol_any")
        u = None
        if rep.get("validation_complete", False) and nt > 0 and nv is not None:
            failures = [1] * int(nv) + [0] * (nt - int(nv))
            u = EU.martingale_conditional_risk_upper_bound(
                failures, confidence=simultaneous_confidence)
            if u is not None:
                u = float(u)
                route_uppers.append(u)
        alloc_hold = (None if u is None else bool(u <= allocation_budget))
        mission_hold = (None if u is None else bool(u <= mission_requirement_budget))
        rec.update(
            upper95_simultaneous=_formal_statistic_value(u),
            allocated_epsilon=_formal_statistic_value(allocation_budget),
            mission_epsilon=mission_requirement_budget,
            allocation_budget_holds=alloc_hold,
            mission_requirement_holds=mission_hold)
        if alloc_hold is not None:
            route_allocation_holds.append(bool(alloc_hold))
        if mission_hold is not None:
            route_mission_holds.append(bool(mission_hold))
    event_fps = [str(r.get("validation_event_fingerprint"))
                 for r in route_validation_records
                 if r.get("validation_event_fingerprint")]
    unique_event_fps = sorted(set(event_fps))
    event_fp_counts = {fp: event_fps.count(fp) for fp in unique_event_fps}
    duplicate_event_groups = {
        fp: int(n) for fp, n in event_fp_counts.items() if int(n) > 1
    }
    validation_complete = bool(n_miss == 0 and len(route_uppers) == len(chosen))
    simultaneous_upper = max(route_uppers) if validation_complete and route_uppers else None
    allocation_budget_holds = (None if simultaneous_upper is None else
                               bool(simultaneous_upper <= allocation_budget))
    mission_requirement_holds = (None if simultaneous_upper is None else
                                 bool(simultaneous_upper <= mission_requirement_budget))
    all_routes_allocation_holds = bool(
        validation_complete and len(route_allocation_holds) == len(chosen)
        and all(route_allocation_holds))
    all_routes_mission_holds = bool(
        validation_complete and len(route_mission_holds) == len(chosen)
        and all(route_mission_holds))
    formal_eligible = bool(validation_complete and disjoint_real_holdout
                           and mode == "real_joint_final_test" and route_reports
                           and all(r.get("formal_reliability_claim_eligible", False)
                                   for r in route_reports))
    r_lo, r_hi = EU.binomial_interval(int(n_real_v), int(n_real), confidence=0.95)
    r_up = (EU.martingale_conditional_risk_upper_bound(
        [1] * int(n_real_v) + [0] * (int(n_real) - int(n_real_v)), confidence=0.95)
        if int(n_real) > 0 else None)
    return dict(per=per, emp=emp, safe_tids=safe_tids,
                n_replayed=n_rep, n_missing=n_miss,
                n_test_total=int(n_tot), n_viol_total=int(v_tot),
                ci95_low=(round(float(ci_lo), 6) if ci_lo is not None else None),
                ci95_high=(round(float(ci_hi), 6) if ci_hi is not None else None),
                pooled_naive_upper95=(round(float(pooled_up), 6) if pooled_up is not None else None),
                upper95=_formal_statistic_value(simultaneous_upper),
                simultaneous_confidence_per_sortie=_formal_statistic_value(simultaneous_confidence),
                ci_method="bonferroni-simultaneous-max-per-sortie-hoeffding-azuma-conditional-risk-upper95",
                validation_complete=validation_complete,
                formal_reliability_claim_eligible=formal_eligible,
                evidence_scope=("confirmatory-purged-disjoint-real-joint-holdout"
                                if formal_eligible else
                                ("validation-selection-only-no-formal-inference"
                                 if mode == "real_validation" else "mechanism-or-partial-evidence")),
                max_col_viol=(round(float(vmax), 4) if vmax is not None else None),
                n_realized=n_real, n_realized_viol=n_real_v,
                realized_ci95_low=(round(float(r_lo), 6) if r_lo is not None else None),
                realized_ci95_high=(round(float(r_hi), 6) if r_hi is not None else None),
                realized_upper95=(round(float(r_up), 6) if r_up is not None else None),
                validation_type=validation_type,
                validation_plan_fingerprint=_frozen_plan_fingerprint(chosen),
                route_validation_records=route_validation_records,
                validation_event_fingerprint_count=int(len(event_fps)),
                validation_unique_event_fingerprint_count=int(len(unique_event_fps)),
                validation_duplicate_event_groups=int(len(duplicate_event_groups)),
                validation_event_group_sizes_json=json.dumps(
                    sorted(event_fp_counts.values()), ensure_ascii=False),
                validation_event_grouping_used_for_gate=False,
                validation_event_grouping_note=(
                    "audit-only: identical empirical masks do not prove identical "
                    "population hypotheses; formal gate remains per-sortie Bonferroni"),
                allocation_budget=_formal_statistic_value(allocation_budget),
                mission_requirement_budget=mission_requirement_budget,
                allocation_budget_holds=allocation_budget_holds,
                mission_requirement_holds=mission_requirement_holds,
                all_routes_allocation_holds=all_routes_allocation_holds,
                all_routes_mission_holds=all_routes_mission_holds,
                validation_gate_contract=(
                    "selection-gate-internal-allocation-budget-v14-exact-qmax-once-only;"
                    "per-sortie-bonferroni-retained;"
                    "event-fingerprints-audit-only;"
                    "mission-0.05-reported-separately"),
                disjoint_xi_holdout=disjoint_xi_holdout,
                disjoint_weather_holdout=disjoint_weather_holdout,
                disjoint_real_holdout=disjoint_real_holdout,
                independent_xi_holdout=False,
                independent_weather_holdout=False,
                independent_real_holdout=False)


def _provenance(args, kind, T_eff, reach, p, t_swap_min, t_launch_min, max_stops_val=None, wx0=None):
    """更新: 每行结果携带完整可追溯口径(结果 CSV 与产生它的代码/参数解耦的教训 —— 上一批
    E1/E2 是合成航迹产物却无从内部识别; 现在 track 列直接写 kind, 合成即显形)。
    更新: max_stops 记【该行实际使用的生成上界】(逐 UAV 解析后的 stops_cap), 非 CLI 原值。"""
    return dict(track=str(kind), T_eff_min=round(float(T_eff), 1), n_reach=len(reach),
                dtau_min=args.dtau_min, deck_delta_min=args.deck_delta_min,
                deck_mode=args.deck_mode, t_swap_min=t_swap_min,
                t_launch_min=(t_launch_min if t_launch_min is not None else args.deck_delta_min),
                max_stops=(int(max_stops_val) if max_stops_val is not None else args.max_stops),
                max_stops_requested=int(args.max_stops),
                stops_cap_spec=str(getattr(args, "stops_cap", args.max_stops)),
                max_stops_effective=(int(max_stops_val) if max_stops_val is not None else int(args.max_stops)),
                replay_n=args.replay_n,
                allow_synth=bool(args.allow_synth),
                track_start_min=(args.track_start_min if args.track_start_min is not None else -1),
                eps=p.eps_E, saa_source=RM.SAA_SOURCE, result_contract=RESULT_CONTRACT,
                formal_experiment_scheduler_contract=FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
                physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
                route_identity_contract=BP.ROUTE_IDENTITY_CONTRACT,
                model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
                resume_input_sha256=getattr(args, "_resume_input_sha256", "missing"),
                source_tree_sha256=getattr(args, "_source_tree_sha256", M.source_tree_sha256(Path(__file__).resolve().parent)),
                predictor=getattr(args, "recovery_predictor", "cv_noleak"),
                weather_information_scope=(wx0 or {}).get("weather_information_scope", "unknown"),
                static_master_information_scope=(wx0 or {}).get("static_master_information_scope", "unknown"),
                operational_nonanticipativity_certified=bool(
                    (wx0 or {}).get("operational_nonanticipativity_certified", False)),
                pair_radius_pruning_used=bool((wx0 or {}).get("pair_radius_pruning_used", True)),
                pair_radius_pruning_formal_outer_bound_certified=bool(
                    (wx0 or {}).get("pair_radius_pruning_formal_outer_bound_certified", False)),
                pool_h=getattr(args, "pool_h", "pareto"),
                # 更新: 三个新溯源字段(问题4局限#6) —— 结果 CSV 自证数据来源与口径
                xi_source=getattr(args, "_xi_source", "unknown"),
                xi_mmsi=getattr(args, "_resolved_xi_mmsi", "ALL"),
                xi_predictor=getattr(args, "_resolved_xi_predictor", "unknown"),
                xi_predictor_contract=getattr(args, "_resolved_xi_predictor_contract", "unknown"),
                weather_uncertainty_source=getattr(args, "_weather_uncertainty_source", "off"),
                weather_formal_eligible=bool(getattr(args, "_weather_formal_eligible", False)),
                weather_risk_contract="vector-route-scalar-landing",
                route_airspeed_contract="per-leg-along-cross-projection",
                time_contract=RM.time_contract_for(p),
                time_contract_id=RM.time_contract_for(p),
                wait_is_recourse=RM.WAIT_IS_RECOURSE,
                speed_is_recourse=bool(getattr(p, "speed_adjustable", False)),
                return_speed_recourse_contract=(RM.SPEED_RECOURSE_CONTRACT if getattr(p, "speed_adjustable", False) else None),
                energy_recourse_contract=(RM.ENERGY_SPEED_RECOURSE_CONTRACT if getattr(p, "speed_adjustable", False) else None),
                dock_risk_contract=RM.DOCK_RISK_CONTRACT,
                pair_radius_m=round(float(getattr(args, "_pair_radius_m", -1.0)), 1),
                pair_radius_mode=getattr(args, "_pair_radius_mode", "unknown"),
                soc_correction=getattr(p, "soc_correction", "none"),
                soc_risk_allocation=getattr(p, "soc_risk_allocation", "fixed"),
                time_recourse_mode=getattr(p, "time_recourse_mode", "wait_only"),
                geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
                weather_alignment_mode=(wx0 or {}).get("weather_alignment_mode", "unknown"),
                weather_start_time=str((wx0 or {}).get("weather_start_time", "unknown")),
                weather_target_time=str((wx0 or {}).get("weather_target_time", "none")),
                weather_match_error_min=(float((wx0 or {}).get("weather_match_error_min"))
                                         if (wx0 or {}).get("weather_match_error_min") is not None else -1.0),
                weather_rows_total=int((wx0 or {}).get("weather_rows_total", 0)),
                weather_rows_complete=int((wx0 or {}).get("weather_rows_complete", 0)),
                weather_rows_incomplete=int((wx0 or {}).get("weather_rows_incomplete", 0)),
                mission_absolute_start=str((wx0 or {}).get("mission_absolute_start", "none")),
                mission_origin_sec=float((wx0 or {}).get("mission_origin_sec", 0.0)),
                track_absolute_start=str((wx0 or {}).get("track_absolute_start", "none")),
                track_time_source=(wx0 or {}).get("track_time_source", "unknown"),
                validation_mode=getattr(args, "validation_mode", "synthetic_stress"),
                validation_samples_hash=(EU.sha256_file(getattr(args, "validation_samples", None)) or "none"),
                xi_train_samples_hash=(EU.sha256_file(getattr(args, "xi_train_samples", None)) or "none"),
                final_test_samples_hash=(EU.sha256_file(getattr(args, "final_test_samples", None)) or "none"),
                study_mode=getattr(args, "study_mode", "mechanism"),
                formal_protocol=bool(getattr(args, "study_mode", "mechanism") == "formal"),
                recovery_target_model=str(getattr(p, "recovery_target_model", "discrete_horizon_ship_prediction")),
                terminal_sensor_error_mode=str(getattr(p, "terminal_sensor_error_mode", "out_of_scope")),
                quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                swap_station_capacity=int(getattr(args, "swap_stations", 1)),
                battery_reuse_mode=getattr(args, "battery_reuse_mode", "exact_soc"),
                solver_mode=getattr(args, "solver_mode", "exact-branch-price-cut"),
                pricing_mode=getattr(args, "pricing_mode", "exact-implicit-dfs"),
                archive_diagnostic_time_limit_s=float(getattr(
                    args, "archive_diagnostic_time_limit_s", 30.0)),
                archive_shadow_diagnostic_time_limit_s=float(getattr(
                    args, "archive_shadow_diagnostic_time_limit_s", 30.0)),
                archive_clique_diagnostic_time_limit_s=float(getattr(
                    args, "archive_clique_diagnostic_time_limit_s", 30.0)),
                formal_warmstart_seconds=float(getattr(args, "formal_warmstart_seconds", 60.0)),
                formal_route_universe=str(getattr(args, "formal_route_universe", "auto")),
                formal_route_universe_max_turbines=int(getattr(
                    args, "formal_route_universe_max_turbines", 8)),
                formal_route_universe_max_stops=int(getattr(
                    args, "formal_route_universe_max_stops", 4)),
                formal_route_universe_time_limit_s=float(getattr(
                    args, "formal_route_universe_time_limit_s", 7200.0)),
                **_formal_instance_manifest_extra(args))



def _e1_complete_universe_enabled(args, n_turbines, max_stops):
    mode = str(getattr(args, "formal_route_universe", "auto")).strip().lower()
    if mode == "off":
        return False
    if mode == "force":
        return True
    if mode != "auto":
        raise ValueError("formal_route_universe must be auto/off/force")
    return bool(
        str(getattr(args, "study_mode", "formal")) == "formal"
        and int(n_turbines) <= int(getattr(args, "formal_route_universe_max_turbines", 8))
        and int(max_stops) <= int(getattr(args, "formal_route_universe_max_stops", 4)))


def _build_or_get_e1_complete_universe(
        reach, opts, p_u, xi_amb, wamb, args, T_eff,
        cap_u, t_launch, uk, outdir):
    """Build once per UAV and reuse across every E1 K/B/target/knee solve."""
    if xi_amb is None or not reach or not opts:
        return None
    if not _e1_complete_universe_enabled(args, len(reach), cap_u):
        return None
    cache = getattr(args, "_e1_route_universes", None)
    if cache is None:
        cache = {}
        setattr(args, "_e1_route_universes", cache)
    if uk in cache:
        return cache[uk]

    limit = getattr(args, "formal_route_universe_time_limit_s", 7200.0)
    limit = None if limit is None or float(limit) <= 0 else float(limit)
    log.info(
        "E1[%s] 构建正式完整物理路线宇宙: n=%d stops<=%d launch=%d; "
        "该步骤只做一次，随后所有 K/B/target 共用。",
        uk, len(reach), int(cap_u), len(opts))
    universe = BP.build_certified_route_universe(
        reach, opts, p_u, xi_amb, T_eff,
        max_stops=int(cap_u), weather_unc=wamb,
        kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, t_launch_min=float(t_launch),
        landing_clear_min=float(args.landing_clear_min),
        deck_mode=str(args.deck_mode), deck_delta_min=float(args.deck_delta_min),
        time_limit_s=limit)
    stats = dict(universe.stats)
    stats.update(uav=str(uk), uav_label=str(p_u.uav_label))
    _save(pd.DataFrame([stats]), Path(outdir),
          f"E1_complete_route_universe_manifest_{uk}.csv")
    if universe.columns:
        audit_rows = []
        for j, c in enumerate(universe.columns):
            tids = list(map(str, c.get("ordered_tids", c.get("tids", ()))))
            sig = BP._exact_route_signature(c)
            res = c.get("resource_intervals") or {}
            diag = c.get("physical_diagnostics") or {}
            margins = diag.get("margins") if isinstance(diag, dict) else {}
            margins = margins if isinstance(margins, dict) else {}
            audit_rows.append(dict(
                route_index=int(j), uav=str(uk), uav_label=str(p_u.uav_label),
                tau_min=float(c.get("tau", 0.0)), h_min=float(c.get("h", 0.0)),
                stops=len(tids), ordered_tids=json.dumps(tids, ensure_ascii=False),
                E_plan_Wh=float(c.get("E_plan_Wh", c.get("E0", 0.0))),
                E_soc_required_Wh=float(c.get("E_soc_required_Wh", c.get("E_plan_Wh", c.get("E0", 0.0)))),
                physical_feasible=bool(diag.get("feasible", True)),
                physical_reason=diag.get("reason"),
                energy_margin_Wh=diag.get("margin_E", margins.get("energy_Wh")),
                time_drcc_margin_s=diag.get("time_drcc_margin_s", margins.get("time_s")),
                nominal_time_margin_s=diag.get("nominal_time_margin_s"),
                time_drcc_tightening_s=diag.get("time_drcc_tightening_s"),
                max_required_airspeed_ms=diag.get("max_required_airspeed_ms"),
                airspeed_margin_ms=diag.get("airspeed_margin_ms", margins.get("route_airspeed_ms")),
                recovery_state=diag.get("recovery_state"),
                recovery_state_source=diag.get("recovery_state_source"),
                mission_eps_budget=diag.get("mission_eps_budget"),
                risk_allocation_json=json.dumps(
                    diag.get("risk_allocation"), ensure_ascii=False,
                    sort_keys=True, default=str),
                route_signature_sha256=hashlib.sha256(
                    repr(sig).encode("utf-8")).hexdigest(),
                resource_intervals_json=json.dumps(
                    res, ensure_ascii=False, sort_keys=True, default=str),
                physical_diagnostics_json=json.dumps(
                    diag, ensure_ascii=False, sort_keys=True, default=str),
                universe_context_sha256=str(universe.context_sha256),
                universe_columns_sha256=str(universe.columns_sha256),
                universe_builder_contract=str(universe.builder_contract)))
        audit_df = pd.DataFrame(audit_rows)
        _save(audit_df, Path(outdir), f"E1_complete_route_universe_{uk}.csv")
        def _finite_summary(col):
            if col not in audit_df.columns:
                return (None, None, None)
            vals = pd.to_numeric(audit_df[col], errors="coerce")
            vals = vals[np.isfinite(vals.to_numpy(float))]
            if not len(vals):
                return (None, None, None)
            return (float(vals.min()), float(vals.median()), float(vals.max()))
        e_lo, e_med, e_hi = _finite_summary("energy_margin_Wh")
        t_lo, t_med, t_hi = _finite_summary("time_drcc_margin_s")
        q_lo, q_med, q_hi = _finite_summary("time_drcc_tightening_s")
        state_counts = (audit_df["recovery_state"].fillna("unknown").astype(str)
                        .value_counts().sort_index().to_dict()
                        if "recovery_state" in audit_df else {})
        diag_summary = dict(
            uav=str(uk), uav_label=str(p_u.uav_label),
            columns=int(len(audit_df)),
            multi_stop_columns=int((audit_df["stops"] >= 2).sum()),
            max_stops=int(audit_df["stops"].max()) if len(audit_df) else 0,
            energy_margin_min_Wh=e_lo, energy_margin_median_Wh=e_med,
            energy_margin_max_Wh=e_hi,
            time_drcc_margin_min_s=t_lo, time_drcc_margin_median_s=t_med,
            time_drcc_margin_max_s=t_hi,
            time_drcc_tightening_min_s=q_lo, time_drcc_tightening_median_s=q_med,
            time_drcc_tightening_max_s=q_hi,
            recovery_state_counts_json=json.dumps(
                state_counts, ensure_ascii=False, sort_keys=True),
            diagnostics_source="certified-complete-universe-retained-columns",
            universe_context_sha256=str(universe.context_sha256),
            universe_columns_sha256=str(universe.columns_sha256))
        _save(pd.DataFrame([diag_summary]), Path(outdir),
              f"E1_complete_route_universe_diagnostics_{uk}.csv")
        log.info(
            "E1[%s] 完整宇宙可行列诊断: cols=%d multi-stop=%d "
            "time_margin[min/med/max]=%s/%s/%s s, "
            "energy_margin[min/med/max]=%s/%s/%s Wh",
            uk, int(diag_summary["columns"]), int(diag_summary["multi_stop_columns"]),
            t_lo, t_med, t_hi, e_lo, e_med, e_hi)
        by = {}
        for c in universe.columns:
            key = (len(c.get("ordered_tids", c.get("tids", ()))), float(c["h"]))
            by[key] = by.get(key, 0) + 1
        rows = [dict(uav=uk, stops=int(k[0]), h_min=float(k[1]), columns=int(v))
                for k, v in sorted(by.items())]
        _save(pd.DataFrame(rows), Path(outdir),
              f"E1_complete_route_universe_counts_{uk}.csv")
    if not universe.complete:
        raise SystemExit(
            f"E1[{uk}] 完整路线宇宙未在给定时限内闭合: "
            f"{universe.stats.get('reason')}. "
            "正式 small-n 加速不会把不完整列池当完整证书；请增大 "
            "--formal-route-universe-time-limit-s，或显式 --formal-route-universe off "
            "退回原隐式 exact BPC。")
    log.info(
        "E1[%s] 完整路线宇宙认证完成: columns=%d seq=%d route-h=%d "
        "multi-stop=%d sha=%s",
        uk, len(universe.columns),
        int(universe.stats.get("evaluated_sequences", 0)),
        int(universe.stats.get("evaluated_route_h", 0)),
        sum(len(c.get("ordered_tids", c.get("tids", ()))) >= 2
            for c in universe.columns),
        str(universe.columns_sha256)[:16])
    cache[uk] = universe
    return universe


def _uav_deck(args, prof_key):
    """解析该 UAV 档位下的 (t_swap, t_launch): CLI 显式传值全局覆盖, 否则用档位默认(OEM 表 SCN)。"""
    prof = M.UAV_PROFILES[prof_key]
    t_swap = float(args.t_swap_min) if args.t_swap_min is not None else float(prof["t_swap_min"])
    t_launch = (float(args.t_launch_min) if args.t_launch_min is not None
                else float(prof["t_launch_min"]))
    return t_swap, t_launch


def _stops_cap(spec, p_u, xi_amb, fallback):
    """更新: 逐 UAV 解析停靠数【生成上界】stops_cap(E1/E2 共用, 消除删失)。
    动机(真航迹 E1 实测): max_stops=4 时 L 档 mean_stops 全程钉死 4.0 = 上界本身 ——
    能力轴被实验参数删失(censored)而非物理揭示, multi_stop_ratio 三档恒 1.0 失去区分度。
    spec:
      'auto'(默认)  = max(fallback, ⌊max(决策 h 网格)/τ_insp⌋) —— 纯【时间预算逻辑上界】
                      (一个回收窗内装不下更多台巡检), 不含物理魔数; 逐档差异交还 DRCC 本身
                      (能量/时间/着舰门)判定, 生成上界只保证【不删失】。默认网格 h_max=45,
                      τ_insp=5min ⇒ cap=9; 与 更新 自适应 h 网格联动 —— 作者本机扩
                      step7 --horizons 后 h_max=60 ⇒ cap 自动=12, 无须改代码。
      整数字符串     = 全档统一上界(如 '4' 完全复现 更新 口径);
      逐档映射       = 'S:4,M:5,L:6'(缺档回退 fallback)。
    注: 这是【生成/标号上界】(算法口径), 非模型参数 —— DRCC 不可行的长路由在池构造时
    即被剪掉, S 档不会因 cap=9 生成 4+ 停的列(能量先绑定), 池规模不会失控。"""
    s = str(spec).strip()
    if s.lower() == "auto":
        h_max = float(max(RM.decision_horizons_of(xi_amb)))
        return max(int(fallback), int(h_max // (p_u.tau_insp / 60.0)))
    if ":" in s:
        m = {}
        for kv in s.split(","):
            k, _, v = kv.partition(":")
            if k.strip() and v.strip():
                m[k.strip()] = int(v)
        return int(m.get(getattr(p_u, "uav_key", ""), fallback))
    return int(s)


def _audit_e1_validation_specificity(df: pd.DataFrame, outdir: Path):
    """Audit that E1 empirical validation is recomputed from each plan's route records.

    Equal empirical rates across different plans are not automatically an error:
    discrete holdout counts can coincide.  What is forbidden is a row whose
    reported ``emp_viol`` is inconsistent with its own route-level counts.  The
    audit also records how many distinct frozen-plan fingerprints share each
    empirical rate so an apparently constant validation signal is explicit.
    """
    rows = []
    route_detail_rows = []
    if df is None or df.empty:
        return pd.DataFrame(rows)
    for idx, r in df.iterrows():
        try:
            covered = int(r.get("covered", 0) or 0)
        except Exception:
            covered = 0
        if covered <= 0:
            continue
        raw = r.get("validation_route_records_json")
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            rows.append(dict(row_index=int(idx), uav=str(r.get("uav")),
                             K=r.get("K"), batteries=r.get("batteries"),
                             validation_plan_fingerprint=r.get("validation_plan_fingerprint"),
                             emp_viol=r.get("emp_viol"),
                             aggregation_consistent=False,
                             reason="missing_route_validation_records"))
            continue
        try:
            recs = json.loads(str(raw))
        except Exception as exc:
            rows.append(dict(row_index=int(idx), uav=str(r.get("uav")),
                             K=r.get("K"), batteries=r.get("batteries"),
                             validation_plan_fingerprint=r.get("validation_plan_fingerprint"),
                             emp_viol=r.get("emp_viol"),
                             aggregation_consistent=False,
                             reason=f"invalid_route_validation_records:{type(exc).__name__}"))
            continue
        for ridx, q in enumerate(recs):
            route_detail_rows.append(dict(
                row_index=int(idx), route_index=int(ridx), uav=str(r.get("uav")),
                K=r.get("K"), batteries=r.get("batteries"),
                validation_plan_fingerprint=r.get("validation_plan_fingerprint"),
                tau=q.get("tau"), h=q.get("h"),
                ordered_tids=">".join(map(str, q.get("ordered_tids", ()))),
                n_test_total=q.get("n_test_total"), n_viol_any=q.get("n_viol_any"),
                viol_rate_any=q.get("viol_rate_any"),
                upper95_simultaneous=q.get("upper95_simultaneous"),
                allocated_epsilon=q.get("allocated_epsilon"),
                mission_epsilon=q.get("mission_epsilon"),
                allocation_budget_holds=q.get("allocation_budget_holds"),
                mission_requirement_holds=q.get("mission_requirement_holds"),
                holdout_cell_sha256=q.get("holdout_cell_sha256"),
                observed_failure_mask_sha256=q.get("observed_failure_mask_sha256"),
                validation_event_fingerprint=q.get("validation_event_fingerprint"),
                validation_event_fingerprint_scope=q.get("validation_event_fingerprint_scope"),
                validation_complete=q.get("validation_complete")))
        n_tot = sum(int(q.get("n_test_total", 0) or 0) for q in recs
                    if bool(q.get("validation_complete", False)))
        n_viol = sum(int(q.get("n_viol_any", 0) or 0) for q in recs
                     if bool(q.get("validation_complete", False))
                     and q.get("n_viol_any") is not None)
        expected = (round(n_viol / n_tot, 4) if n_tot else None)
        actual = r.get("emp_viol")
        if actual is None or (isinstance(actual, float) and math.isnan(actual)):
            ok = expected is None
        else:
            ok = expected is not None and float(actual) == float(expected)
        rows.append(dict(
            row_index=int(idx), uav=str(r.get("uav")), K=r.get("K"),
            batteries=r.get("batteries"),
            validation_plan_fingerprint=r.get("validation_plan_fingerprint"),
            emp_viol=actual, reconstructed_emp_viol=expected,
            validation_n_total_from_routes=int(n_tot),
            validation_n_viol_from_routes=int(n_viol),
            route_records=len(recs), aggregation_consistent=bool(ok),
            reason=("ok" if ok else "emp_viol_route_count_mismatch")))
    audit = pd.DataFrame(rows)
    route_detail = pd.DataFrame(route_detail_rows)
    if not route_detail.empty:
        _save(route_detail, outdir, "E1_validation_route_detail.csv")
    if not audit.empty:
        _save(audit, outdir, "E1_validation_specificity_audit.csv")
        bad = audit[~audit["aggregation_consistent"].astype(bool)]
        if not bad.empty:
            raise RuntimeError(
                "E1 validation aggregation audit failed: emp_viol is not derived "
                "from the row's own route-level holdout counts.")
        for val, g in audit.dropna(subset=["emp_viol"]).groupby("emp_viol"):
            fps = set(g["validation_plan_fingerprint"].dropna().astype(str))
            if len(fps) > 1:
                log.warning(
                    "E1 validation audit: emp_viol=%s is shared by %d distinct "
                    "plan fingerprints; counts are internally consistent, so this "
                    "is recorded as a diagnostic coincidence rather than reused evidence.",
                    val, len(fps))
    return audit


def _e1_detail_rows(chosen, p_u, xi_amb, wamb, uk):
    """更新: 逐 UAV 的代表配置飞行明细 + 逐列鲁棒裕度与绑定约束诊断。
    动机(真航迹 E1 实测): S/M 覆盖逐位相同(M 被支配)、M 474Wh 也做不到 4 停 —— 需要逐列
    margin_E/margin_T 才能回答"卡在能量还是时间"(供论文 discussion, 免去二次仪器化重跑)。
    binding = 归一化短板: margin_E/B_use vs margin_T/(60h)(两者化为预算占比后可比)。"""
    out, orig_kappa = [], RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["vp_unimodal"]      # 与 E1 池构造同一判据, 裕度口径一致
        for c in sorted(chosen, key=lambda c: c["tau"]):
            dd = RM.route_feasible_at_h(c["route"], int(c["h"]), p_u, c["wx"], xi_amb,
                                        weather_unc=wamb, chance_mode="drcc")
            schedule = RM.route_nominal_schedule(
                c["route"], float(c["h"]), p_u, c["wx"],
                t_dock_s=float(dd["t_dock_s"]))
            relE = float(dd["margin_E"]) / max(float(p_u.B_use), 1e-9)
            relT = float(dd["margin_T"]) / max(60.0 * float(c["h"]), 1e-9)
            out.append(dict(result_contract=RESULT_CONTRACT,
                            tau_min=c["tau"], h_min=c["h"], recover_min=c["tau"] + c["h"],
                            stops=len(c["tids"]), turbines=";".join(map(str, c["tids"])),
                            E0_Wh=round(c["E0"], 1),
                            # Formal on-demand BPC columns are recomputed under the
                            # E1-wide vp_unimodal criterion but need not carry the
                            # historical pre-enumeration-only ``kappa_used`` field.
                            kappa=c.get("kappa_used", "vp_unimodal"),
                            c_state=c["ship"].c_state, uav=uk,
                            margin_E_Wh=round(float(dd["margin_E"]), 1),
                            margin_T_s=round(float(dd["margin_T"]), 0),
                            rel_margin_E=round(relE, 4), rel_margin_T=round(relT, 4),
                            binding=("energy" if relE <= relT else "time"),
                            E_flight_Wh=round(float(dd.get("E_flight_Wh", 0.0)), 1),
                            E_escort_Wh=round(float(dd.get("E_escort_Wh", 0.0)), 1),
                            E_dock_Wh=round(float(dd["E_dock_Wh"]), 1),
                            E_plan_Wh=round(float(dd.get("E_plan_Wh", c.get("E_plan_Wh", c["E0"]))), 1),
                            E_soc_required_Wh=round(float(dd.get("E_soc_required_Wh",
                                                                  c.get("E_soc_required_Wh", c["E0"]))), 1),
                            t_dock_s=round(float(dd["t_dock_s"]), 0),
                            uav_id=c.get("uav_id"), battery_group=c.get("battery_group"),
                            post_service_mode=c.get("post_service_mode", "none_after_last_mission"),
                            post_service_start_min=((c.get("post_service_interval") or [None, None])[0]),
                            post_service_end_min=((c.get("post_service_interval") or [None, None])[1]),
                            schedule_json=json.dumps(schedule, ensure_ascii=False,
                                                     sort_keys=True, separators=(",", ":"))))
    finally:
        RM.kappa = orig_kappa
    return out


def E1_frontier(reach, opts, p_base, xi_amb, wamb, outdir, args, kind, T_eff):
    """【E1·主实验, 更新 修订(基于真航迹 162 行首跑的问题清单)】UAV 种类 × K × B:
        uavs ∈ --e1-uavs(默认 S,M,L; OEM 档位见 step9.UAV_PROFILES)
        K    ∈ --fleet-ks(更新 默认 1..8: 首跑 K∈3..8 全平坦=零信息量, 加 1,2 定位 K 何时真正绑定)
        B    ∈ --e1-batteries(基础网格 0..8; B=0 为退化锚点)+【饱和自动延伸】(更新):
              formal 模式不再用 validation 后的 safe_served 零边际声称饱和；每格先返回
              C_inc<=C*<=UB_C 的严格覆盖区间，并利用 K/B 资源嵌套单调性做端点 sandwich：
              若 K_max 上 B0<B1 且 C_inc(B0)=UB_C(B1)=P，则区间内 C*=P。只有该证明或
              hard coverable cap 才停止自动延伸；机制模式仍保留观测 safe_served 规则。
              上限 --e1-b-cap; --e1-b-auto off 只执行固定基础网格。
    停靠上界逐 UAV 解析(--stops-cap, 默认 auto=⌊h_max/τ_insp⌋, 见 _stops_cap): 消除 更新
    max_stops=4 对 L 档的删失; 每行新增 stops_cap 与 stops_cap_hit(命中=该行结果仍可能被截断)。
    明细逐 UAV 各存一份 E1_detail_Kmax_<uav>.csv(最大资源/平台代表方案);
    通用 E1_detail_Kmax.csv 在完成选型后重新求解并保存【最终 selected knee】方案，
    文件内 solution_role/uav/K/batteries 可自证其语义。
    正式模式不预建多停靠路线池：每个 K/B 格由同一按需精确定价入口求解；机制实验可显式
    保留旧共享列池诊断，但该列池及其 Gap 不参与正式全局证书。
    口径(作者立正): headline = safe_served 与 台/电池、台/架; coverable 只是可达诊断。"""
    uavs = [x.strip() for x in str(args.e1_uavs).split(",") if x.strip()]
    # 更新: --uav 默认已为 'auto'(E2/A 自动回填用); E1 的旧口径明细档回退首个档
    legacy_uk = args.uav if args.uav in uavs else uavs[0]
    ks = tuple(sorted(int(x) for x in str(args.fleet_ks).split(",") if x.strip()))
    bs_base = sorted(int(x) for x in str(args.e1_batteries).split(",") if x.strip())
    # ---- Strict resume provenance: bind the actual binary64 instance inputs. ----
    _resume_input_sha256 = _resume_context_sha256(
        reach=reach, launch_options=opts, params=p_base, xi_ambiguity=xi_amb,
        weather_uncertainty=wamb, track_kind=kind, T_eff_min=float(T_eff))
    setattr(args, "_resume_input_sha256", _resume_input_sha256)
    # ---- 更新 任务4: 断点续跑 —— E1_frontier.csv 本身即检查点 ----
    _sig = dict(track=kind, T_eff_min=round(float(T_eff), 1), eps=p_base.eps_E,
                physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
                route_identity_contract=BP.ROUTE_IDENTITY_CONTRACT,
                model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
                resume_input_sha256=_resume_input_sha256,
                dtau_min=args.dtau_min, deck_delta_min=args.deck_delta_min,
                deck_mode=args.deck_mode, replay_n=args.replay_n, result_contract=RESULT_CONTRACT,
                formal_experiment_scheduler_contract=FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
                max_stops_requested=int(args.max_stops), stops_cap_spec=str(args.stops_cap),
                saa_source=RM.SAA_SOURCE,
                xi_source=getattr(args, "_xi_source", "unknown"),
                xi_mmsi=getattr(args, "_resolved_xi_mmsi", "ALL"),
                xi_predictor=getattr(args, "_resolved_xi_predictor", "unknown"),
                xi_predictor_contract=getattr(args, "_resolved_xi_predictor_contract", "unknown"),
                weather_uncertainty_source=getattr(args, "_weather_uncertainty_source", "off"),
                weather_risk_contract="vector-route-scalar-landing",
                route_airspeed_contract="per-leg-along-cross-projection",
                pair_radius_m=round(float(getattr(args, "_pair_radius_m", -1.0)), 1),
                soc_correction=getattr(p_base, "soc_correction", "none"),
                soc_risk_allocation=getattr(p_base, "soc_risk_allocation", "fixed"),
                time_recourse_mode=getattr(p_base, "time_recourse_mode", "wait_only"),
                time_contract_id=RM.time_contract_for(p_base),
                speed_is_recourse=bool(getattr(p_base, "speed_adjustable", False)),
                return_speed_recourse_contract=(RM.SPEED_RECOURSE_CONTRACT if getattr(p_base, "speed_adjustable", False) else None),
                energy_recourse_contract=(RM.ENERGY_SPEED_RECOURSE_CONTRACT if getattr(p_base, "speed_adjustable", False) else None),
                geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
                predictor=getattr(args, "recovery_predictor", "cv_noleak"),
                pool_h=getattr(args, "pool_h", "pareto"),
                validation_mode=getattr(args, "validation_mode", "synthetic_stress"),
                validation_samples_hash=(EU.sha256_file(getattr(args, "validation_samples", None)) or "none"),
                xi_train_samples_hash=(EU.sha256_file(getattr(args, "xi_train_samples", None)) or "none"),
                final_test_samples_hash=(EU.sha256_file(getattr(args, "final_test_samples", None)) or "none"),
                study_mode=getattr(args, "study_mode", "mechanism"),
                quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                swap_station_capacity=int(getattr(args, "swap_stations", 1)),
                battery_reuse_mode=getattr(args, "battery_reuse_mode", "exact_soc"),
                recovery_target_model=str(getattr(p_base, "recovery_target_model", "discrete_horizon_ship_prediction")),
                terminal_sensor_error_mode=str(getattr(p_base, "terminal_sensor_error_mode", "out_of_scope")),
                solver_mode=getattr(args, "solver_mode", "exact-branch-price-cut"),
                pricing_mode=getattr(args, "pricing_mode", "exact-implicit-dfs"),
                archive_diagnostic_time_limit_s=float(getattr(
                    args, "archive_diagnostic_time_limit_s", 30.0)),
                archive_shadow_diagnostic_time_limit_s=float(getattr(
                    args, "archive_shadow_diagnostic_time_limit_s", 30.0)),
                archive_clique_diagnostic_time_limit_s=float(getattr(
                    args, "archive_clique_diagnostic_time_limit_s", 30.0)),
                formal_warmstart_seconds=float(getattr(args, "formal_warmstart_seconds", 60.0)),
                formal_route_universe=str(getattr(args, "formal_route_universe", "auto")),
                formal_route_universe_max_turbines=int(getattr(
                    args, "formal_route_universe_max_turbines", 8)),
                formal_route_universe_max_stops=int(getattr(
                    args, "formal_route_universe_max_stops", 4)),
                formal_route_universe_time_limit_s=float(getattr(
                    args, "formal_route_universe_time_limit_s", 7200.0)),
                weather_alignment_mode=(getattr(args, "_wx0", {}) or {}).get("weather_alignment_mode", "unknown"),
                weather_start_time=str((getattr(args, "_wx0", {}) or {}).get("weather_start_time", "unknown")),
                weather_match_error_min=(float((getattr(args, "_wx0", {}) or {}).get("weather_match_error_min"))
                                         if (getattr(args, "_wx0", {}) or {}).get("weather_match_error_min") is not None
                                         else -1.0))
    rows, _done = _resume_load(
        outdir, "E1_frontier.csv", ["uav", "K", "batteries"],
        _sig, getattr(args, "resume", "on"),
        completed_status_col="frontier_completion_state",
        completed_values=("coverage-certified", "lexicographic-certified"))
    row_index = {(str(r.get("uav")), int(r.get("K")), int(r.get("batteries"))): i
                 for i, r in enumerate(rows)
                 if r.get("uav") is not None and r.get("K") is not None
                 and r.get("batteries") is not None}
    effective_caps = {}
    uav_context = {}
    certified_knee_results = {}
    for uk in uavs:
        p_u = M.apply_uav_profile(p_base, uk)
        t_swap, t_launch = _uav_deck(args, uk)
        cap_u = _stops_cap(args.stops_cap, p_u, xi_amb, args.max_stops)
        effective_caps[uk] = int(cap_u)
        prov = _provenance(args, kind, T_eff, reach, p_u, t_swap, t_launch,
                           max_stops_val=cap_u, wx0=getattr(args, "_wx0", None))
        route_ledger = []
        formal_ondemand = _formal_ondemand_pricing(args)
        route_universe = None
        if formal_ondemand:
            route_universe = _build_or_get_e1_complete_universe(
                reach, opts, p_u, xi_amb, wamb, args, T_eff,
                cap_u, t_launch, uk, outdir)
        if formal_ondemand and route_universe is not None:
            # v12 small-n exact acceleration: the complete physical universe
            # already contains every possible heuristic seed, so warm-start
            # construction is redundant and intentionally skipped.
            cols = []
            stage_counts = dict(
                status="formal-certified-complete-route-universe",
                warmstart_status="superseded-by-complete-universe",
                warmstart_seconds=0.0,
                formal_prebuilt_route_pool=True,
                heuristic_seed_candidates=0,
                heuristic_multistop_seed_candidates=0,
                final_shared_pool=len(route_universe.columns),
                route_universe_complete=True,
                route_universe_columns=len(route_universe.columns),
                route_universe_columns_sha256=route_universe.columns_sha256,
                route_universe_builder_contract=route_universe.builder_contract,
            )
        elif formal_ondemand:
            # Large-instance exact fallback: bounded heuristic seeds are primal
            # candidates only; every supplied seed is revalidated and implicit
            # exact pricing remains mandatory for certificates.
            cols = []
            warm_s = max(0.0, float(getattr(args, "formal_warmstart_seconds", 60.0)))
            warm_status = "disabled"
            if warm_s > 0.0:
                try:
                    cols = RA.build_route_columns(
                        reach, opts, p_u, xi_amb, T_eff, args.deck_delta_min,
                        cap_u, wamb, "drcc", 2.0, "vp_unimodal", 8.0,
                        pool_h_mode=getattr(args, "pool_h", "pareto"),
                        diagnostics_sink=None,
                        deadline=time.monotonic() + warm_s)
                    warm_status = "completed"
                except TimeoutError:
                    cols = []
                    warm_status = "timeout-no-seeds"
                except Exception as exc:
                    cols = []
                    warm_status = f"error-no-seeds:{type(exc).__name__}"
            stage_counts = dict(
                status="formal-revalidated-heuristic-warmstart",
                warmstart_status=warm_status,
                warmstart_seconds=float(warm_s),
                formal_prebuilt_route_pool=False,
                heuristic_seed_candidates=len(cols),
                heuristic_multistop_seed_candidates=sum(
                    len(c.get("ordered_tids", c.get("tids", ()))) >= 2 for c in cols),
                final_shared_pool=0,
                route_universe_complete=False,
            )
        else:
            # 仅机制实验/研究基线保留旧共享列池诊断；其列池 Gap 不具全局证书意义。
            cols = RA.build_route_columns(reach, opts, p_u, xi_amb, T_eff, args.deck_delta_min,
                                     cap_u, wamb, "drcc", 2.0, "vp_unimodal", 8.0,
                                     pool_h_mode=getattr(args, "pool_h", "pareto"),
                                     diagnostics_sink=route_ledger)
            stage_counts = dict(getattr(RA.build_route_columns, "last_stage_counts", {}))
        if route_ledger:
            flat_ledger = pd.json_normalize(route_ledger, sep="_")
            _save(flat_ledger, outdir, f"E1_route_diagnostics_{uk}.csv")
            failure_cols = [c for c in flat_ledger.columns if c.startswith("failure_flags_")]
            if failure_cols:
                fs = []
                for col in failure_cols:
                    vals = flat_ledger[col].map(lambda v: bool(v) if pd.notna(v) else False)
                    fs.append(dict(uav=uk, constraint=col.replace("failure_flags_", ""),
                                   failed_count=int(vals.sum()), total_evaluated=int(len(flat_ledger)),
                                   failed_fraction=round(float(vals.mean()), 8)))
                _save(pd.DataFrame(fs), outdir, f"E1_failure_summary_{uk}.csv")
        _save(pd.DataFrame([dict(uav=uk, **stage_counts)]), outdir,
              f"E1_route_stage_counts_{uk}.csv")
        uav_context[uk] = dict(
            p=p_u, t_swap=t_swap, t_launch=t_launch, cap=cap_u,
            seed_cols=(list(cols) if cols else None),
            route_universe=route_universe,
            route_ledger=route_ledger, stage_counts=stage_counts)
        if formal_ondemand and route_universe is not None:
            log.info(
                "E1[%s=%s]: stops_cap=%d，使用认证完整物理路线宇宙 %d 列；"
                "所有 K/B/target 节点无遗漏列，不再重复隐式全排列定价。",
                uk, p_u.uav_label, cap_u, len(route_universe.columns))
        elif formal_ondemand:
            log.info("E1[%s=%s]: stops_cap=%d，formal warm-start候选=%d(其中multi-stop=%d)；"
                     "全部由exact BPC重验证，证书仍依赖完整按需精确定价",
                     uk, p_u.uav_label, cap_u, len(cols),
                     sum(len(c.get("ordered_tids", c.get("tids", ()))) >= 2 for c in cols))
        else:
            log.info("E1[%s=%s]: stops_cap=%d 研究诊断列池 %d 条",
                     uk, p_u.uav_label, cap_u, len(cols))
        inside = [rec for rec in route_ledger if rec.get("primary_reason") != "outside_mission_window"]
        if formal_ondemand and route_universe is not None and not inside:
            log.info(
                "E1[%s] complete-universe 物理可行性已对全部 seq/route-h 穷举；"
                "旧 heuristic route_ledger 不适用于该路径，因此 time_evaluable/"
                "median_tightening 不再伪报 0/NaN。详细正式列见 "
                "E1_complete_route_universe_%s.csv。", uk, uk)
        def _time_evaluable(rec):
            flags = rec.get("failure_flags") or {}
            if any(bool(flags.get(k, False)) for k in (
                    "missing_xi_cell", "missing_recovery_prediction",
                    "missing_recovery_weather", "risk_budget_invalid")):
                return False
            vals = []
            for name in ("time_core_nom_s", "time_drcc_tightening_s", "time_drcc_margin_s"):
                try:
                    vals.append(float(rec.get(name)))
                except (TypeError, ValueError):
                    return False
            return all(math.isfinite(v) and abs(v) < 1.0e20 for v in vals)
        time_rows = [rec for rec in inside if _time_evaluable(rec)]
        missing_xi_rows = sum(bool((rec.get("failure_flags") or {}).get("missing_xi_cell", False))
                              for rec in inside)
        nominal_time_feasible = sum(
            1 for rec in time_rows if not bool((rec.get("failure_flags") or {}).get("nominal_time_failed", False)))
        time_drcc_feasible = sum(
            1 for rec in time_rows if not bool((rec.get("failure_flags") or {}).get("time_drcc_failed", False)))
        tightenings = [float(rec["time_drcc_tightening_s"]) for rec in time_rows]
        time_margins = [float(rec["time_drcc_margin_s"]) for rec in time_rows]
        median_tightening = float(np.median(tightenings)) if tightenings else float("nan")
        best_margin = max(time_margins) if time_margins else float("nan")
        worst_margin = min(time_margins) if time_margins else float("nan")
        if inside or route_universe is None:
            log.info("E1[%s] time_contract=%s time_evaluable=%d missing_xi=%d "
                     "nominal_time_feasible=%d time_drcc_feasible=%d "
                     "median_tightening=%.3fs best_time_drcc_margin_s=%.3fs "
                     "worst_time_drcc_margin_s=%.3fs",
                     uk, RM.time_contract_for(p_u), len(time_rows), int(missing_xi_rows),
                     nominal_time_feasible, time_drcc_feasible,
                     median_tightening, best_margin, worst_margin)
        def _median_field(name):
            vals = [float(rec[name]) for rec in time_rows if rec.get(name) is not None
                    and math.isfinite(float(rec[name])) and abs(float(rec[name])) < 1.0e20]
            return float(np.median(vals)) if vals else float("nan")
        bases = {str(rec.get("time_feasibility_basis", "core_time")) for rec in time_rows}
        if bases == {"return_required_airspeed"}:
            log.info(
                "E1[%s] speed_recourse_certificate_median: return_budget_nom=%.3fs "
                "return_budget_safe=%.3fs nonreturn_weather_reserve=%.3fs "
                "required_nom=%.3fm/s required_safe=%.3fm/s airspeed_margin=%.3fm/s "
                "power_envelope=%.3fW energy_envelope=%.3fWh",
                uk, _median_field("return_time_budget_s"),
                _median_field("return_time_budget_safe_s"),
                _median_field("nonreturn_weather_reserve_s"),
                _median_field("return_required_airspeed_nom_ms"),
                _median_field("return_required_airspeed_safe_ms"),
                _median_field("return_airspeed_margin_ms"),
                _median_field("return_power_envelope_W"),
                _median_field("return_energy_envelope_Wh"))
        elif inside or route_universe is None:
            log.info(
                "E1[%s] time_tightening_components_median_s: xi_mean=%.3f xi_std=%.3f "
                "geometry_correction=%.3f xi_geo_total=%.3f weather_mean=%.3f "
                "weather_std=%.3f weather_total=%.3f allocation=%s",
                uk, _median_field("time_xi_mean_shift_s"), _median_field("time_xi_std_term_s"),
                _median_field("time_geometry_correction_s"), _median_field("time_xi_geo_total_s"),
                _median_field("time_weather_mean_shift_s"), _median_field("time_weather_std_term_s"),
                _median_field("time_weather_total_s"),
                str(getattr(p_u, "soc_risk_allocation", "fixed")))
        finite_rows = list(time_rows)
        if finite_rows:
            best_rec = max(finite_rows, key=lambda rec: float(rec["time_drcc_margin_s"]))
            if str(best_rec.get("time_feasibility_basis", "core_time")) == "return_required_airspeed":
                log.info(
                    "E1[%s] best_time_candidate[speed-recourse]: tau=%s h=%s state=%s turbines=%s "
                    "physical_core=%.3fs nominal_wait=%.3fs return_budget_nom=%.3fs "
                    "return_budget_safe=%.3fs nonreturn_weather_reserve=%.3fs "
                    "required_nom=%.3fm/s required_safe=%.3fm/s airspeed_margin=%.3fm/s "
                    "equiv_time_margin=%.3fs xi_std_along=%.3fm xi_std_cross=%.3fm "
                    "d_ret0=%.3fm eps_nonreturn=%.6g eps_return=%.6g eps_along=%.6g eps_cross=%.6g "
                    "xi_state_change_rate=%.3f xi_recovery_mode=%s",
                    uk, best_rec.get("tau"), best_rec.get("h"), best_rec.get("launch_state"),
                    best_rec.get("turbines"), float(best_rec.get("time_core_nom_s", float("nan"))),
                    float(best_rec.get("time_wait_nom_s", float("nan"))),
                    float(best_rec.get("return_time_budget_s", float("nan"))),
                    float(best_rec.get("return_time_budget_safe_s", float("nan"))),
                    float(best_rec.get("nonreturn_weather_reserve_s", float("nan"))),
                    float(best_rec.get("return_required_airspeed_nom_ms", float("nan"))),
                    float(best_rec.get("return_required_airspeed_safe_ms", float("nan"))),
                    float(best_rec.get("return_airspeed_margin_ms", float("nan"))),
                    float(best_rec.get("time_drcc_margin_s", float("nan"))),
                    float(best_rec.get("xi_std_along_m", float("nan"))),
                    float(best_rec.get("xi_std_cross_m", float("nan"))),
                    float(best_rec.get("d_ret0_m", float("nan"))),
                    float(best_rec.get("eps_time_nonreturn_weather", float("nan"))),
                    float(best_rec.get("eps_time_return_required_airspeed", float("nan"))),
                    float(best_rec.get("eps_time_along", float("nan"))),
                    float(best_rec.get("eps_time_cross", float("nan"))),
                    float(best_rec.get("xi_launch_to_recovery_state_change_rate", float("nan"))),
                    str(best_rec.get("xi_actual_recovery_state_mode", "unknown")))
            else:
                log.info(
                    "E1[%s] best_time_candidate: tau=%s h=%s state=%s turbines=%s "
                    "core=%.3fs nominal_wait=%.3fs tightening=%.3fs margin=%.3fs "
                    "xi_geo=%.3fs weather=%.3fs geo_correction=%.3fs "
                    "xi_std_along=%.3fm xi_std_cross=%.3fm d_ret0=%.3fm "
                    "eps_weather=%.6g eps_along=%.6g eps_cross=%.6g "
                    "xi_state_change_rate=%.3f xi_recovery_mode=%s",
                    uk, best_rec.get("tau"), best_rec.get("h"), best_rec.get("launch_state"),
                    best_rec.get("turbines"), float(best_rec.get("time_core_nom_s", float("nan"))),
                    float(best_rec.get("time_wait_nom_s", float("nan"))),
                    float(best_rec.get("time_drcc_tightening_s", float("nan"))),
                    float(best_rec.get("time_drcc_margin_s", float("nan"))),
                    float(best_rec.get("time_xi_geo_total_s", float("nan"))),
                    float(best_rec.get("time_weather_total_s", float("nan"))),
                    float(best_rec.get("time_geometry_correction_s", float("nan"))),
                    float(best_rec.get("xi_std_along_m", float("nan"))),
                    float(best_rec.get("xi_std_cross_m", float("nan"))),
                    float(best_rec.get("d_ret0_m", float("nan"))),
                    float(best_rec.get("eps_time_weather", float("nan"))),
                    float(best_rec.get("eps_time_along", float("nan"))),
                    float(best_rec.get("eps_time_cross", float("nan"))),
                    float(best_rec.get("xi_launch_to_recovery_state_change_rate", float("nan"))),
                    str(best_rec.get("xi_actual_recovery_state_mode", "unknown")))
            by_h = {}
            for rec in finite_rows:
                hh = rec.get("h")
                if hh is None:
                    continue
                by_h.setdefault(float(hh), []).append(rec)
            h_parts = []
            for hh in sorted(by_h):
                rr = by_h[hh]
                feasible_h = sum(float(x.get("time_drcc_margin_s", -1e300)) >= -RM.TIME_TOL_S for x in rr)
                best_h = max(float(x.get("time_drcc_margin_s", -1e300)) for x in rr)
                med_h = float(np.median([float(x.get("time_drcc_tightening_s", float("nan"))) for x in rr]))
                if all(str(x.get("time_feasibility_basis", "")) == "return_required_airspeed" for x in rr):
                    best_air = max(float(x.get("return_airspeed_margin_ms", -1e300)) for x in rr)
                    med_safe = float(np.median([float(x.get("return_required_airspeed_safe_ms", float("nan"))) for x in rr]))
                    h_parts.append(f"h={hh:g}:feas={feasible_h}/{len(rr)},best_eq={best_h:.1f}s,"
                                   f"best_air={best_air:.2f}m/s,med_req={med_safe:.2f}m/s")
                else:
                    h_parts.append(f"h={hh:g}:feas={feasible_h}/{len(rr)},best={best_h:.1f}s,med_tight={med_h:.1f}s")
            log.info("E1[%s] time_by_h: %s", uk, " | ".join(h_parts))
        if (not formal_ondemand) and not cols and reach and opts:
            # 研究诊断池为空 ≠ 静默零行 —— 打出可诊断的物理原因(standoff vs 本档外包络)。
            TP = np.array([t.local for t in reach], float)
            d_stand = [float(np.min(np.linalg.norm(TP - o.ship.P_launch, axis=1))) for o in opts]
            _dz = max(float(reach[0].H_tip) - float(p_u.z_cruise), 0.0)
            _R = M.max_flight_radius_m(p_u, float(max(RM.decision_horizons_of(xi_amb))), dz_insp_m=_dz)
            top_fail = []
            if route_ledger:
                counts = {}
                for rec in route_ledger:
                    for name, failed in (rec.get("failure_flags") or {}).items():
                        if failed:
                            counts[name] = counts.get(name, 0) + 1
                top_fail = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            log.error("E1[%s]: 共享列池=0，路线层已拒绝全部候选；资源优化未开始。窗内 standoff 中位 %.0fm "
                      "vs 本档静风外包络 R_max=%.0fm; c(τ)=动力定位 %d/%d 时隙；首要拒绝统计=%s。"
                      "请检查 E1_route_diagnostics_%s.csv，修复后用 --resume off 重跑。",
                      uk, float(np.median(d_stand)), _R["R_max_m"],
                      sum(1 for o in opts if o.ship.c_state == "动力定位"), len(opts),
                      top_fail, uk)
            log.error("路线池为空；资源层未启动；扩大 --e1-b-cap 不会改变结果。")
        # 更新: resume 回填该档已完成格(供 B 延伸饱和判据与跳过查询)
        vals = {(int(r["K"]), int(r["batteries"])): int(r["safe_served"])
                for r in rows if str(r.get("uav")) == str(uk)}
        chosen_at = {}

        # Cross-cell warm starts are acceleration only. Every seed is revalidated
        # by the formal physical oracle inside solve_fleet_anytime, so this cannot
        # carry feasibility or certificate semantics across resource cells.
        shared_seed_cols = list(cols[:64]) if (formal_ondemand and cols) else []
        shared_seed_keys = set()
        for _c in shared_seed_cols:
            try:
                shared_seed_keys.add(
                    (float(_c["tau"]), float(_c["h"]),
                     tuple(_c.get("ordered_tids", _c.get("tids", ())))))
            except Exception:
                pass
        shared_seed_cap = 64

        def _remember_seed_columns(columns):
            for c in (columns or []):
                try:
                    key = (float(c["tau"]), float(c["h"]),
                           tuple(c.get("ordered_tids", c.get("tids", ()))))
                except Exception:
                    continue
                if not key[2] or key in shared_seed_keys:
                    continue
                shared_seed_keys.add(key)
                shared_seed_cols.append(c)
                if len(shared_seed_cols) >= shared_seed_cap:
                    break

        def _cell(K, B, *, force=False, full_lex=False, coverage_certify=False):
            checkpoint_key = (str(uk), str(int(K)), str(int(B)))
            if checkpoint_key in _done and not force:
                return vals[(int(K), int(B))]
            solver_kw = (_e1_certify_solver_kwargs(args) if full_lex else
                         _e1_coverage_certify_solver_kwargs(args) if coverage_certify else
                         _e1_frontier_solver_kwargs(args))
            r = BP.solve_fleet_anytime(
                reach, opts, p_u, xi_amb, K, T_eff,
                deck_delta_min=args.deck_delta_min, t_swap_min=t_swap,
                landing_clear_min=args.landing_clear_min,
                swap_station_capacity=args.swap_stations,
                quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                battery_reuse_mode=args.battery_reuse_mode,
                max_stops=cap_u, weather_unc=wamb,
                kappa_mode="vp_unimodal", batteries=B,
                seed_cols=((list(shared_seed_cols) if shared_seed_cols else None)
                           if formal_ondemand else cols),
                deck_mode=args.deck_mode, t_launch_min=t_launch,
                solver_mode="exact-branch-price-cut",
                certified_route_universe=route_universe,
                **solver_kw)
            _vmode = getattr(args, "validation_mode", "synthetic_stress")
            rp = _replay_columns(r["chosen"], p_u, xi_amb, p_u.eps_E,
                                 n_per=args.replay_n, seed=7, wamb=wamb,
                                 validation_mode=_vmode,
                                 real_samples_csv=getattr(args, "validation_samples", None),
                                 weather_sample_mode=("real" if _vmode in ("real_validation", "real_holdout")
                                                      else "synthetic"))
            safe = rp["safe_tids"]
            # 更新(采纳外部审计 6.3): 计划级 holds 阈与 _replay_columns.safe 同源 ——
            # 统一走 RM.mission_eps_budget(weather-on 含 ε_cap+ε_gate+ε_air), 不再硬编码 2ε。
            _bud = float(rp.get("allocation_budget",
                                RM.mission_eps_budget(p_u, wamb is not None)))
            plan_holds = _formal_validation_selection_gate(rp)
            strict_plan_holds = rp.get("mission_requirement_holds")
            route_pool_status, route_pool_count = _route_pool_metadata(formal_ondemand, cols, r)
            optimization_status = str(r.get("status", "solver_error"))
            if not r.get("chosen"):
                validation_status = "not_applicable_empty_plan"
                plan_holds = None
                strict_plan_holds = None
            elif rp.get("upper95") is None or int(rp.get("n_missing", 0)) > 0:
                validation_status = "incomplete_missing_samples"
            elif plan_holds:
                validation_status = "passed"
            else:
                validation_status = "failed"
            zero_coverage_reason = _zero_coverage_reason(formal_ondemand, cols, r, B)
            max_stops_observed = max((len(c["tids"]) for c in r["chosen"]), default=0)
            cap_hit = bool(max_stops_observed >= cap_u and len(r["chosen"]) > 0)
            inventory_kwh = float(B) * float(p_u.B_k) / 1000.0
            record = dict(uav=uk, uav_label=p_u.uav_label, B_k_Wh=p_u.B_k,
                             power_scale=round(p_u.power_scale, 4),
                             K=K, batteries=r["batteries"],
                             route_pool_status=route_pool_status, route_pool_count=route_pool_count,
                             prebuilt_route_pool_count=len(cols),
                             on_demand_generated_columns=(int(r.get("generated_columns", 0) or 0)
                                                          if formal_ondemand else 0),
                             nominal_time_feasible_count=int(nominal_time_feasible),
                             time_drcc_feasible_count=int(time_drcc_feasible),
                             median_time_drcc_tightening_s=median_tightening,
                             best_time_drcc_margin_s=best_margin,
                             worst_time_drcc_margin_s=worst_margin,
                             optimization_status=optimization_status,
                             validation_status=validation_status,
                             zero_coverage_reason=zero_coverage_reason,
                             covered=r["covered"], safe_served=len(safe),
                             safe_ratio=(round(len(safe) / r["covered"], 3)
                                         if r["covered"] else None),
                             per_battery=(round(len(safe) / B, 3) if B else None),
                             inventory_energy_kWh=round(inventory_kwh, 6),
                             safe_per_inventory_kWh=(round(len(safe) / inventory_kwh, 6)
                                                     if inventory_kwh > 0 else None),
                             per_drone=round(len(safe) / max(K, 1), 3),
                             flights=r["flights"], mean_stops=r["mean_stops"],
                             multi_stop_ratio=r["multi_stop_ratio"],
                             stops_cap=cap_u,
                             max_stops_observed=max_stops_observed, stops_cap_hit=cap_hit,
                             energy_Wh=r["energy_Wh"],
                             energy_per_safe=(round(r["energy_Wh"] / len(safe), 1)
                                              if safe else None),
                             makespan_min=r["makespan_min"],
                             emp_viol=rp["emp"], max_col_viol=rp["max_col_viol"],
                             validation_plan_fingerprint=rp.get("validation_plan_fingerprint"),
                             validation_route_records_json=json.dumps(
                                 rp.get("route_validation_records", []),
                                 ensure_ascii=False, sort_keys=True),
                             component_eps=round(float(p_u.eps_E), 6),
                             mission_eps_budget=round(_bud, 6),
                             eps_budget=round(_bud, 6),
                             allocation_budget=rp.get("allocation_budget"),
                             mission_requirement_budget=rp.get("mission_requirement_budget"),
                             allocation_budget_holds=rp.get("allocation_budget_holds"),
                             mission_requirement_holds=rp.get("mission_requirement_holds"),
                             all_routes_allocation_holds=rp.get("all_routes_allocation_holds"),
                             all_routes_mission_holds=rp.get("all_routes_mission_holds"),
                             validation_gate_contract=rp.get("validation_gate_contract"),
                             validation_event_fingerprint_count=rp.get("validation_event_fingerprint_count"),
                             validation_unique_event_fingerprint_count=rp.get(
                                 "validation_unique_event_fingerprint_count"),
                             validation_duplicate_event_groups=rp.get("validation_duplicate_event_groups"),
                             validation_event_group_sizes_json=rp.get(
                                 "validation_event_group_sizes_json"),
                             validation_event_grouping_used_for_gate=rp.get(
                                 "validation_event_grouping_used_for_gate"),
                             strict_5pct_holds=strict_plan_holds,
                             union_budget_holds=plan_holds,
                             plan_holds=plan_holds,
                             n_test_total=rp.get("n_test_total", 0),
                             n_viol_total=rp.get("n_viol_total", 0),
                             emp_viol_ci95_low=rp.get("ci95_low"),
                             emp_viol_ci95_high=rp.get("ci95_high"),
                             emp_viol_upper95=rp.get("upper95"),
                             n_replayed_cols=rp["n_replayed"],
                             n_missing_replay=rp["n_missing"],
                             n_realized=rp.get("n_realized", 0),
                             n_realized_viol=rp.get("n_realized_viol", 0),
                             realized_viol_ci95_low=rp.get("realized_ci95_low"),
                             realized_viol_ci95_high=rp.get("realized_ci95_high"),
                             realized_viol_upper95=rp.get("realized_upper95"),
                             validation_type=rp["validation_type"] + "+single-realized-audit",
                             disjoint_xi_holdout=rp.get("disjoint_xi_holdout", False),
                             disjoint_weather_holdout=rp.get("disjoint_weather_holdout", False),
                             disjoint_real_holdout=rp.get("disjoint_real_holdout", False),
                             independent_xi_holdout=False,
                             independent_weather_holdout=False,
                             independent_real_holdout=False,
                             coverable_note=r["coverable"], pool_size=r["pool_size"],
                             solver=r["solver"],
                             formal_algorithm=r.get("algorithm"),
                             route_universe_source=r.get("route_universe_source"),
                             route_space_complete=r.get("route_space_complete"),
                             route_space_materialized=r.get("route_space_materialized"),
                             complete_route_universe_contract=r.get(
                                 "complete_route_universe_contract"),
                             complete_route_universe_columns_sha256=r.get(
                                 "complete_route_universe_columns_sha256"),
                             termination_reason=r.get("termination_reason"),
                             coverage_incumbent=r.get("coverage_incumbent"),
                             coverage_upper_bound=r.get("coverage_upper_bound"),
                             coverage_gap_abs=r.get("coverage_gap_abs"),
                             coverage_gap_pct=r.get("coverage_gap_pct"),
                             coverage_optimal=r.get("coverage_optimal"),
                             energy_incumbent_Wh=r.get("energy_incumbent_Wh"),
                             energy_lower_bound_Wh=r.get("energy_lower_bound_Wh"),
                             energy_gap_abs_Wh=r.get("energy_gap_abs_Wh"),
                             energy_gap_pct=r.get("energy_gap_pct"),
                             conditional_energy_gap_pct=r.get("conditional_energy_gap_pct"),
                             global_energy_gap_reason=r.get("global_energy_gap_reason"),
                             lexicographic_optimal=r.get("lexicographic_optimal"),
                             global_certificate_available=_global_certificate_flag(r),
                             global_route_space_certificate=_global_certificate_flag(r),
                             implicit_route_space_certified=_implicit_route_space_certificate(r),
                             certificate_field_conflict=_certificate_field_conflict(r),
                             certificate_field_invalid=_certificate_field_invalid(r),
                             **_e1_certificate_provenance_fields(r),
                             model_contract_sha256=r.get("model_contract_sha256"),
                             parameter_contract_sha256=r.get("parameter_contract_sha256"),
                             instance_contract_sha256=r.get("instance_contract_sha256"),
                             algorithm_contract_sha256=r.get("algorithm_contract_sha256"),
                             resource_numeric_contract=r.get("resource_numeric_contract"),
                             pricing_complete=r.get("pricing_complete"),
                             pricing_bound_available=r.get("pricing_bound_available"),
                             resource_audit_complete=r.get("resource_audit_complete"),
                             branching_complete=r.get("branching_complete"),
                             farkas_pricing_complete=r.get("farkas_pricing_complete"),
                             bound_scope=r.get("bound_scope"),
                             bound_source=r.get("bound_source"),
                             solve_scope=r.get("solve_scope", "lexicographic"),
                             runtime_s=r.get("runtime_s"),
                             formal_open_nodes=r.get("open_nodes"),
                             formal_processed_nodes=r.get("processed_nodes"),
                             formal_rmp_solves=r.get("rmp_solves"),
                             formal_phase_one_solves=r.get("phase_one_solves"),
                             formal_resource_audit_calls=r.get("resource_audit_calls"),
                             formal_resource_cuts_added=r.get("resource_cuts_added"),
                             formal_pricing_candidates=r.get("pricing_candidates"),
                             formal_pricing_nodes=r.get("pricing_nodes"),
                             # pricing_mode is already carried by **prov below; keeping a\n                             # second explicit keyword makes dict(...) raise TypeError at row write.\n                             pricing_method=r.get("pricing_method"),
                             pricing_calls=r.get("pricing_calls"),
                             exact_pricing_calls=r.get("exact_pricing_calls"),
                             exact_certification_calls=r.get("exact_certification_calls"),
                             pricing_discovery_calls=r.get("pricing_discovery_calls"),
                             pricing_discovery_early_returns=r.get(
                                 "pricing_discovery_early_returns"),
                             pricing_discovery_improving_seen=r.get(
                                 "pricing_discovery_improving_seen"),
                             pricing_discovery_improving_returned=r.get(
                                 "pricing_discovery_improving_returned"),
                             pricing_discovery_diverse_returns=r.get(
                                 "pricing_discovery_diverse_returns"),
                             pricing_discovery_hard_cap_returns=r.get(
                                 "pricing_discovery_hard_cap_returns"),
                             pricing_discovery_max_return_batch=r.get(
                                 "pricing_discovery_max_return_batch"),
                             pricing_discovery_max_distinct_launches=r.get(
                                 "pricing_discovery_max_distinct_launches"),
                             pricing_discovery_max_distinct_service_sets=r.get(
                                 "pricing_discovery_max_distinct_service_sets"),
                             primal_refresh_calls=r.get(
                                 "primal_refresh_calls"),
                             primal_refresh_audit_calls=r.get(
                                 "primal_refresh_audit_calls"),
                             primal_refresh_timeouts=r.get(
                                 "primal_refresh_timeouts"),
                             primal_refresh_improvements=r.get(
                                 "primal_refresh_improvements"),
                             primal_refresh_best_coverage=r.get(
                                 "primal_refresh_best_coverage"),
                             primal_refresh_columns_seen=r.get(
                                 "primal_refresh_columns_seen"),
                             primal_refresh_rebuilds=r.get(
                                 "primal_refresh_rebuilds"),
                             primal_refresh_repairs=r.get(
                                 "primal_refresh_repairs"),
                             primal_refresh_augmentation_audits=r.get(
                                 "primal_refresh_augmentation_audits"),
                             primal_refresh_rebuild_audits=r.get(
                                 "primal_refresh_rebuild_audits"),
                             primal_refresh_repair_audits=r.get(
                                 "primal_refresh_repair_audits"),
                             primal_refresh_augmentation_improvements=r.get(
                                 "primal_refresh_augmentation_improvements"),
                             primal_refresh_rebuild_improvements=r.get(
                                 "primal_refresh_rebuild_improvements"),
                             primal_refresh_repair_improvements=r.get(
                                 "primal_refresh_repair_improvements"),
                             primal_refresh_duplicate_trials_skipped=r.get(
                                 "primal_refresh_duplicate_trials_skipped"),
                             primal_refresh_cached_infeasible_trials=r.get(
                                 "primal_refresh_cached_infeasible_trials"),
                             primal_refresh_uncovered_fair_rounds=r.get(
                                 "primal_refresh_uncovered_fair_rounds"),
                             primal_refresh_failure_reasons_json=json.dumps(
                                 r.get("primal_refresh_failure_reasons", {}),
                                 ensure_ascii=False, sort_keys=True),
                             primal_deck_diagnostic_enabled=r.get(
                                 "primal_deck_diagnostic_enabled"),
                             primal_deck_archive_conflict_edges=r.get(
                                 "primal_deck_archive_conflict_edges"),
                             primal_deck_archive_max_degree=r.get(
                                 "primal_deck_archive_max_degree"),
                             primal_deck_archive_max_component=r.get(
                                 "primal_deck_archive_max_component"),
                             primal_deck_candidate_scored=r.get(
                                 "primal_deck_candidate_scored"),
                             primal_deck_candidate_zero_conflict=r.get(
                                 "primal_deck_candidate_zero_conflict"),
                             primal_deck_candidate_positive_conflict=r.get(
                                 "primal_deck_candidate_positive_conflict"),
                             primal_deck_prefilter_skips=r.get(
                                 "primal_deck_prefilter_skips"),
                             primal_deck_max_candidate_conflicts=r.get(
                                 "primal_deck_max_candidate_conflicts"),
                             primal_deck_conflict_pairs_sample_json=json.dumps(
                                 r.get("primal_deck_conflict_pairs_sample", []),
                                 ensure_ascii=False, sort_keys=True),
                             pricing_multistop_merge_enabled=r.get(
                                 "pricing_multistop_merge_enabled"),
                             pricing_multistop_merge_triggers=r.get(
                                 "pricing_multistop_merge_triggers"),
                             pricing_multistop_merge_attempts=r.get(
                                 "pricing_multistop_merge_attempts"),
                             pricing_multistop_merge_physical_feasible=r.get(
                                 "pricing_multistop_merge_physical_feasible"),
                             pricing_multistop_merge_new_candidates=r.get(
                                 "pricing_multistop_merge_new_candidates"),
                             pricing_multistop_merge_returned=r.get(
                                 "pricing_multistop_merge_returned"),
                             pricing_multistop_merge_added=r.get(
                                 "pricing_multistop_merge_added"),
                             pricing_multistop_merge_batches=r.get(
                                 "pricing_multistop_merge_batches"),
                             pricing_multistop_merge_distinct_pairs=r.get(
                                 "pricing_multistop_merge_distinct_pairs"),
                             pricing_multistop_merge_best_rc_ub=r.get(
                                 "pricing_multistop_merge_best_rc_ub"),
                             pricing_multistop_merge_best_energy_per_stop_Wh=r.get(
                                 "pricing_multistop_merge_best_energy_per_stop_Wh"),
                             pricing_multistop_merge_best_uncovered_gain=r.get(
                                 "pricing_multistop_merge_best_uncovered_gain"),
                             pricing_multistop_merge_used_in_incumbent=r.get(
                                 "pricing_multistop_merge_used_in_incumbent"),
                             pricing_resource_variant_enabled=r.get(
                                 "pricing_resource_variant_enabled"),
                             pricing_resource_variant_triggers=r.get(
                                 "pricing_resource_variant_triggers"),
                             pricing_resource_variant_attempts=r.get(
                                 "pricing_resource_variant_attempts"),
                             pricing_resource_variant_deck_compatible_specs=r.get(
                                 "pricing_resource_variant_deck_compatible_specs"),
                             pricing_resource_variant_deck_prefilter_skips=r.get(
                                 "pricing_resource_variant_deck_prefilter_skips"),
                             pricing_resource_variant_physical_feasible=r.get(
                                 "pricing_resource_variant_physical_feasible"),
                             pricing_resource_variant_new_candidates=r.get(
                                 "pricing_resource_variant_new_candidates"),
                             pricing_resource_variant_returned=r.get(
                                 "pricing_resource_variant_returned"),
                             pricing_resource_variant_added=r.get(
                                 "pricing_resource_variant_added"),
                             pricing_resource_variant_batches=r.get(
                                 "pricing_resource_variant_batches"),
                             pricing_resource_variant_distinct_turbines=r.get(
                                 "pricing_resource_variant_distinct_turbines"),
                             pricing_resource_variant_best_rc_ub=r.get(
                                 "pricing_resource_variant_best_rc_ub"),
                             pricing_resource_variant_best_energy_Wh=r.get(
                                 "pricing_resource_variant_best_energy_Wh"),
                             pricing_resource_variant_used_in_incumbent=r.get(
                                 "pricing_resource_variant_used_in_incumbent"),
                             pricing_resource_variant_records_json=json.dumps(
                                 r.get("pricing_resource_variant_records", []),
                                 ensure_ascii=False, sort_keys=True),
                             archive_primal_recovery_enabled=r.get(
                                 "archive_primal_recovery_enabled"),
                             archive_primal_recovery_time_limit_s=r.get(
                                 "archive_primal_recovery_time_limit_s"),
                             archive_primal_recovery_calls=r.get(
                                 "archive_primal_recovery_calls"),
                             archive_primal_recovery_runtime_s=r.get(
                                 "archive_primal_recovery_runtime_s"),
                             archive_primal_recovery_audit_calls=r.get(
                                 "archive_primal_recovery_audit_calls"),
                             archive_primal_recovery_timeouts=r.get(
                                 "archive_primal_recovery_timeouts"),
                             archive_primal_recovery_improvements=r.get(
                                 "archive_primal_recovery_improvements"),
                             archive_primal_recovery_best_coverage=r.get(
                                 "archive_primal_recovery_best_coverage"),
                             archive_primal_recovery_best_archive_columns=r.get(
                                 "archive_primal_recovery_best_archive_columns"),
                             archive_primal_recovery_records_json=json.dumps(
                                 r.get("archive_primal_recovery_records", []),
                                 ensure_ascii=False, sort_keys=True),
                             archive_primal_recovery_witness_selection_indices_json=json.dumps(
                                 r.get("archive_primal_recovery_witness_selection_indices", []),
                                 ensure_ascii=False, sort_keys=True),
                             archive_primal_recovery_witness_route_signatures_json=json.dumps(
                                 r.get("archive_primal_recovery_witness_route_signatures", []),
                                 ensure_ascii=False, sort_keys=True),
                             archive_primal_recovery_witness_covered_turbines_json=json.dumps(
                                 r.get("archive_primal_recovery_witness_covered_turbines", []),
                                 ensure_ascii=False, sort_keys=True),
                             primal_exchange_enabled=r.get(
                                 "primal_exchange_enabled"),
                             primal_exchange_calls=r.get(
                                 "primal_exchange_calls"),
                             primal_exchange_candidate_routes=r.get(
                                 "primal_exchange_candidate_routes"),
                             primal_exchange_trials_built=r.get(
                                 "primal_exchange_trials_built"),
                             primal_exchange_audit_calls=r.get(
                                 "primal_exchange_audit_calls"),
                             primal_exchange_improvements=r.get(
                                 "primal_exchange_improvements"),
                             primal_exchange_consolidation_trials=r.get(
                                 "primal_exchange_consolidation_trials"),
                             primal_exchange_optional_drop_trials=r.get(
                                 "primal_exchange_optional_drop_trials"),
                             primal_exchange_max_stop_count_considered=r.get(
                                 "primal_exchange_max_stop_count_considered"),
                             primal_exchange_best_coverage=r.get(
                                 "primal_exchange_best_coverage"),
                             primal_exchange_multistop_used_in_incumbent=r.get(
                                 "primal_exchange_multistop_used_in_incumbent"),
                             pricing_depth1_prefixes=r.get("pricing_depth1_prefixes"),
                             pricing_depth2_prefixes=r.get("pricing_depth2_prefixes"),
                             pricing_depth3_prefixes=r.get("pricing_depth3_prefixes"),
                             pricing_depth4_prefixes=r.get("pricing_depth4_prefixes"),
                             pricing_depth1_improving=r.get("pricing_depth1_improving"),
                             pricing_depth2_improving=r.get("pricing_depth2_improving"),
                             pricing_depth3_improving=r.get("pricing_depth3_improving"),
                             pricing_depth4_improving=r.get("pricing_depth4_improving"),
                             pricing_depth1_returned=r.get("pricing_depth1_returned"),
                             pricing_depth2_returned=r.get("pricing_depth2_returned"),
                             pricing_depth3_returned=r.get("pricing_depth3_returned"),
                             pricing_depth4_returned=r.get("pricing_depth4_returned"),
                             pricing_depth_prefixes_json=json.dumps(
                                 r.get("pricing_depth_prefixes_evaluated", {}),
                                 ensure_ascii=False, sort_keys=True),
                             pricing_depth_improving_json=json.dumps(
                                 r.get("pricing_depth_improving_seen", {}),
                                 ensure_ascii=False, sort_keys=True),
                             pricing_depth_returned_json=json.dumps(
                                 r.get("pricing_depth_improving_returned", {}),
                                 ensure_ascii=False, sort_keys=True),
                             pricing_pattern_cut_active_dual_rows=r.get(
                                 "pricing_pattern_cut_active_dual_rows"),
                             pricing_pattern_cut_dual_abs_sum=r.get(
                                 "pricing_pattern_cut_dual_abs_sum"),
                             pricing_pattern_cut_improving_seen_count=r.get(
                                 "pricing_pattern_cut_improving_seen_count"),
                             pricing_pattern_cut_improving_seen_contribution_sum=r.get(
                                 "pricing_pattern_cut_improving_seen_contribution_sum"),
                             pricing_pattern_cut_improving_seen_sign_essential=r.get(
                                 "pricing_pattern_cut_improving_seen_sign_essential"),
                             pricing_pattern_cut_returned_count=r.get(
                                 "pricing_pattern_cut_returned_count"),
                             pricing_pattern_cut_returned_contribution_sum=r.get(
                                 "pricing_pattern_cut_returned_contribution_sum"),
                             pricing_pattern_cut_returned_sign_essential=r.get(
                                 "pricing_pattern_cut_returned_sign_essential"),
                             pricing_pattern_cut_returned_by_depth_json=json.dumps(
                                 r.get("pricing_pattern_cut_returned_by_depth", {}),
                                 ensure_ascii=False, sort_keys=True),
                             battery_halfcap_formal_enabled=r.get(
                                 "battery_halfcap_formal_enabled"),
                             battery_halfcap_usable_capacity_Wh=r.get(
                                 "battery_halfcap_usable_capacity_Wh"),
                             battery_halfcap_rhs=r.get("battery_halfcap_rhs"),
                             battery_halfcap_archive_route_count=r.get(
                                 "battery_halfcap_archive_route_count"),
                             battery_halfcap_archive_high_energy_routes=r.get(
                                 "battery_halfcap_archive_high_energy_routes"),
                             battery_halfcap_archive_low_energy_routes=r.get(
                                 "battery_halfcap_archive_low_energy_routes"),
                             battery_halfcap_dual_active_rmp_solves=r.get(
                                 "battery_halfcap_dual_active_rmp_solves"),
                             battery_halfcap_dual_abs_sum=r.get(
                                 "battery_halfcap_dual_abs_sum"),
                             battery_halfcap_dual_max_abs=r.get(
                                 "battery_halfcap_dual_max_abs"),
                             coverage_battery_halfcap_dual_active_rmp_solves=r.get(
                                 "coverage_battery_halfcap_dual_active_rmp_solves"),
                             coverage_battery_halfcap_dual_abs_sum=r.get(
                                 "coverage_battery_halfcap_dual_abs_sum"),
                             coverage_battery_halfcap_dual_max_abs=r.get(
                                 "coverage_battery_halfcap_dual_max_abs"),
                             energy_battery_halfcap_dual_active_rmp_solves=r.get(
                                 "energy_battery_halfcap_dual_active_rmp_solves"),
                             energy_battery_halfcap_dual_abs_sum=r.get(
                                 "energy_battery_halfcap_dual_abs_sum"),
                             energy_battery_halfcap_dual_max_abs=r.get(
                                 "energy_battery_halfcap_dual_max_abs"),
                             archive_diag_enabled=r.get("archive_diag_enabled"),
                             archive_diag_scope=r.get("archive_diag_scope"),
                             archive_diag_status=r.get("archive_diag_status"),
                             archive_diag_time_limit_s=r.get("archive_diag_time_limit_s"),
                             archive_diag_runtime_s=r.get("archive_diag_runtime_s"),
                             archive_diag_columns=r.get("archive_diag_columns"),
                             archive_diag_coverage_lower_bound=r.get(
                                 "archive_diag_coverage_lower_bound"),
                             archive_diag_coverage_upper_bound=r.get(
                                 "archive_diag_coverage_upper_bound"),
                             archive_diag_exact_optimum=r.get(
                                 "archive_diag_exact_optimum"),
                             archive_diag_optimal_proven=r.get(
                                 "archive_diag_optimal_proven"),
                             archive_diag_open_nodes=r.get("archive_diag_open_nodes"),
                             archive_diag_processed_nodes=r.get(
                                 "archive_diag_processed_nodes"),
                             archive_diag_rmp_solves=r.get("archive_diag_rmp_solves"),
                             archive_diag_resource_audit_calls=r.get(
                                 "archive_diag_resource_audit_calls"),
                             archive_diag_resource_cuts_added=r.get(
                                 "archive_diag_resource_cuts_added"),
                             archive_diag_witness_selection_indices_json=json.dumps(
                                 r.get("archive_diag_witness_selection_indices", []),
                                 ensure_ascii=False, sort_keys=True),
                             archive_diag_witness_route_signatures_json=json.dumps(
                                 r.get("archive_diag_witness_route_signatures", []),
                                 ensure_ascii=False, sort_keys=True),
                             archive_diag_witness_covered_turbines_json=json.dumps(
                                 r.get("archive_diag_witness_covered_turbines", []),
                                 ensure_ascii=False, sort_keys=True),
                             resource_variant_diag_enabled=r.get(
                                 "resource_variant_diag_enabled"),
                             resource_variant_diag_scope=r.get(
                                 "resource_variant_diag_scope"),
                             resource_variant_diag_status=r.get(
                                 "resource_variant_diag_status"),
                             resource_variant_diag_time_limit_s=r.get(
                                 "resource_variant_diag_time_limit_s"),
                             resource_variant_diag_runtime_s=r.get(
                                 "resource_variant_diag_runtime_s"),
                             resource_variant_diag_timed_out=r.get(
                                 "resource_variant_diag_timed_out"),
                             resource_variant_diag_records_analyzed=r.get(
                                 "resource_variant_diag_records_analyzed"),
                             resource_variant_diag_records_missing_from_archive=r.get(
                                 "resource_variant_diag_records_missing_from_archive"),
                             resource_variant_diag_final_coverage=r.get(
                                 "resource_variant_diag_final_coverage"),
                             resource_variant_diag_final_uncovered_turbines_json=json.dumps(
                                 r.get("resource_variant_diag_final_uncovered_turbines", []),
                                 ensure_ascii=False, sort_keys=True),
                             resource_variant_diag_direct_augmentation_audits=r.get(
                                 "resource_variant_diag_direct_augmentation_audits"),
                             resource_variant_diag_direct_augmentation_feasible=r.get(
                                 "resource_variant_diag_direct_augmentation_feasible"),
                             resource_variant_diag_direct_augmentation_infeasible=r.get(
                                 "resource_variant_diag_direct_augmentation_infeasible"),
                             resource_variant_diag_direct_augmentation_unknown=r.get(
                                 "resource_variant_diag_direct_augmentation_unknown"),
                             resource_variant_diag_single_blocker_records=r.get(
                                 "resource_variant_diag_single_blocker_records"),
                             resource_variant_diag_blocker_retime_candidates=r.get(
                                 "resource_variant_diag_blocker_retime_candidates"),
                             resource_variant_diag_blocker_retime_audits=r.get(
                                 "resource_variant_diag_blocker_retime_audits"),
                             resource_variant_diag_blocker_retime_feasible=r.get(
                                 "resource_variant_diag_blocker_retime_feasible"),
                             resource_variant_diag_final_uncovered_singleton_routes=r.get(
                                 "resource_variant_diag_final_uncovered_singleton_routes"),
                             resource_variant_diag_final_uncovered_zero_deck_conflict_routes=r.get(
                                 "resource_variant_diag_final_uncovered_zero_deck_conflict_routes"),
                             resource_variant_diag_final_uncovered_single_blocker_routes=r.get(
                                 "resource_variant_diag_final_uncovered_single_blocker_routes"),
                             resource_variant_diag_final_uncovered_multi_blocker_routes=r.get(
                                 "resource_variant_diag_final_uncovered_multi_blocker_routes"),
                             resource_variant_diag_final_uncovered_single_blocker_distinct_turbines=r.get(
                                 "resource_variant_diag_final_uncovered_single_blocker_distinct_turbines"),
                             resource_variant_diag_records_json=json.dumps(
                                 r.get("resource_variant_diag_records", []),
                                 ensure_ascii=False, sort_keys=True),
                             resource_variant_diag_single_blocker_pairs_sample_json=json.dumps(
                                 r.get("resource_variant_diag_single_blocker_pairs_sample", []),
                                 ensure_ascii=False, sort_keys=True),
                             fullspace_target_diag_enabled=r.get(
                                 "fullspace_target_diag_enabled"),
                             fullspace_target_diag_scope=r.get(
                                 "fullspace_target_diag_scope"),
                             fullspace_target_diag_status=r.get(
                                 "fullspace_target_diag_status"),
                             fullspace_target_diag_time_limit_s=r.get(
                                 "fullspace_target_diag_time_limit_s"),
                             fullspace_target_diag_runtime_s=r.get(
                                 "fullspace_target_diag_runtime_s"),
                             fullspace_target_diag_archive_columns_start=r.get(
                                 "fullspace_target_diag_archive_columns_start"),
                             fullspace_target_diag_archive_columns_end=r.get(
                                 "fullspace_target_diag_archive_columns_end"),
                             fullspace_target_diag_start_coverage=r.get(
                                 "fullspace_target_diag_start_coverage"),
                             fullspace_target_diag_best_coverage=r.get(
                                 "fullspace_target_diag_best_coverage"),
                             fullspace_target_diag_highest_feasible_target=r.get(
                                 "fullspace_target_diag_highest_feasible_target"),
                             fullspace_target_diag_first_infeasible_target=r.get(
                                 "fullspace_target_diag_first_infeasible_target"),
                             fullspace_target_diag_unresolved_target=r.get(
                                 "fullspace_target_diag_unresolved_target"),
                             fullspace_target_diag_targets_attempted=r.get(
                                 "fullspace_target_diag_targets_attempted"),
                             fullspace_target_diag_records_json=json.dumps(
                                 r.get("fullspace_target_diag_records", []),
                                 ensure_ascii=False, sort_keys=True),
                             fullspace_target_diag_witness_selection_indices_json=json.dumps(
                                 r.get("fullspace_target_diag_witness_selection_indices", []),
                                 ensure_ascii=False, sort_keys=True),
                             fullspace_target_diag_witness_route_signatures_json=json.dumps(
                                 r.get("fullspace_target_diag_witness_route_signatures", []),
                                 ensure_ascii=False, sort_keys=True),
                             fullspace_target_diag_witness_covered_turbines_json=json.dumps(
                                 r.get("fullspace_target_diag_witness_covered_turbines", []),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_enabled=r.get(
                                 "archive_target_enabled"),
                             archive_target_scope=r.get(
                                 "archive_target_scope"),
                             archive_target_min=r.get(
                                 "archive_target_min"),
                             archive_target_status=r.get(
                                 "archive_target_status"),
                             archive_target_time_limit_s=r.get(
                                 "archive_target_time_limit_s"),
                             archive_target_runtime_s=r.get(
                                 "archive_target_runtime_s"),
                             archive_target_columns=r.get(
                                 "archive_target_columns"),
                             archive_target_feasible_proven=r.get(
                                 "archive_target_feasible_proven"),
                             archive_target_infeasible_proven=r.get(
                                 "archive_target_infeasible_proven"),
                             archive_target_witness_coverage=r.get(
                                 "archive_target_witness_coverage"),
                             archive_target_coverage_incumbent=r.get(
                                 "archive_target_coverage_incumbent"),
                             archive_target_coverage_upper_bound=r.get(
                                 "archive_target_coverage_upper_bound"),
                             archive_target_open_nodes=r.get(
                                 "archive_target_open_nodes"),
                             archive_target_processed_nodes=r.get(
                                 "archive_target_processed_nodes"),
                             archive_target_rmp_solves=r.get(
                                 "archive_target_rmp_solves"),
                             archive_target_resource_audit_calls=r.get(
                                 "archive_target_resource_audit_calls"),
                             archive_target_resource_cuts_added=r.get(
                                 "archive_target_resource_cuts_added"),
                             archive_target_rejected_pattern_count=r.get(
                                 "archive_target_rejected_pattern_count"),
                             archive_target_rejected_pattern_size_avg=r.get(
                                 "archive_target_rejected_pattern_size_avg"),
                             archive_target_rejected_pattern_size_min=r.get(
                                 "archive_target_rejected_pattern_size_min"),
                             archive_target_rejected_pattern_size_max=r.get(
                                 "archive_target_rejected_pattern_size_max"),
                             archive_target_rejected_pattern_coverage_avg=r.get(
                                 "archive_target_rejected_pattern_coverage_avg"),
                             archive_target_rejected_pattern_coverage_min=r.get(
                                 "archive_target_rejected_pattern_coverage_min"),
                             archive_target_rejected_pattern_coverage_max=r.get(
                                 "archive_target_rejected_pattern_coverage_max"),
                             archive_target_rejected_hamming_avg=r.get(
                                 "archive_target_rejected_hamming_avg"),
                             archive_target_rejected_hamming_min=r.get(
                                 "archive_target_rejected_hamming_min"),
                             archive_target_rejected_hamming_max=r.get(
                                 "archive_target_rejected_hamming_max"),
                             archive_target_resource_failure_event_counts_json=json.dumps(
                                 r.get("archive_target_resource_failure_event_counts", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_resource_failure_pattern_counts_json=json.dumps(
                                 r.get("archive_target_resource_failure_pattern_counts", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_rejected_morphology_counts_json=json.dumps(
                                 r.get("archive_target_rejected_morphology_counts", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_rejected_route_stop_count_totals_json=json.dumps(
                                 r.get("archive_target_rejected_route_stop_count_totals", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_rejected_coverage_route_count_joint_json=json.dumps(
                                 r.get("archive_target_rejected_coverage_route_count_joint", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_certificate_shadow_json=json.dumps(
                                 r.get("archive_target_certificate_shadow", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_shadow_analyzed_patterns=r.get(
                                 "archive_target_shadow_analyzed_patterns"),
                             archive_target_shadow_total_patterns=r.get(
                                 "archive_target_shadow_total_patterns"),
                             archive_target_shadow_timed_out=r.get(
                                 "archive_target_shadow_timed_out"),
                             archive_target_shadow_pooled_energy_infeasible=r.get(
                                 "archive_target_shadow_pooled_energy_infeasible"),
                             archive_target_shadow_battery_binpack_infeasible=r.get(
                                 "archive_target_shadow_battery_binpack_infeasible"),
                             archive_target_shadow_battery_binpack_feasible=r.get(
                                 "archive_target_shadow_battery_binpack_feasible"),
                             archive_target_shadow_battery_binpack_unknown=r.get(
                                 "archive_target_shadow_battery_binpack_unknown"),
                             archive_target_shadow_battery_core_unique_count=r.get(
                                 "archive_target_shadow_battery_core_unique_count"),
                             archive_target_shadow_battery_core_size_avg=r.get(
                                 "archive_target_shadow_battery_core_size_avg"),
                             archive_target_shadow_battery_core_size_min=r.get(
                                 "archive_target_shadow_battery_core_size_min"),
                             archive_target_shadow_battery_core_size_max=r.get(
                                 "archive_target_shadow_battery_core_size_max"),
                             archive_target_shadow_prior_core_cover_count=r.get(
                                 "archive_target_shadow_prior_core_cover_count"),
                             archive_target_shadow_prior_core_cover_fraction=r.get(
                                 "archive_target_shadow_prior_core_cover_fraction"),
                             archive_target_shadow_fastest_turnaround_infeasible=r.get(
                                 "archive_target_shadow_fastest_turnaround_infeasible"),
                             archive_target_shadow_battery_min_required_counts_json=json.dumps(
                                 r.get("archive_target_shadow_battery_min_required_counts", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_shadow_first_proof_layer_counts_json=json.dumps(
                                 r.get("archive_target_shadow_first_proof_layer_counts", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_battery_clique_shadow_json=json.dumps(
                                 r.get("archive_target_battery_clique_shadow", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_clique_halfcap_rows=r.get(
                                 "archive_target_clique_halfcap_rows"),
                             archive_target_clique_anchor_rows=r.get(
                                 "archive_target_clique_anchor_rows"),
                             archive_target_clique_total_rows=r.get(
                                 "archive_target_clique_total_rows"),
                             archive_target_clique_archive_halfcap_routes=r.get(
                                 "archive_target_clique_archive_halfcap_routes"),
                             archive_target_clique_archive_nonhalfcap_routes=r.get(
                                 "archive_target_clique_archive_nonhalfcap_routes"),
                             archive_target_clique_archive_halfcap_stop_count_counts_json=json.dumps(
                                 r.get("archive_target_clique_archive_halfcap_stop_count_counts", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_target_clique_rejected_halfcap_violations=r.get(
                                 "archive_target_clique_rejected_halfcap_violations"),
                             archive_target_clique_rejected_anchor_violations=r.get(
                                 "archive_target_clique_rejected_anchor_violations"),
                             archive_target_clique_rejected_anchor_only_violations=r.get(
                                 "archive_target_clique_rejected_anchor_only_violations"),
                             archive_target_clique_rejected_any_violations=r.get(
                                 "archive_target_clique_rejected_any_violations"),
                             archive_target_clique_rejected_uncovered=r.get(
                                 "archive_target_clique_rejected_uncovered"),
                             archive_target_clique_rejected_halfcap_count_distribution_json=json.dumps(
                                 r.get("archive_target_clique_rejected_halfcap_count_distribution", {}),
                                 ensure_ascii=False, sort_keys=True),
                             archive_clique_target_enabled=r.get(
                                 "archive_clique_target_enabled"),
                             archive_clique_target_scope=r.get(
                                 "archive_clique_target_scope"),
                             archive_clique_target_status=r.get(
                                 "archive_clique_target_status"),
                             archive_clique_target_time_limit_s=r.get(
                                 "archive_clique_target_time_limit_s"),
                             archive_clique_target_runtime_s=r.get(
                                 "archive_clique_target_runtime_s"),
                             archive_clique_target_rows=r.get(
                                 "archive_clique_target_rows"),
                             archive_clique_target_feasible_proven=r.get(
                                 "archive_clique_target_feasible_proven"),
                             archive_clique_target_infeasible_proven=r.get(
                                 "archive_clique_target_infeasible_proven"),
                             archive_clique_target_witness_coverage=r.get(
                                 "archive_clique_target_witness_coverage"),
                             archive_clique_target_coverage_incumbent=r.get(
                                 "archive_clique_target_coverage_incumbent"),
                             archive_clique_target_coverage_upper_bound=r.get(
                                 "archive_clique_target_coverage_upper_bound"),
                             archive_clique_target_open_nodes=r.get(
                                 "archive_clique_target_open_nodes"),
                             archive_clique_target_processed_nodes=r.get(
                                 "archive_clique_target_processed_nodes"),
                             archive_clique_target_rmp_solves=r.get(
                                 "archive_clique_target_rmp_solves"),
                             archive_clique_target_resource_audit_calls=r.get(
                                 "archive_clique_target_resource_audit_calls"),
                             archive_clique_target_resource_cuts_added=r.get(
                                 "archive_clique_target_resource_cuts_added"),
                             archive_clique_target_rejected_pattern_count=r.get(
                                 "archive_clique_target_rejected_pattern_count"),
                             diagnostic_runtime_s=r.get("diagnostic_runtime_s"),
                             total_wall_runtime_s=r.get("total_wall_runtime_s"),
                             pricing_shadow_prefixes_evaluated=r.get(
                                 "pricing_shadow_prefixes_evaluated"),
                             pricing_shadow_prunable_prefixes=r.get(
                                 "pricing_shadow_prunable_prefixes"),
                             pricing_shadow_false_prune_witnesses=r.get(
                                 "pricing_shadow_false_prune_witnesses"),
                             pricing_shadow_bound_errors=r.get(
                                 "pricing_shadow_bound_errors"),
                             pricing_shadow_complete_calls=r.get(
                                 "pricing_shadow_complete_calls"),
                             pricing_guided_order_calls=r.get(
                                 "pricing_guided_order_calls"),
                             pricing_guided_order_reorders=r.get(
                                 "pricing_guided_order_reorders"),
                             pricing_guided_order_failures=r.get(
                                 "pricing_guided_order_failures"),
                             pricing_layered_depths_started=r.get(
                                 "pricing_layered_depths_started"),
                             pricing_layered_depths_completed=r.get(
                                 "pricing_layered_depths_completed"),
                             pricing_layered_max_depth_completed=r.get(
                                 "pricing_layered_max_depth_completed"),
                             pricing_layered_rounds=r.get(
                                 "pricing_layered_rounds"),
                             pricing_depth_fair_requested_calls=r.get(
                                 "pricing_depth_fair_requested_calls"),
                             pricing_depth_fair_active_calls=r.get(
                                 "pricing_depth_fair_active_calls"),
                             pricing_depth_fair_rounds=r.get(
                                 "pricing_depth_fair_rounds"),
                             pricing_depth_fair_halfcap_dual_abs_sum=r.get(
                                 "pricing_depth_fair_halfcap_dual_abs_sum"),
                             pricing_multistop_neutral_enabled_calls=r.get(
                                 "pricing_multistop_neutral_enabled_calls"),
                             pricing_multistop_candidates_seen=r.get(
                                 "pricing_multistop_candidates_seen"),
                             pricing_multistop_physical_feasible=r.get(
                                 "pricing_multistop_physical_feasible"),
                             pricing_multistop_cross_zero_seen=r.get(
                                 "pricing_multistop_cross_zero_seen"),
                             pricing_multistop_nonnegative_seen=r.get(
                                 "pricing_multistop_nonnegative_seen"),
                             pricing_multistop_neutral_returned=r.get(
                                 "pricing_multistop_neutral_returned"),
                             pricing_multistop_neutral_added=r.get(
                                 "pricing_multistop_neutral_added"),
                             pricing_multistop_neutral_batches=r.get(
                                 "pricing_multistop_neutral_batches"),
                             pricing_multistop_depth2_neutral=r.get(
                                 "pricing_multistop_depth2_neutral"),
                             pricing_multistop_depth3_neutral=r.get(
                                 "pricing_multistop_depth3_neutral"),
                             pricing_multistop_depth4_neutral=r.get(
                                 "pricing_multistop_depth4_neutral"),
                             pricing_multistop_best_stop_count=r.get(
                                 "pricing_multistop_best_stop_count"),
                             pricing_multistop_best_uncovered_gain=r.get(
                                 "pricing_multistop_best_uncovered_gain"),
                             pricing_multistop_best_rc_ub=r.get(
                                 "pricing_multistop_best_rc_ub"),
                             pricing_multistop_best_energy_per_stop_Wh=r.get(
                                 "pricing_multistop_best_energy_per_stop_Wh"),
                             pricing_multistop_neutral_used_in_incumbent=r.get(
                                 "pricing_multistop_neutral_used_in_incumbent"),
                             pricing_multistop_neutral_returned_by_depth_json=json.dumps(
                                 r.get("pricing_multistop_neutral_returned_by_depth", {}),
                                 ensure_ascii=False, sort_keys=True),
                             pricing_physical_cache_hits=r.get(
                                 "pricing_physical_cache_hits"),
                             pricing_physical_cache_misses=r.get(
                                 "pricing_physical_cache_misses"),
                             physical_pricing_cache_entries_before_pricing=r.get(
                                 "physical_pricing_cache_entries_before_pricing"),
                             pricing_candidates=r.get("pricing_candidates"),
                             pricing_nodes=r.get("pricing_nodes"),
                             coverage_global_certificate_available=_coverage_certificate_flag(r),
                             coverage_physical_model_certificate=r.get("coverage_physical_model_certificate"),
                             coverage_algorithmic_certificate=r.get("coverage_algorithmic_certificate"),
                             frontier_evaluated=True,
                             frontier_coverage_certified=_coverage_certificate_flag(r),
                             frontier_lexicographic_certified=_global_certificate_flag(r),
                             frontier_completion_state=(
                                 "lexicographic-certified" if _global_certificate_flag(r) else
                                 "coverage-certified" if _coverage_certificate_flag(r) else
                                 "anytime-bounds-only"),
                             **prov)
            key = (str(uk), int(K), int(B))
            if key in row_index:
                rows[row_index[key]] = record
            else:
                row_index[key] = len(rows)
                rows.append(record)
            vals[(K, B)] = len(safe)
            chosen_at[(K, B)] = r["chosen"]
            if formal_ondemand:
                _remember_seed_columns(r.get("chosen", []))
            if full_lex and _global_certificate_flag(r):
                certified_knee_results[(str(uk), int(K), int(B))] = r
            _done.add(checkpoint_key)
            _save(pd.DataFrame(rows), outdir, "E1_frontier.csv")
            return len(safe)

        for K in ks:
            for B in bs_base:
                _cell(K, B)
        # ---- Formal B-axis saturation: prove it from optimization coverage bounds. ----
        # Resource monotonicity applies to C*(K,B), not to post-validation ``safe_served``.
        # At fixed Kmax, if B0 < B1 and
        #     C_inc(Kmax,B0) == UB_C(Kmax,B1) == P,
        # then C_inc(B0) <= C*(B0) <= C*(B) <= C*(B1) <= UB(B1)
        # forces every intermediate exact coverage to equal P.  This certifies
        # ``patience`` zero increments without requiring every cell to close.
        b_run = list(bs_base)

        def _row_for(K, B):
            idx = row_index.get((str(uk), int(K), int(B)))
            return None if idx is None else rows[idx]

        def _coverage_interval(K, B):
            rec = _row_for(K, B)
            if rec is None:
                return None
            try:
                lb = int(rec.get("coverage_incumbent", rec.get("covered", 0)) or 0)
                ub = int(rec.get("coverage_upper_bound", rec.get("coverable_note", len(reach)))
                         if rec.get("coverage_upper_bound", None) is not None
                         else rec.get("coverable_note", len(reach)))
            except (TypeError, ValueError, OverflowError):
                return None
            if lb < 0 or ub < lb:
                return None
            return lb, ub

        def _monotone_coverage_interval(K, B):
            dsub = pd.DataFrame([r for r in rows if str(r.get("uav")) == str(uk)])
            return _e1_monotone_coverage_interval(dsub, K, B)

        def _formal_saturation_certificate(Kx, seq, patience):
            seq = sorted(set(int(b) for b in seq))
            if not seq:
                return False, None, "no-cells"
            end_b = seq[-1]
            end_iv = _monotone_coverage_interval(Kx, end_b)
            if end_iv is None:
                return False, None, "missing-end-bound"
            end_lb, end_ub = end_iv
            end_row = _row_for(Kx, end_b) or {}
            hard_cap = int(end_row.get("coverable_note", len(reach)) or len(reach))
            if end_lb >= hard_cap:
                return True, hard_cap, "hard-coverable-cap"
            if len(seq) < int(patience) + 1:
                return False, None, "insufficient-tail"
            start_b = seq[-int(patience) - 1]
            start_iv = _monotone_coverage_interval(Kx, start_b)
            if start_iv is None:
                return False, None, "missing-start-bound"
            start_lb, _start_ub = start_iv
            if start_lb == end_ub:
                return True, int(start_lb), f"monotone-sandwich:{start_b}->{end_b}"
            return False, None, f"open-sandwich:{start_lb}..{end_ub}"

        if str(args.e1_b_auto).lower() == "on" and (formal_ondemand or cols):
            Kx, pat = max(ks), max(1, int(args.e1_sat_patience))
            # Structural exact cap: every selected route is nonempty and turbine
            # packing is <=1, hence an integer plan selects at most |I| routes.
            # With B=|I| each selected route can receive a distinct battery, so
            # additional battery groups cannot enlarge the feasible set.
            battery_axis_cap = min(int(args.e1_b_cap), int(len(reach)))
            if int(args.e1_b_cap) > battery_axis_cap:
                log.info(
                    "E1[%s]: formal battery轴结构上界 B<=|I|=%d；"
                    "--e1-b-cap=%d 的更大值冗余，不再求解 B>%d。",
                    uk, len(reach), int(args.e1_b_cap), battery_axis_cap)
            while max(b_run) < battery_axis_cap:
                seq = sorted(b_run)
                if formal_ondemand and str(getattr(args, "study_mode", "formal")) == "formal":
                    sat_ok, sat_value, sat_proof = _formal_saturation_certificate(Kx, seq, pat)
                    if sat_ok:
                        log.info("E1[%s]: formal B轴覆盖平台已证明 P=%s (%s), Kmax=%d; 停止延伸",
                                 uk, sat_value, sat_proof, Kx)
                        break
                else:
                    gains = [vals[(Kx, b2)] - vals[(Kx, b1)]
                             for b1, b2 in zip(seq, seq[1:])]
                    if len(gains) >= pat and all(g <= 0 for g in gains[-pat:]):
                        log.info("E1[%s]: B轴在 B=%d 饱和(Kmax=%d 连续%d次观测边际=0), 停止延伸",
                                 uk, max(b_run), Kx, pat)
                        break
                nb = max(b_run) + 1
                b_run.append(nb)
                for K in ks:
                    _cell(K, nb)
            if max(b_run) >= battery_axis_cap:
                if formal_ondemand and str(getattr(args, "study_mode", "formal")) == "formal":
                    sat_ok, sat_value, sat_proof = _formal_saturation_certificate(Kx, b_run, pat)
                    if not sat_ok:
                        log.warning(
                            "E1[%s]: B轴达到结构上界 B=%d (min(cli_cap=%d,|I|=%d))，"
                            "但覆盖平台仍未严格证明(%s)。无需再增加电池；"
                            "应收紧 coverage upper bound / pricing certificate。",
                            uk, battery_axis_cap, int(args.e1_b_cap), len(reach), sat_proof)
                else:
                    seq = sorted(b_run)
                    tail = [vals[(Kx, b2)] - vals[(Kx, b1)]
                            for b1, b2 in zip(seq, seq[1:])][-pat:]
                    if any(g > 0 for g in tail):
                        log.warning("E1[%s]: B触顶 --e1-b-cap=%d 仍未饱和(尾部边际=%s)",
                                    uk, args.e1_b_cap, tail)
        elif str(args.e1_b_auto).lower() == "on" and not cols:
            log.error("E1[%s]: 研究共享路线列池为空，跳过 B 轴饱和判定。", uk)
        # ---- Formal adaptive refinement: long-clock only on proof-blocking cells. ----
        if formal_ondemand and str(getattr(args, "study_mode", "formal")) == "formal":
            def _uav_selection_row():
                dsub = pd.DataFrame([r for r in rows if str(r.get("uav")) == str(uk)])
                if not len(dsub):
                    return None
                ss = e1_select_from_df(
                    dsub, frac=getattr(args, "knee_frac", 0.95),
                    order=getattr(args, "knee_order", "BK"),
                    patience=max(1, int(args.e1_sat_patience)))
                return (None if not len(ss) else ss.iloc[0])

            refinement_no_gain = set()

            def _refinement_keys(selrow):
                if selrow is None:
                    return []
                dsub = pd.DataFrame([r for r in rows if str(r.get("uav")) == str(uk)])
                Kx = int(dsub["K"].max())
                status = str(selrow.get("selection_status", ""))
                keys = []

                def iv(rr):
                    try:
                        return _monotone_coverage_interval(
                            int(rr["K"]), int(rr["batteries"]))
                    except (KeyError, TypeError, ValueError, OverflowError):
                        return None

                if status == "uncertified_coverage_plateau":
                    # Explicit fixed-B means no automatic resource-axis extension.
                    # Long endpoint solves are therefore skipped; the unresolved
                    # status is preserved until the user enables B extension.
                    if not _e1_plateau_long_refinement_allowed(args.e1_b_auto):
                        return []

                    cur = dsub[dsub["K"] == Kx].sort_values("batteries")
                    pat = max(1, int(args.e1_sat_patience))
                    if not len(cur):
                        return []
                    end = cur.iloc[-1]
                    end_key = (Kx, int(end["batteries"]))
                    end_iv = iv(end)
                    start = cur.iloc[-pat-1] if len(cur) >= pat + 1 else None
                    start_key = (Kx, int(start["batteries"])) if start is not None else None
                    start_iv = iv(start) if start is not None else None

                    candidates = []
                    if end_iv is not None and end_iv[0] < end_iv[1]:
                        candidates.append((end_iv[1] - end_iv[0], end_key))
                    if start_iv is not None and start_iv[0] < start_iv[1]:
                        candidates.append((start_iv[1] - start_iv[0], start_key))
                    candidates.sort(key=lambda z: (z[0], z[1]))
                    for _width, key in candidates:
                        rec = _row_for(*key) or {}
                        persisted_no_gain = bool(rec.get("coverage_refinement_no_gain", False))
                        if key not in refinement_no_gain and not persisted_no_gain:
                            return [key]
                    return []

                if status != "uncertified_resource_knee":
                    return []
                # Once hard-coverable-cap fixes P and therefore T, generic
                # max-coverage refinement answers a weaker question than the
                # exact predecessor target decision and caused repeated no-gain
                # long solves in v11.  Formal v12 routes this state directly to
                # E1_knee_refine instead of retrying coverage optimization.
                if not _formal_resource_knee_generic_refinement_allowed(selrow):
                    return []
                T = selrow.get("coverage_threshold")
                if T is None or pd.isna(T):
                    return []
                T = int(T)
                order_key = str(getattr(args, "knee_order", "BK")).upper()
                if order_key == "BK":
                    cur = dsub[dsub["K"] == Kx].sort_values("batteries")
                    possible = feasible = None
                    for _, rr in cur.iterrows():
                        q = iv(rr)
                        if q is None:
                            continue
                        b = int(rr["batteries"])
                        if possible is None and q[1] >= T:
                            possible = b
                        if feasible is None and q[0] >= T:
                            feasible = b
                    if possible is not None and (feasible is None or possible < feasible):
                        keys.append((Kx, possible))
                        return keys
                    if feasible is not None:
                        prev = cur[cur["batteries"] < feasible].tail(1)
                        if len(prev) and iv(prev.iloc[0]) is not None and iv(prev.iloc[0])[1] >= T:
                            keys.append((Kx, int(prev.iloc[0]["batteries"])))
                        atb = dsub[dsub["batteries"] == feasible].sort_values("K")
                        k_possible = k_feasible = None
                        for _, rr in atb.iterrows():
                            q = iv(rr)
                            if q is None:
                                continue
                            k = int(rr["K"])
                            if k_possible is None and q[1] >= T:
                                k_possible = k
                            if k_feasible is None and q[0] >= T:
                                k_feasible = k
                        if k_possible is not None and (k_feasible is None or k_possible < k_feasible):
                            keys.append((k_possible, feasible))
                        elif k_feasible is not None:
                            prevk = atb[atb["K"] < k_feasible].tail(1)
                            if len(prevk) and iv(prevk.iloc[0]) is not None and iv(prevk.iloc[0])[1] >= T:
                                keys.append((int(prevk.iloc[0]["K"]), feasible))
                else:
                    bmax = int(dsub["batteries"].max())
                    cur = dsub[dsub["batteries"] == bmax].sort_values("K")
                    possible = feasible = None
                    for _, rr in cur.iterrows():
                        q = iv(rr)
                        if q is None:
                            continue
                        k = int(rr["K"])
                        if possible is None and q[1] >= T:
                            possible = k
                        if feasible is None and q[0] >= T:
                            feasible = k
                    if possible is not None and (feasible is None or possible < feasible):
                        keys.append((possible, bmax))
                    elif feasible is not None:
                        prev = cur[cur["K"] < feasible].tail(1)
                        if len(prev) and iv(prev.iloc[0]) is not None and iv(prev.iloc[0])[1] >= T:
                            keys.append((int(prev.iloc[0]["K"]), bmax))
                return list(dict.fromkeys(keys))

            # A small deterministic number of refinement rounds prevents a
            # pathological frontier from turning the experiment controller into
            # an unbounded retry loop.  Failure to close remains explicit.
            for _ref_round in range(3):
                srow = _uav_selection_row()
                if srow is None:
                    break
                status = str(srow.get("selection_status", ""))
                if status == "needs_lexicographic_knee_certification":
                    kk, bb = int(srow["knee_K"]), int(srow["knee_B"])
                    log.info("E1[%s]: coverage/resource knee 已证明为 (K=%d,B=%d); "
                             "仅对此候选执行完整 lexicographic certification", uk, kk, bb)
                    _cell(kk, bb, force=True, full_lex=True)
                    break
                blockers = _refinement_keys(srow)
                if not blockers:
                    if (status == "uncertified_coverage_plateau"
                            and not _e1_plateau_long_refinement_allowed(args.e1_b_auto)):
                        log.warning(
                            "E1[%s]: fixed B grid 无法证明 formal coverage plateau；"
                            "--e1-b-auto=off，因此跳过长预算平台重试。"
                            "请用 --e1-b-auto on --resume on 只扩展新 B 格。",
                            uk)
                    elif status == "uncertified_coverage_plateau" and refinement_no_gain:
                        log.warning(
                            "E1[%s]: plateau 仍未证明，但所有可收紧端点本轮均无严格 bound 信息增益；"
                            "停止重复长跑并保持 unresolved。", uk)
                    elif (status == "uncertified_resource_knee"
                          and str(srow.get("saturation_proof", "")) == "hard-coverable-cap"):
                        log.info(
                            "E1[%s]: hard-coverable-cap 已给出离散阈值；跳过 generic coverage "
                            "长预算重试，直接交由 E1_knee_refine 的 exact target predecessor 求解。",
                            uk)
                    break
                log.info("E1[%s]: formal 选型证明尚缺 %s；按 bound 信息增益仅认证 %s",
                         uk, status, blockers)
                for kk, bb in blockers:
                    key = (int(kk), int(bb))
                    before = _monotone_coverage_interval(*key)
                    _cell(*key, force=True, coverage_certify=True)
                    after = _monotone_coverage_interval(*key)
                    improved = _e1_bound_strictly_improved(before, after)
                    rec = _row_for(*key)
                    if rec is not None:
                        rec["coverage_refinement_attempts"] = int(
                            rec.get("coverage_refinement_attempts", 0) or 0) + 1
                        rec["coverage_refinement_last_before"] = str(before)
                        rec["coverage_refinement_last_after"] = str(after)
                        rec["coverage_refinement_no_gain"] = bool(not improved)
                        _save(pd.DataFrame(rows), outdir, "E1_frontier.csv")
                    if not improved:
                        refinement_no_gain.add(key)
                        log.warning(
                            "E1[%s]: coverage certification cell %s 未严格收紧 [L,U] (%s -> %s); "
                            "已写入 checkpoint no-gain 标记；该 generic solve 不再重复"
                            "(本次预算 %.0fs)。",
                            uk, key, before, after,
                            float(getattr(args, "e1_certify_time_limit_s", 0.0)))

            # If refinement exposed a knee only in the final round, certify it now.
            srow = _uav_selection_row()
            if (srow is not None
                    and str(srow.get("selection_status", "")) == "needs_lexicographic_knee_certification"):
                kk, bb = int(srow["knee_K"]), int(srow["knee_B"])
                _cell(kk, bb, force=True, full_lex=True)

        # ---- 更新: 明细逐 UAV 各存一份(含裕度/绑定诊断); 旧文件语义 = --uav 档位不变 ----
        Bx = max(b_run)
        det = []
        if (max(ks), Bx) in chosen_at:
            det = _e1_detail_rows(chosen_at[(max(ks), Bx)], p_u, xi_amb, wamb, uk)
        elif (outdir / f"E1_detail_Kmax_{uk}.csv").is_file():
            # 更新(resume): 该格是上次会话完成的, chosen 不在内存; 明细文件已存在则沿用
            log.info("E1[%s]: resume 跳过明细重生成(E1_detail_Kmax_%s.csv 已存在)", uk, uk)
            try:
                det = pd.read_csv(outdir / f"E1_detail_Kmax_{uk}.csv",
                                  encoding="utf-8-sig").to_dict("records")
            except Exception:
                det = []
        else:
            # 更新(resume): 明细文件也缺 → 单独补解该代表格一次(不追加 rows)
            log.info("E1[%s]: resume 补解代表格 (K=%d,B=%d) 以生成明细", uk, max(ks), Bx)
            r_det = BP.solve_fleet_anytime(
                reach, opts, p_u, xi_amb, max(ks), T_eff,
                deck_delta_min=args.deck_delta_min, t_swap_min=t_swap,
                landing_clear_min=args.landing_clear_min,
                swap_station_capacity=args.swap_stations,
                quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                battery_reuse_mode=args.battery_reuse_mode,
                max_stops=cap_u, weather_unc=wamb,
                kappa_mode="vp_unimodal", batteries=Bx,
                seed_cols=(None if formal_ondemand else cols),
                deck_mode=args.deck_mode, t_launch_min=t_launch,
                solver_mode="exact-branch-price-cut",
                certified_route_universe=route_universe,
                **_e1_frontier_solver_kwargs(args))
            det = _e1_detail_rows(r_det["chosen"], p_u, xi_amb, wamb, uk)
        if det:
            _save(pd.DataFrame(det), outdir, f"E1_detail_Kmax_{uk}.csv")
    df = pd.DataFrame(rows); _save(df, outdir, "E1_frontier.csv")
    _audit_e1_validation_specificity(df, outdir)

    # 选型必须在完整前沿形成后完成；final test 在方案冻结后才运行，绝不回流改变选型。
    sel = e1_select_from_df(df, frac=getattr(args, "knee_frac", 0.95),
                            order=getattr(args, "knee_order", "BK"),
                            patience=max(1, int(args.e1_sat_patience)))
    pick, warns = _pick_selection(sel, metric=getattr(args, "selection_metric",
                                                       "safe_per_inventory_kWh"))
    sel = sel.copy()
    sel["selected"] = False
    sel["selected_by_metric"] = getattr(args, "selection_metric", "safe_per_inventory_kWh")
    for msg in warns:
        log.warning("E1 selection: %s", msg)
    post_selection_error = None
    if pick is not None:
        uk = str(pick["uav"]); Ksel = int(pick["knee_K"]); Bsel = int(pick["knee_B"])
        sel.loc[sel["uav"].astype(str) == uk, "selected"] = True
        ctx = uav_context[uk]
        r_sel = certified_knee_results.get((uk, Ksel, Bsel))
        if r_sel is None:
            r_sel = BP.solve_fleet_anytime(
                reach, opts, ctx["p"], xi_amb, Ksel, T_eff,
                deck_delta_min=args.deck_delta_min, t_swap_min=ctx["t_swap"],
                landing_clear_min=args.landing_clear_min,
                quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                swap_station_capacity=args.swap_stations,
                max_stops=ctx["cap"], weather_unc=wamb,
                kappa_mode="vp_unimodal", batteries=Bsel,
                deck_mode=args.deck_mode, t_launch_min=ctx["t_launch"],
                pool_h_mode=getattr(args, "pool_h", "pareto"),
                solver_mode=getattr(args, "solver_mode", "exact-branch-price-cut"),
                seed_cols=ctx["seed_cols"],
                certified_route_universe=ctx.get("route_universe"),
                **_e1_certify_solver_kwargs(args))
        detail = _e1_detail_rows(r_sel["chosen"], ctx["p"], xi_amb, wamb, uk)
        frontier_match = df[(df["uav"].astype(str) == uk) & (df["K"] == Ksel)
                            & (df["batteries"] == Bsel)]
        if len(frontier_match) != 1:
            raise RuntimeError(f"selected knee 在前沿中不是唯一格: {(uk, Ksel, Bsel)}")
        fr = frontier_match.iloc[0]
        if int(r_sel["covered"]) < int(fr["covered"]):
            raise RuntimeError("最终求解器覆盖低于其受限列池前沿，内部一致性失败")
        if int(r_sel["covered"]) != int(fr["covered"]):
            log.warning("完整路线空间将 selected knee 覆盖从 %d 改进到 %d；"
                        "选型仍基于 E1 前沿，最终方案以重解结果为准。",
                        int(fr["covered"]), int(r_sel["covered"]))
        detail_meta = dict(solution_role="selected_knee_final_resolve", K=Ksel, batteries=Bsel,
                           covered=int(r_sel["covered"]), frontier_covered=int(fr["covered"]),
                           selected_by_metric=str(pick["selected_by_metric"]),
                           final_solver_status=str(r_sel.get("status")),
                           global_certificate_available=_global_certificate_flag(r_sel),
                           global_route_space_certificate=_global_certificate_flag(r_sel),
                           implicit_route_space_certified=_implicit_route_space_certificate(r_sel),
                           certificate_field_conflict=_certificate_field_conflict(r_sel),
                           certificate_field_invalid=_certificate_field_invalid(r_sel),
                           model_contract_sha256=r_sel.get("model_contract_sha256"),
                           parameter_contract_sha256=r_sel.get("parameter_contract_sha256"),
                           instance_contract_sha256=r_sel.get("instance_contract_sha256"),
                           algorithm_contract_sha256=r_sel.get("algorithm_contract_sha256"),
                           resource_numeric_contract=r_sel.get("resource_numeric_contract"),
                           max_stops_requested=int(args.max_stops), stops_cap_spec=str(args.stops_cap),
                           max_stops_effective=int(ctx["cap"]),
                           max_stops_observed=max((len(c["tids"]) for c in r_sel["chosen"]), default=0))
        for row in detail:
            row.update(detail_meta)
        _save(pd.DataFrame(detail), outdir, "E1_detail_Kmax.csv")

        # 完整路线空间重解可能改变路线/资源分配；必须先在 validation 上重新审计
        # 这个【精确最终方案】，再冻结计划指纹。未经该门禁不得消费 final test。
        mask = sel["uav"].astype(str) == uk
        plan_fp = _frozen_plan_fingerprint(r_sel["chosen"])
        sel.loc[mask, "frozen_plan_fingerprint"] = plan_fp
        sel.loc[mask, "final_solver_status"] = str(r_sel.get("status"))
        sel.loc[mask, "global_certificate_available"] = _global_certificate_flag(r_sel)
        sel.loc[mask, "global_route_space_certificate"] = _global_certificate_flag(r_sel)
        sel.loc[mask, "implicit_route_space_certified"] = _implicit_route_space_certificate(r_sel)
        sel.loc[mask, "certificate_field_conflict"] = _certificate_field_conflict(r_sel)
        sel.loc[mask, "certificate_field_invalid"] = _certificate_field_invalid(r_sel)
        sel.loc[mask, "model_contract_sha256"] = r_sel.get("model_contract_sha256")
        sel.loc[mask, "parameter_contract_sha256"] = r_sel.get("parameter_contract_sha256")
        sel.loc[mask, "instance_contract_sha256"] = r_sel.get("instance_contract_sha256")
        sel.loc[mask, "algorithm_contract_sha256"] = r_sel.get("algorithm_contract_sha256")
        sel.loc[mask, "resource_numeric_contract"] = r_sel.get("resource_numeric_contract")
        _vmode = str(getattr(args, "validation_mode", "synthetic_stress"))
        validation_rp = _replay_columns(
            r_sel["chosen"], ctx["p"], xi_amb, ctx["p"].eps_E,
            n_per=args.replay_n, seed=7, wamb=wamb,
            validation_mode=_vmode,
            real_samples_csv=getattr(args, "validation_samples", None),
            weather_sample_mode=("real" if _vmode in ("real_validation", "real_holdout")
                                 else "synthetic"),
            holdout_disjointness_verified=False)
        validation_budget = RM.mission_eps_budget(ctx["p"], wamb is not None)
        exact_validation_holds = bool(
            _formal_validation_selection_gate(validation_rp) is True
            and int(validation_rp.get("n_missing", 0)) == 0)
        sel.loc[mask, "final_plan_validation_type"] = validation_rp.get("validation_type")
        sel.loc[mask, "final_plan_validation_upper95"] = validation_rp.get("upper95")
        sel.loc[mask, "final_plan_validation_n_total"] = int(
            validation_rp.get("n_test_total", 0))
        sel.loc[mask, "final_plan_validation_n_viol"] = int(
            validation_rp.get("n_viol_total", 0))
        sel.loc[mask, "final_plan_validation_n_missing"] = int(
            validation_rp.get("n_missing", 0))
        sel.loc[mask, "final_plan_validation_budget"] = float(validation_budget)
        sel.loc[mask, "final_plan_validation_holds"] = exact_validation_holds
        for row in detail:
            row.update(
                frozen_plan_fingerprint=str(plan_fp),
                final_plan_validation_type=validation_rp.get("validation_type"),
                final_plan_validation_upper95=validation_rp.get("upper95"),
                final_plan_validation_n_missing=int(validation_rp.get("n_missing", 0)),
                final_plan_validation_budget=float(validation_budget),
                final_plan_validation_holds=bool(exact_validation_holds))
        _save(pd.DataFrame(detail), outdir, "E1_detail_Kmax.csv")
        # selected=True 表示【完整路线空间最终方案】本身通过validation，而不是仅前沿格通过。
        sel.loc[mask, "selected"] = exact_validation_holds
        if not exact_validation_holds:
            post_selection_error = (
                "E1完整路线空间重解后的精确方案未通过validation风险门；"
                "已取消selected，且禁止消费final test。")
            log.error(post_selection_error)

        # Formal v9 protocol never consumes the independent test at E1.
        # E1 only freezes a validation-approved exact configuration; the test is
        # consumed once, after E2/A selection is frozen.  This prevents E1 from
        # turning the final test into a second validation set.
        if exact_validation_holds and getattr(args, "study_mode", "mechanism") == "formal":
            sel.loc[mask, "final_test_deferred_to_e2"] = True
            sel.loc[mask, "test_formal_holds"] = None
            stale_final = outdir / "E1_final_test.csv"
            if stale_final.exists():
                raise SystemExit(
                    "检测到旧协议 E1_final_test.csv。v9 正式协议禁止 E1 消费 final test；"
                    "请备份旧结果并删除该旧文件后重新运行，test 只能在 E2/A 冻结后消费一次。")
        # Legacy/mechanism-only compatibility: retain the old standalone E1 test
        # path outside the formal publication protocol.
        if (exact_validation_holds
                and _e1_final_test_consumption_allowed(args)):
            final_path = outdir / "E1_final_test.csv"
            test_hash = EU.sha256_file(args.final_test_samples) or "none"
            expected_key = (str(uk), int(Ksel), int(Bsel), str(plan_fp),
                            str(test_hash), RESULT_CONTRACT)
            reuse = False
            final_rec = None
            prior_invocations = 0
            if final_path.is_file():
                prev = pd.read_csv(final_path, encoding="utf-8-sig")
                if len(prev) == 1:
                    prow = prev.iloc[0]
                    try:
                        prior_invocations = int(prow.get("final_test_invocations", 1))
                    except Exception:
                        prior_invocations = 1
                    got_key = (str(prow.get("selected_uav")),
                               int(prow.get("selected_K")),
                               int(prow.get("selected_batteries")),
                               str(prow.get("frozen_plan_fingerprint")),
                               str(prow.get("final_test_samples_sha256")),
                               str(prow.get("result_contract")))
                    reuse = (got_key == expected_key)
                    if reuse:
                        final_rec = prow.to_dict()
                if not reuse and not bool(getattr(args, "allow_final_test_rerun", False)):
                    raise SystemExit(
                        "E1_final_test.csv 已存在且冻结方案/测试哈希不一致。"
                        "为避免反复查看test，默认拒绝重跑；确需审计性重跑时显式加 "
                        "--allow-final-test-rerun，并记录为授权重跑。")
            if not reuse:
                final_mode = (
                    "real_joint_final_test"
                    if getattr(args, "final_weather_mode", "real") == "real"
                    else "real_xi_final_test")
                test_rp = _replay_columns(
                    r_sel["chosen"], ctx["p"], xi_amb, ctx["p"].eps_E,
                    n_per=args.replay_n, seed=1707, wamb=wamb,
                    validation_mode=final_mode,
                    real_samples_csv=args.final_test_samples,
                    weather_sample_mode=getattr(args, "final_weather_mode", "real"),
                    holdout_disjointness_verified=bool(
                        getattr(args, "_holdout_disjointness_verified", False)))
                test_budget = RM.mission_eps_budget(
                    ctx["p"],
                    wamb is not None or getattr(args, "final_weather_mode", "real") == "real")
                final_rec = dict(
                    result_contract=RESULT_CONTRACT, project_name=PROJECT_NAME,
                    selection_rule=(
                        "E1 frontier validation selection; complete finite-route final resolve; "
                        "exact-plan revalidation; frozen-plan one-time final test"),
                    selected_uav=str(uk), selected_K=int(Ksel),
                    selected_batteries=int(Bsel),
                    frozen_plan_fingerprint=str(plan_fp),
                    train_samples_sha256=(
                        EU.sha256_file(getattr(args, "xi_train_samples", None)) or "none"),
                    validation_samples_sha256=(
                        EU.sha256_file(getattr(args, "validation_samples", None)) or "none"),
                    validation_upper95=validation_rp.get("upper95"),
                    validation_n_missing=int(validation_rp.get("n_missing", 0)),
                    validation_holds=bool(exact_validation_holds),
                    final_test_samples_sha256=str(test_hash),
                    final_test_invocations=int(prior_invocations + 1),
                    final_test_validation_type=test_rp.get("validation_type"),
                    final_test_n_total=int(test_rp.get("n_test_total", 0)),
                    final_test_n_viol=int(test_rp.get("n_viol_total", 0)),
                    final_test_upper95=test_rp.get("upper95"),
                    final_test_n_missing=int(test_rp.get("n_missing", 0)),
                    final_test_ci_method=test_rp.get("ci_method"),
                    final_test_formal_reliability_claim_eligible=bool(
                        test_rp.get("formal_reliability_claim_eligible", False)),
                    final_test_holds=bool(
                        test_rp.get("formal_reliability_claim_eligible", False)
                        and test_rp.get("allocation_budget_holds") is True
                        and int(test_rp.get("n_missing", 0)) == 0),
                    evidence_scope=test_rp.get("evidence_scope",
                                               "mechanism-or-partial-evidence"),
                    final_solver_status=str(r_sel.get("status")),
                    global_certificate_available=_global_certificate_flag(r_sel),
                    global_route_space_certificate=_global_certificate_flag(r_sel),
                    implicit_route_space_certified=_implicit_route_space_certificate(r_sel),
                    certificate_field_conflict=_certificate_field_conflict(r_sel),
                    certificate_field_invalid=_certificate_field_invalid(r_sel),
                    model_contract_sha256=r_sel.get("model_contract_sha256"),
                    parameter_contract_sha256=r_sel.get("parameter_contract_sha256"),
                    instance_contract_sha256=r_sel.get("instance_contract_sha256"),
                    algorithm_contract_sha256=r_sel.get("algorithm_contract_sha256"),
                    resource_numeric_contract=r_sel.get("resource_numeric_contract"),
                    final_test_rerun_authorized=bool(
                        getattr(args, "allow_final_test_rerun", False)))
                _save(pd.DataFrame([final_rec]), outdir, "E1_final_test.csv")
            else:
                log.info("E1 final test 已按相同冻结方案和测试哈希执行过，"
                         "本次复用结果，不重复消费test。")

            # E1_selection.csv 仅镜像一次性审计结果，证据原件是 E1_final_test.csv。
            key_map = {
                "test_validation_type": "final_test_validation_type",
                "test_n_total": "final_test_n_total",
                "test_n_viol": "final_test_n_viol",
                "test_upper95": "final_test_upper95",
                "test_n_missing_replay": "final_test_n_missing",
                "test_formal_reliability_claim_eligible":
                    "final_test_formal_reliability_claim_eligible",
                "test_formal_holds": "final_test_holds",
                "test_evidence_scope": "evidence_scope",
            }
            for dst, src in key_map.items():
                sel.loc[mask, dst] = final_rec.get(src)
            sel.loc[mask, "test_final_result_reused"] = bool(reuse)
    else:
        stale = outdir / "E1_detail_Kmax.csv"
        if stale.exists():
            stale.unlink()
    _save(sel, outdir, "E1_selection.csv")
    EU.write_run_manifest(outdir, "E1_frontier", args,
                          input_paths=[x for x in [getattr(args, "xi_train_samples", None),
                                                  getattr(args, "validation_samples", None),
                                                  getattr(args, "final_test_samples", None),
                                                  getattr(args, "_resolved_track_csv", None),
                                                  getattr(args, "weather_moments_csv", None)]
                                       if x is not None],
                          extra={"rows": len(df), "result_contract": RESULT_CONTRACT,
                                 "uavs": uavs, "fleet_ks": list(ks),
                                 "max_stops_requested": int(args.max_stops),
                                 "stops_cap_spec": str(args.stops_cap),
                                 "max_stops_effective_by_uav": effective_caps,
                                 **_formal_instance_manifest_extra(args)})
    if post_selection_error is not None and getattr(args, "study_mode", "mechanism") == "formal":
        raise SystemExit(post_selection_error)
    return df




def E1_lex_certify(reach, opts, p_base, xi_amb, wamb, outdir, args, kind, T_eff):
    """Direct full lexicographic certification for one explicit resource point.

    This paper-facing entry is intentionally independent of the E1 plateau/knee
    controller.  It does not change R-BPC semantics: the underlying solver is the
    same ``solve_scope="lexicographic"`` path used for certified knee resolves.
    It is useful when a fixed (UAV,K,B) configuration has already been chosen for
    a main-instance theorem and Stage-1 coverage is known, but Stage-2 energy still
    needs a formal certificate.

    Required CLI fields: ``--uav``, ``--k`` and ``--batteries``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if str(getattr(args, "uav", "auto")) == "auto":
        raise SystemExit("E1_lex_certify requires explicit --uav (for example --uav M)")
    if getattr(args, "k", None) is None:
        raise SystemExit("E1_lex_certify requires explicit --k")
    if getattr(args, "batteries", None) is None:
        raise SystemExit("E1_lex_certify requires explicit --batteries")

    uk = str(args.uav)
    K = int(args.k)
    B = int(args.batteries)
    p_u = M.apply_uav_profile(p_base, uk)
    t_swap, t_launch = _uav_deck(args, uk)
    cap_u = _stops_cap(args.stops_cap, p_u, xi_amb, args.max_stops)

    route_universe = _build_or_get_e1_complete_universe(
        reach, opts, p_u, xi_amb, wamb, args, T_eff,
        cap_u, t_launch, uk, outdir)
    seeds = (None if route_universe is not None else
             _e1_build_formal_warmstart(
                 reach, opts, p_u, xi_amb, wamb, args, T_eff, cap_u))

    result = BP.solve_fleet_anytime(
        reach, opts, p_u, xi_amb, K, T_eff,
        deck_delta_min=args.deck_delta_min,
        t_swap_min=t_swap,
        landing_clear_min=args.landing_clear_min,
        quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
        swap_station_capacity=args.swap_stations,
        battery_reuse_mode=args.battery_reuse_mode,
        max_stops=cap_u,
        weather_unc=wamb,
        kappa_mode="vp_unimodal",
        batteries=B,
        seed_cols=seeds,
        deck_mode=args.deck_mode,
        t_launch_min=t_launch,
        pool_h_mode=getattr(args, "pool_h", "pareto"),
        solver_mode=getattr(args, "solver_mode", "exact-branch-price-cut"),
        certified_route_universe=route_universe,
        **_e1_certify_solver_kwargs(args))

    # Persist a compact certificate row plus the exact chosen-route detail.
    scalar_keys = [
        "status", "termination_reason", "solve_scope", "runtime_s",
        "coverage_incumbent", "coverage_upper_bound", "coverage_gap_abs",
        "coverage_gap_pct", "coverage_optimal",
        "energy_incumbent_Wh", "energy_lower_bound_Wh", "energy_gap_abs_Wh",
        "energy_gap_pct", "conditional_energy_gap_pct", "global_energy_gap_reason",
        "energy_optimal", "lexicographic_optimal", "global_certificate_available",
        "global_route_space_certificate", "coverage_global_certificate_available",
        "coverage_physical_model_certificate", "coverage_algorithmic_certificate",
        "pricing_complete", "pricing_bound_available", "resource_audit_complete",
        "branching_complete", "farkas_pricing_complete", "bound_scope", "bound_source",
        "open_nodes", "processed_nodes", "rmp_solves", "phase_one_solves",
        "resource_audit_calls", "resource_cuts_added", "pricing_candidates",
        "pricing_nodes", "pricing_calls", "exact_pricing_calls",
        "pricing_runtime_s", "pricing_physical_evaluator_runtime_s",
        "pricing_prefix_bound_runtime_s", "pricing_prefix_service_runtime_s",
        "pricing_certified_prefix_prunes", "pricing_service_floor_prunes",
        "pricing_physical_cache_hits", "pricing_physical_cache_misses",
        "rmp_runtime_s", "phase_one_runtime_s", "resource_audit_runtime_s",
        "coverage_pricing_runtime_s", "energy_pricing_runtime_s",
        "coverage_rmp_runtime_s", "energy_rmp_runtime_s",
        "coverage_resource_audit_runtime_s", "energy_resource_audit_runtime_s",
        "archive_primal_recovery_calls", "archive_primal_recovery_improvements",
        "archive_primal_recovery_best_coverage", "archive_primal_recovery_runtime_s",
        "battery_halfcap_formal_enabled", "battery_halfcap_rhs",
        "pricing_resource_variant_enabled", "pricing_resource_variant_added",
        "model_contract_sha256", "parameter_contract_sha256",
        "instance_contract_sha256", "algorithm_contract_sha256",
        "result_certificate_contract", "formal_proof_contract", "proof_contract_sha256",
    ]
    row = dict(
        experiment="E1_lex_certify",
        uav=uk, K=K, batteries=B,
        pricing_mode=getattr(args, "pricing_mode", None),
        solver_mode=getattr(args, "solver_mode", None),
        max_stops_requested=int(args.max_stops),
        max_stops_effective=int(cap_u),
        route_universe_source=result.get("route_universe_source"),
        route_space_complete=result.get("route_space_complete"),
    )
    for key in scalar_keys:
        row[key] = result.get(key)
    row["covered_turbine_ids_json"] = json.dumps(
        result.get("covered_turbine_ids", []), ensure_ascii=False, default=str)
    row["chosen_route_signatures_json"] = json.dumps(
        [c.get("exact_route_signature", c.get("route_signature", c.get("signature")))
         for c in result.get("chosen", [])],
        ensure_ascii=False, default=str)
    row["archive_primal_recovery_records_json"] = result.get(
        "archive_primal_recovery_records_json")
    row["pricing_resource_variant_records_json"] = result.get(
        "pricing_resource_variant_records_json")
    row["pricing_depth_certified_prefix_prunes_json"] = json.dumps(
        result.get("pricing_depth_certified_prefix_prunes", {}),
        ensure_ascii=False, default=str)
    row["pricing_depth_service_floor_prunes_json"] = json.dumps(
        result.get("pricing_depth_service_floor_prunes", {}),
        ensure_ascii=False, default=str)

    _save(pd.DataFrame([row]), outdir, "E1_lex_certify.csv")
    telemetry = dict(
        pricing_call_records=result.get("pricing_call_records", []),
        rmp_records=result.get("rmp_records", []),
        resource_audit_records=result.get("resource_audit_records", []),
        aggregate={key: result.get(key) for key in scalar_keys if (
            key.endswith("_runtime_s") or key.startswith("pricing_")
            or key in {"rmp_solves", "phase_one_solves",
                       "resource_audit_calls", "resource_cuts_added",
                       "pricing_nodes", "pricing_calls", "exact_pricing_calls"})},
        depth_certified_prefix_prunes=result.get(
            "pricing_depth_certified_prefix_prunes", {}),
        depth_service_floor_prunes=result.get(
            "pricing_depth_service_floor_prunes", {}))
    (outdir / "E1_lex_certify_telemetry.json").write_text(
        json.dumps(telemetry, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    detail = _e1_detail_rows(result.get("chosen", []), p_u, xi_amb, wamb, uk)
    if detail:
        ddf = pd.DataFrame(detail)
        ddf.insert(0, "solution_role", "direct_full_lex_certification")
        ddf.insert(1, "K", K)
        ddf.insert(2, "batteries", B)
        _save(ddf, outdir, "E1_lex_certify_detail.csv")

    EU.write_run_manifest(
        outdir, "E1_lex_certify", args,
        input_paths=[x for x in [
            getattr(args, "xi_train_samples", None),
            getattr(args, "validation_samples", None),
            getattr(args, "final_test_samples", None),
            getattr(args, "_resolved_track_csv", None),
            getattr(args, "weather_moments_csv", None)]
            if x is not None],
        extra=dict(
            result_contract=RESULT_CONTRACT,
            uav=uk, K=K, batteries=B,
            solve_scope="lexicographic",
            coverage_optimal=bool(result.get("coverage_optimal", False)),
            energy_optimal=bool(result.get("energy_optimal", False)),
            lexicographic_optimal=bool(result.get("lexicographic_optimal", False)),
            global_certificate_available=bool(
                _global_certificate_flag(result)),
            **_formal_instance_manifest_extra(args)))
    return result


def _e1_update_knee_row_from_lex(df, idx, result, p_u, xi_amb, wamb, args):
    """Replace one frontier knee row by its full-lex exact result and validation."""
    if not _global_certificate_flag(result):
        return False
    chosen = result.get("chosen", [])
    if getattr(args, "study_mode", "mechanism") == "formal":
        _verify_formal_sample_hashes_unchanged(args)
    rp = _replay_columns(
        chosen, p_u, xi_amb, p_u.eps_E, n_per=args.replay_n, seed=7,
        wamb=wamb, validation_mode=getattr(args, "validation_mode", "real_validation"),
        real_samples_csv=getattr(args, "validation_samples", None),
        weather_sample_mode=("real" if getattr(args, "validation_mode", "") in
                             ("real_validation", "real_holdout") else "synthetic"))
    safe = set(rp.get("safe_tids", set()))
    B = int(result.get("batteries", df.loc[idx, "batteries"]))
    inventory_kwh = float(B) * float(p_u.B_k) / 1000.0
    max_stops_observed = max((len(c.get("tids", ())) for c in chosen), default=0)
    energy = float(result.get("energy_Wh", 0.0) or 0.0)
    allocation_hold = _formal_validation_selection_gate(rp)
    mission_hold = rp.get("mission_requirement_holds")
    updates = dict(
        covered=int(result.get("covered", 0)),
        safe_served=int(len(safe)),
        per_battery=(round(len(safe) / B, 3) if B else None),
        flights=int(result.get("flights", len(chosen))),
        mean_stops=float(result.get("mean_stops", 0.0) or 0.0),
        max_stops_observed=int(max_stops_observed),
        stops_cap_hit=bool(max_stops_observed >= int(df.loc[idx].get("max_stops_effective", 0) or 0)
                           and len(chosen) > 0),
        energy_Wh=energy,
        energy_per_safe=(round(energy / len(safe), 6) if safe else None),
        inventory_energy_kWh=inventory_kwh,
        safe_per_inventory_kWh=(round(len(safe) / inventory_kwh, 6)
                                if inventory_kwh > 0 else None),
        emp_viol=rp.get("emp"),
        max_col_viol=rp.get("max_col_viol"),
        validation_plan_fingerprint=rp.get("validation_plan_fingerprint"),
        validation_route_records_json=json.dumps(
            rp.get("route_validation_records", []), ensure_ascii=False, sort_keys=True),
        allocation_budget=rp.get("allocation_budget"),
        mission_requirement_budget=rp.get("mission_requirement_budget"),
        allocation_budget_holds=rp.get("allocation_budget_holds"),
        mission_requirement_holds=rp.get("mission_requirement_holds"),
        all_routes_allocation_holds=rp.get("all_routes_allocation_holds"),
        all_routes_mission_holds=rp.get("all_routes_mission_holds"),
        validation_gate_contract=rp.get("validation_gate_contract"),
        validation_event_fingerprint_count=rp.get("validation_event_fingerprint_count"),
        validation_unique_event_fingerprint_count=rp.get(
            "validation_unique_event_fingerprint_count"),
        validation_duplicate_event_groups=rp.get("validation_duplicate_event_groups"),
        validation_event_group_sizes_json=rp.get("validation_event_group_sizes_json"),
        validation_event_grouping_used_for_gate=rp.get(
            "validation_event_grouping_used_for_gate"),
        strict_5pct_holds=mission_hold,
        union_budget_holds=allocation_hold,
        plan_holds=allocation_hold,
        n_test_total=rp.get("n_test_total", 0),
        n_viol_total=rp.get("n_viol_total", 0),
        emp_viol_ci95_low=rp.get("ci95_low"),
        emp_viol_ci95_high=rp.get("ci95_high"),
        emp_viol_upper95=rp.get("upper95"),
        n_replayed_cols=rp.get("n_replayed", 0),
        n_missing_replay=rp.get("n_missing", 0),
        validation_type=rp.get("validation_type"),
        coverage_incumbent=result.get("coverage_incumbent"),
        coverage_upper_bound=result.get("coverage_upper_bound"),
        coverage_gap_abs=result.get("coverage_gap_abs"),
        coverage_gap_pct=result.get("coverage_gap_pct"),
        coverage_optimal=result.get("coverage_optimal"),
        energy_incumbent_Wh=result.get("energy_incumbent_Wh"),
        energy_lower_bound_Wh=result.get("energy_lower_bound_Wh"),
        energy_gap_abs_Wh=result.get("energy_gap_abs_Wh"),
        energy_gap_pct=result.get("energy_gap_pct"),
        conditional_energy_gap_pct=result.get("conditional_energy_gap_pct"),
        energy_optimal=result.get("energy_optimal"),
        lexicographic_optimal=result.get("lexicographic_optimal"),
        global_certificate_available=_global_certificate_flag(result),
        global_route_space_certificate=_global_certificate_flag(result),
        implicit_route_space_certified=_implicit_route_space_certificate(result),
        coverage_global_certificate_available=_coverage_certificate_flag(result),
        frontier_evaluated=True,
        frontier_coverage_certified=_coverage_certificate_flag(result),
        frontier_lexicographic_certified=_global_certificate_flag(result),
        frontier_completion_state="lexicographic-certified",
        solve_scope=result.get("solve_scope", "lexicographic"),
        runtime_s=result.get("runtime_s"),
        **_e1_certificate_provenance_fields(result),
        optimization_status=result.get("status"),
        termination_reason=result.get("termination_reason"),
    )
    for k, v in updates.items():
        df.loc[idx, k] = v
    return True



def _save_e1_target_witness(result, p_u, xi_amb, wamb, uk, K, B, target, outdir):
    """Persist every certified target YES witness with route/resource assignments."""
    valid, cert = _strict_certificate_bool(result.get("target_decision_certified", False))
    if not (valid and cert and str(result.get("target_decision", "")).upper() == "FEASIBLE"):
        return None
    chosen = result.get("chosen", [])
    if not chosen:
        raise RuntimeError("certified TARGET_FEASIBLE has no persisted witness columns")
    rows = _e1_detail_rows(chosen, p_u, xi_amb, wamb, uk)
    df = pd.DataFrame(rows)
    df.insert(0, "solution_role", "certified_target_feasible_witness")
    df.insert(1, "target_coverage", int(target))
    df.insert(2, "K", int(K))
    df.insert(3, "batteries", int(B))
    df["target_certificate_type"] = result.get("target_certificate_type")
    df["route_universe_source"] = result.get("route_universe_source")
    df["route_space_complete"] = result.get("route_space_complete")
    df["complete_route_universe_columns_sha256"] = result.get(
        "complete_route_universe_columns_sha256")
    name = f"E1_target_witness_{uk}_K{int(K)}_B{int(B)}_T{int(target)}.csv"
    return _save(df, Path(outdir), name)


def _save_e1_knee_detail(detail_rows, outdir, uk, K, B):
    """Persist knee detail from the list returned by `_e1_detail_rows`.

    The helper makes the list/DataFrame boundary explicit.  Before v11 this
    path was never reached by the unresolved real knee and a latent `.empty`
    access on the Python list remained hidden.
    """
    if not detail_rows:
        return None
    detail_df = pd.DataFrame(detail_rows)
    detail_df.insert(0, "solution_role", "certified_resource_knee_full_lex")
    detail_df.insert(1, "K", int(K))
    detail_df.insert(2, "batteries", int(B))
    return _save(detail_df, Path(outdir), f"E1_detail_knee_{uk}.csv")


def E1_knee_refine(reach, opts, p_base, xi_amb, wamb, outdir, args, kind, T_eff):
    """v12 direct target/resource closure using a certified complete universe when eligible."""
    outdir = Path(outdir)
    path = outdir / "E1_frontier.csv"
    if not path.is_file():
        raise SystemExit(f"E1_knee_refine requires existing {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    _target_cols = {
        "coverage_incumbent_refined": np.nan,
        "coverage_upper_bound_refined": np.nan,
        "target_coverage_last": np.nan,
        "target_decision_last": None,
        "target_decision_certified": False,
        "target_certificate_type": None,
        "target_result_certificate_contract": None,
        "target_formal_proof_contract": None,
        "target_proof_contract_sha256": None,
        "target_runtime_s": np.nan,
    }
    _missing_target_cols = {k: v for k, v in _target_cols.items() if k not in df.columns}
    if _missing_target_cols:
        df = pd.concat(
            [df, pd.DataFrame(
                {k: [v] * len(df) for k, v in _missing_target_cols.items()},
                index=df.index)],
            axis=1)
    current_resume_sha = _resume_context_sha256(
        reach=reach, launch_options=opts, params=p_base, xi_ambiguity=xi_amb,
        weather_uncertainty=wamb, track_kind=kind, T_eff_min=float(T_eff))
    setattr(args, "_resume_input_sha256", current_resume_sha)
    _validate_e1_frontier_for_target_refine(df, args, current_resume_sha)

    cert_path = outdir / "E1_knee_target_certificates.csv"
    cert_df = (pd.read_csv(cert_path, encoding="utf-8-sig")
               if cert_path.is_file() else pd.DataFrame())
    cert_rows = cert_df.to_dict("records") if not cert_df.empty else []

    def cached_cert(uk, K, B, target):
        for q in reversed(cert_rows):
            try:
                same = (str(q.get("uav")) == str(uk)
                        and int(q.get("K")) == int(K)
                        and int(q.get("batteries")) == int(B)
                        and int(q.get("target_coverage")) == int(target)
                        and str(q.get("resume_input_sha256")) == str(current_resume_sha)
                        and str(q.get("result_certificate_contract"))
                            == str(BP.RESULT_CERTIFICATE_CONTRACT)
                        and str(q.get("formal_proof_contract"))
                            == str(BP.FORMAL_PROOF_CONTRACT))
            except Exception:
                continue
            if not same:
                continue
            valid, value = _strict_certificate_bool(q.get("target_decision_certified", False))
            if valid and value:
                return q
        return None

    max_rounds = max(8, len(df) + 2)
    for uk in sorted(df["uav"].astype(str).unique()):
        p_u = M.apply_uav_profile(p_base, uk)
        t_swap, t_launch = _uav_deck(args, uk)
        cap_u = _stops_cap(args.stops_cap, p_u, xi_amb, args.max_stops)
        route_universe = _build_or_get_e1_complete_universe(
            reach, opts, p_u, xi_amb, wamb, args, T_eff,
            cap_u, t_launch, uk, outdir)
        seeds = (None if route_universe is not None else
                 _e1_build_formal_warmstart(
                     reach, opts, p_u, xi_amb, wamb, args, T_eff, cap_u))

        for _ in range(max_rounds):
            sel = e1_select_from_df(
                df, frac=args.knee_frac, order=args.knee_order,
                patience=max(1, int(args.e1_sat_patience)))
            sr = sel[sel["uav"].astype(str) == uk]
            if len(sr) != 1:
                raise RuntimeError(f"E1 selection row not unique for {uk}")
            srow = sr.iloc[0]
            status = str(srow.get("selection_status", ""))
            if status != "uncertified_resource_knee":
                break
            blockers = _e1_target_blockers(df, uk, srow, order=args.knee_order)
            if not blockers:
                log.warning("E1[%s] resource knee unresolved but no monotone target blocker found.", uk)
                break
            K, B, target, role = blockers[0]
            result = cached_cert(uk, K, B, target)
            if result is None:
                log.info("E1[%s] exact target refine: %s K=%d B=%d target=%d",
                         uk, role, K, B, target)
                result = BP.solve_fleet_anytime(
                    reach, opts, p_u, xi_amb, int(K), T_eff,
                    deck_delta_min=args.deck_delta_min, t_swap_min=t_swap,
                    landing_clear_min=args.landing_clear_min,
                    quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                    swap_station_capacity=args.swap_stations,
                    max_stops=cap_u, weather_unc=wamb,
                    kappa_mode="vp_unimodal", batteries=int(B),
                    seed_cols=seeds, deck_mode=args.deck_mode, t_launch_min=t_launch,
                    pool_h_mode=getattr(args, "pool_h", "pareto"),
                    solver_mode=getattr(args, "solver_mode", "exact-branch-price-cut"),
                    certified_route_universe=route_universe,
                    target_closure_checkpoint_path=str(
                        outdir / f"E1_target_closure_{uk}_K{int(K)}_B{int(B)}_T{int(target)}.json"),
                    target_closure_resume=(str(getattr(args, "resume", "on")).lower() == "on"),
                    **_e1_target_solver_kwargs(args, target))
                _save_e1_target_witness(
                    result, p_u, xi_amb, wamb, uk, K, B, target, outdir)
                rec = dict(
                    uav=uk, K=int(K), batteries=int(B), target_coverage=int(target),
                    predecessor_role=role,
                    target_decision=result.get("target_decision"),
                    target_decision_certified=result.get("target_decision_certified"),
                    target_feasible_proven=result.get("target_feasible_proven"),
                    target_infeasible_proven=result.get("target_infeasible_proven"),
                    target_certificate_type=result.get("target_certificate_type"),
                    target_coverage_lower_bound=result.get("target_coverage_lower_bound"),
                    target_coverage_upper_bound=result.get("target_coverage_upper_bound"),
                    target_witness_coverage=result.get("target_witness_coverage"),
                    termination_reason=result.get("termination_reason"),
                    runtime_s=result.get("runtime_s"),
                    open_nodes=result.get("open_nodes"),
                    target_master_backend=result.get("target_master_backend"),
                    target_master_solves=result.get("target_master_solves"),
                    target_fullcover_strong_cuts=result.get("target_fullcover_strong_cuts"),
                    target_fullcover_cuts_loaded=result.get("target_fullcover_cuts_loaded"),
                    target_battery_core_cuts=result.get("target_battery_core_cuts"),
                    target_resource_audit_nodes=result.get("target_resource_audit_nodes"),
                    target_resource_audit_memo_hits=result.get("target_resource_audit_memo_hits"),
                    target_exact_cover_nodes=result.get("target_exact_cover_nodes"),
                    target_battery_relaxation_nodes=result.get("target_battery_relaxation_nodes"),
                    target_checkpoint_writes=result.get("target_checkpoint_writes"),
                    target_closure_context_sha256=result.get("target_closure_context_sha256"),
                    target_closure_checkpoint_contract=result.get("target_closure_checkpoint_contract"),
                    target_global_battery_relaxation_status=result.get(
                        "target_global_battery_relaxation_status"),
                    target_global_battery_min_required=result.get(
                        "target_global_battery_min_required"),
                    target_global_battery_dp_states=result.get(
                        "target_global_battery_dp_states"),
                    target_global_battery_one_pack_masks=result.get(
                        "target_global_battery_one_pack_masks"),
                    pricing_bound_available=result.get("pricing_bound_available"),
                    farkas_pricing_complete=result.get("farkas_pricing_complete"),
                    resource_audit_complete=result.get("resource_audit_complete"),
                    branching_complete=result.get("branching_complete"),
                    result_certificate_contract=result.get("result_certificate_contract"),
                    formal_proof_contract=result.get("formal_proof_contract"),
                    proof_contract_sha256=result.get("proof_contract_sha256"),
                    route_universe_source=result.get("route_universe_source"),
                    route_space_complete=result.get("route_space_complete"),
                    route_space_materialized=result.get("route_space_materialized"),
                    complete_route_universe_columns_sha256=result.get(
                        "complete_route_universe_columns_sha256"),
                    complete_route_universe_contract=result.get(
                        "complete_route_universe_contract"),
                    resume_input_sha256=current_resume_sha)
                cert_rows.append(rec)
                _save(pd.DataFrame(cert_rows), outdir, "E1_knee_target_certificates.csv")
            changed = _apply_target_decision_to_frontier(
                df, uk, K, B, target, result)
            _save(df, outdir, "E1_frontier.csv")
            valid, cert = _strict_certificate_bool(result.get("target_decision_certified", False))
            if not (valid and cert):
                log.warning("E1[%s] target K=%d B=%d T=%d unresolved; no bound tightened.",
                            uk, K, B, target)
                break
            if not changed:
                break
        # After target closure, full lex only the resulting certified-resource knee.
        sel = e1_select_from_df(
            df, frac=args.knee_frac, order=args.knee_order,
            patience=max(1, int(args.e1_sat_patience)))
        srow = sel[sel["uav"].astype(str) == uk].iloc[0]
        if str(srow.get("selection_status")) == "needs_lexicographic_knee_certification":
            Kk, Bk = int(srow["knee_K"]), int(srow["knee_B"])
            idxs = list(df.index[(df["uav"].astype(str) == uk)
                                 & (pd.to_numeric(df["K"], errors="coerce") == Kk)
                                 & (pd.to_numeric(df["batteries"], errors="coerce") == Bk)])
            if len(idxs) != 1:
                raise RuntimeError(f"knee cell not unique: {(uk, Kk, Bk)}")
            log.info("E1[%s] resource knee certified at K=%d B=%d; full lex resolve.", uk, Kk, Bk)
            lex = BP.solve_fleet_anytime(
                reach, opts, p_u, xi_amb, Kk, T_eff,
                deck_delta_min=args.deck_delta_min, t_swap_min=t_swap,
                landing_clear_min=args.landing_clear_min,
                quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                swap_station_capacity=args.swap_stations,
                max_stops=cap_u, weather_unc=wamb,
                kappa_mode="vp_unimodal", batteries=Bk,
                seed_cols=seeds, deck_mode=args.deck_mode, t_launch_min=t_launch,
                pool_h_mode=getattr(args, "pool_h", "pareto"),
                solver_mode=getattr(args, "solver_mode", "exact-branch-price-cut"),
                certified_route_universe=route_universe,
                **_e1_certify_solver_kwargs(args))
            if _global_certificate_flag(lex):
                _e1_update_knee_row_from_lex(
                    df, idxs[0], lex, p_u, xi_amb, wamb, args)
                detail = _e1_detail_rows(lex.get("chosen", []), p_u, xi_amb, wamb, uk)
                _save_e1_knee_detail(detail, outdir, uk, Kk, Bk)
                _save(df, outdir, "E1_frontier.csv")
            else:
                log.warning("E1[%s] knee full-lex solve unresolved; result not promoted.", uk)

    _audit_e1_validation_specificity(df, outdir)
    sel = e1_select_from_df(
        df, frac=args.knee_frac, order=args.knee_order,
        patience=max(1, int(args.e1_sat_patience)))
    _save(sel, outdir, "E1_selection.csv")
    args._e1_sel_df = sel
    return df, sel


def e1_select_from_df(df, frac=0.95, order="BK", patience=2):
    """Build an auditable E1 resource knee.

    Formal exact mode uses the optimization coverage function ``C*(K,B)`` rather
    than post-validation ``safe_served`` for monotonicity/saturation arguments.
    The key inequalities are

        C_inc(K,B) <= C*(K,B) <= UB_C(K,B),
        C*(K1,B1) <= C*(K2,B2)  for K1<=K2, B1<=B2.

    Hence at fixed Kmax, if ``C_inc(B0) == UB_C(B1) == P`` then every resource
    point between B0 and B1 has exact coverage P.  A knee threshold T is proved
    resource-minimal by the analogous one-dimensional boundary sandwiches along
    the selected BK/KB order.  Only after this coverage/resource proof is the
    knee plan eligible for a full lexicographic exact resolve and validation.

    Mechanism/legacy runs retain the historical observed-safe frontier rule, but
    those rows are never promoted to a formal physical optimization certificate.
    """
    need = {"uav", "K", "batteries", "safe_served", "per_battery"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"E1_select: CSV 缺列 {sorted(miss)} —— 请确认传入的是 E1_frontier.csv")

    def _bool_col(row, name):
        if name not in row.index:
            return False
        valid, value = _strict_certificate_bool(row.get(name))
        return bool(valid and value)

    def _iv(row):
        return _e1_raw_coverage_interval_record(row)

    def _metric_fields(knee, sub):
        if knee is None:
            return None, None, None
        eps_knee = (float(knee["energy_per_safe"])
                    if "energy_per_safe" in sub and pd.notna(knee.get("energy_per_safe")) else None)
        inv = None
        per_inv = None
        if "inventory_energy_kWh" in sub and pd.notna(knee.get("inventory_energy_kWh")):
            inv = float(knee["inventory_energy_kWh"])
        elif "B_k_Wh" in sub and pd.notna(knee.get("B_k_Wh")):
            inv = float(knee["batteries"]) * float(knee["B_k_Wh"]) / 1000.0
        if "safe_per_inventory_kWh" in sub and pd.notna(knee.get("safe_per_inventory_kWh")):
            per_inv = float(knee["safe_per_inventory_kWh"])
        elif inv and inv > 0:
            per_inv = float(knee["safe_served"]) / inv
        return eps_knee, inv, per_inv

    out = []
    for uk in df["uav"].drop_duplicates():
        sub = df[df["uav"] == uk].copy()
        formal = bool(
            "study_mode" in sub.columns
            and len(sub)
            and sub["study_mode"].astype(str).str.lower().eq("formal").all()
            and {"coverage_incumbent", "coverage_upper_bound"}.issubset(sub.columns))

        if formal:
            Kx = int(sub["K"].max())

            def _miv(K, B):
                return _e1_monotone_coverage_interval(sub, K, B)

            cur = sub[sub["K"] == Kx].sort_values("batteries")
            plateau_obs_safe = int(sub["safe_served"].max())
            sat = False
            plateau_cov = None
            sat_proof = "unproved"
            if len(cur):
                end = cur.iloc[-1]
                end_iv = _miv(Kx, int(end["batteries"]))
                hard_cap = int(end.get("coverable_note", len(sub)) or 0)
                if end_iv is not None and hard_cap >= 0 and end_iv[0] >= hard_cap:
                    sat, plateau_cov, sat_proof = True, hard_cap, "hard-coverable-cap"
                elif len(cur) >= int(patience) + 1 and end_iv is not None:
                    start_row = cur.iloc[-int(patience) - 1]
                    start_iv = _miv(Kx, int(start_row["batteries"]))
                    if start_iv is not None and start_iv[0] == end_iv[1]:
                        sat = True
                        plateau_cov = int(start_iv[0])
                        sat_proof = (f"monotone-sandwich:B={int(start_row['batteries'])}"
                                     f"->{int(end['batteries'])}")

            threshold = (None if plateau_cov is None else
                         int(math.ceil(float(frac) * int(plateau_cov))))
            knee = None
            minimality = False
            minimality_proof = "unproved"
            order_key = str(order).upper()
            if sat and plateau_cov is not None and plateau_cov > 0 and threshold is not None:
                if order_key == "BK":
                    # B-minimality is decided on Kmax.  Since coverage is
                    # nondecreasing in K, if Kmax at the previous B is below T,
                    # every smaller K is also below T.
                    bvals = sorted(int(x) for x in sub["batteries"].dropna().unique())
                    bstar = None
                    for b in bvals:
                        rmax = sub[(sub["K"] == Kx) & (sub["batteries"] == b)]
                        qmaxb = (_miv(Kx, b) if len(rmax) == 1 else None)
                        if qmaxb is not None and qmaxb[0] >= threshold:
                            bstar = b
                            break
                    if bstar is not None:
                        ib = bvals.index(bstar)
                        b_min_ok = (ib == 0)
                        if ib > 0:
                            prev = sub[(sub["K"] == Kx) & (sub["batteries"] == bvals[ib-1])]
                            qprev = (_miv(Kx, bvals[ib-1]) if len(prev) == 1 else None)
                            b_min_ok = bool(qprev is not None and qprev[1] < threshold)
                        at_b = sub[sub["batteries"] == bstar].sort_values("K")
                        kvals = [int(x) for x in at_b["K"].tolist()]
                        kstar = None
                        for _, rr in at_b.iterrows():
                            iv = _miv(int(rr["K"]), bstar)
                            if iv is not None and iv[0] >= threshold:
                                kstar = int(rr["K"]); break
                        if kstar is not None:
                            ik = kvals.index(kstar)
                            k_min_ok = (ik == 0)
                            if ik > 0:
                                prevk = at_b.iloc[ik-1]
                                qprevk = _miv(int(prevk["K"]), bstar)
                                k_min_ok = bool(qprevk is not None and qprevk[1] < threshold)
                            if b_min_ok and k_min_ok:
                                rr = sub[(sub["K"] == kstar) & (sub["batteries"] == bstar)]
                                if len(rr) == 1:
                                    knee = rr.iloc[0]
                                    minimality = True
                                    minimality_proof = "BK-monotone-boundary-sandwich"
                else:  # KB
                    bmax = int(sub["batteries"].max())
                    kgrid = sorted(int(x) for x in sub["K"].dropna().unique())
                    kstar = None
                    for k in kgrid:
                        rr = sub[(sub["K"] == k) & (sub["batteries"] == bmax)]
                        qk = (_miv(k, bmax) if len(rr) == 1 else None)
                        if qk is not None and qk[0] >= threshold:
                            kstar = k; break
                    if kstar is not None:
                        ik = kgrid.index(kstar)
                        k_min_ok = (ik == 0)
                        if ik > 0:
                            prev = sub[(sub["K"] == kgrid[ik-1]) & (sub["batteries"] == bmax)]
                            qprev = (_miv(kgrid[ik-1], bmax) if len(prev) == 1 else None)
                            k_min_ok = bool(qprev is not None and qprev[1] < threshold)
                        at_k = sub[sub["K"] == kstar].sort_values("batteries")
                        bvals = [int(x) for x in at_k["batteries"].tolist()]
                        bstar = None
                        for _, rr in at_k.iterrows():
                            iv = _miv(kstar, int(rr["batteries"]))
                            if iv is not None and iv[0] >= threshold:
                                bstar = int(rr["batteries"]); break
                        if bstar is not None:
                            ib = bvals.index(bstar)
                            b_min_ok = (ib == 0)
                            if ib > 0:
                                prevb = at_k.iloc[ib-1]
                                qprevb = _miv(kstar, int(prevb["batteries"]))
                                b_min_ok = bool(qprevb is not None and qprevb[1] < threshold)
                            if k_min_ok and b_min_ok:
                                rr = sub[(sub["K"] == kstar) & (sub["batteries"] == bstar)]
                                if len(rr) == 1:
                                    knee = rr.iloc[0]
                                    minimality = True
                                    minimality_proof = "KB-monotone-boundary-sandwich"

            knee_holds = bool(knee is not None and _bool_col(knee, "plan_holds"))
            knee_global = bool(
                knee is not None and _e1_global_certificate_with_provenance(knee.to_dict()))
            knee_cov_cert = bool(knee is not None and _coverage_certificate_flag(knee.to_dict()))
            eligible = bool(sat and minimality and knee is not None and knee_holds and knee_global)
            if not sat:
                status = "uncertified_coverage_plateau"
            elif not minimality or knee is None:
                status = "uncertified_resource_knee"
            elif not knee_global:
                status = "needs_lexicographic_knee_certification"
            elif not knee_holds:
                status = "unsafe_no_validated_candidate"
            else:
                status = "eligible"
            eps_knee, inv, per_inv = _metric_fields(knee, sub)
            cap_hit = bool(sub["stops_cap_hit"].fillna(False).astype(bool).any()) if "stops_cap_hit" in sub else None
            out.append(dict(
                uav=uk,
                plateau_safe=plateau_obs_safe,
                plateau_coverage=plateau_cov,
                plateau_coverage_certified=bool(sat),
                saturation_proof=sat_proof,
                coverage_threshold=threshold,
                sat_reached=bool(sat),
                # Formal E1 is a discrete minimum-resource threshold problem,
                # not a geometric curve-knee detector.  With |I|=8 and rho=.95,
                # ceil(rho*P)=P is mathematically expected and must not invalidate
                # an otherwise certified threshold configuration.
                degenerate_knee=False,
                threshold_equals_plateau=bool(threshold == plateau_cov),
                threshold_rounded_to_full_coverage=bool(
                    threshold == plateau_cov and frac < 1.0 and (plateau_cov or 0) > 0),
                resource_threshold_point_valid=bool(sat and threshold is not None),
                selection_status=status,
                knee_resource_minimality_certified=bool(minimality),
                knee_minimality_proof=minimality_proof,
                knee_global_certificate_available=bool(knee_global),
                knee_coverage_certificate_available=bool(knee_cov_cert),
                knee_K=(int(knee["K"]) if knee is not None else None),
                knee_B=(int(knee["batteries"]) if knee is not None else None),
                knee_safe=(int(knee["safe_served"]) if knee is not None else None),
                knee_coverage_incumbent=(int(knee.get("coverage_incumbent", knee.get("covered", 0)))
                                         if knee is not None else None),
                knee_per_battery=(float(knee["per_battery"])
                                  if knee is not None and pd.notna(knee["per_battery"]) else None),
                knee_energy_per_safe=eps_knee,
                knee_inventory_energy_kWh=inv,
                knee_safe_per_inventory_kWh=per_inv,
                knee_plan_holds=(bool(knee_holds) if knee is not None else None),
                max_stops_requested=(int(sub["max_stops_requested"].iloc[0])
                                     if "max_stops_requested" in sub else None),
                stops_cap_spec=(str(sub["stops_cap_spec"].iloc[0]) if "stops_cap_spec" in sub else None),
                max_stops_effective=(int(sub["max_stops_effective"].iloc[0])
                                     if "max_stops_effective" in sub else
                                     (int(sub["stops_cap"].iloc[0]) if "stops_cap" in sub else None)),
                stops_cap=(int(sub["stops_cap"].iloc[0]) if "stops_cap" in sub else None),
                max_stops_observed=(int(sub["max_stops_observed"].max())
                                    if "max_stops_observed" in sub else None),
                stops_cap_hit_any=cap_hit,
                coverable=(int(sub["coverable_note"].iloc[0]) if "coverable_note" in sub else None),
                source_result_contract=(str(knee.get("result_contract", "")) if knee is not None else ""),
                source_formal_experiment_scheduler_contract=(
                    str(knee.get("formal_experiment_scheduler_contract", "")) if knee is not None else ""),
                source_result_certificate_contract=(
                    str(knee.get("result_certificate_contract", "")) if knee is not None else ""),
                source_formal_proof_contract=(
                    str(knee.get("formal_proof_contract", "")) if knee is not None else ""),
                source_resume_input_sha256=(
                    str(knee.get("resume_input_sha256", "")) if knee is not None else ""),
                knee_frac=frac, knee_order=order_key,
                formal_selection_contract="discrete-coverage-threshold-min-resource-plus-lex-v3"))
            continue

        # Mechanism/legacy selection: observed validation-safe frontier only.
        plateau = int(sub["safe_served"].max())
        Kx = int(sub["K"].max())
        cur = sub[sub["K"] == Kx].sort_values("batteries")
        gains = cur["safe_served"].astype(float).diff().dropna().tolist()
        sat = (None if plateau <= 0 else
               bool(len(gains) >= patience and all(g <= 0 for g in gains[-patience:])))
        thr = frac * plateau - 1e-9
        cand = (sub.iloc[:0].copy() if plateau <= 0 else
                sub[(sub["safe_served"] >= thr) & (sub["batteries"] > 0)])
        knee_holds = None
        unsafe_no_candidate = False
        if len(cand):
            if "plan_holds" not in cand.columns:
                cand = cand.iloc[:0].copy(); unsafe_no_candidate = True
            else:
                ok_mask = cand["plan_holds"].astype("boolean").fillna(False).astype(bool)
                cand = cand[ok_mask].copy(); knee_holds = bool(len(cand))
                if not len(cand): unsafe_no_candidate = True
        keys = ["batteries", "K"] if str(order).upper() == "BK" else ["K", "batteries"]
        knee = cand.sort_values(keys).iloc[0] if len(cand) else None
        b_max = int(sub["batteries"].max())
        near_boundary = bool(knee is not None
                             and int(knee["batteries"]) >= b_max - max(int(patience) - 1, 0))
        degenerate = bool(plateau <= 0 or knee is None or sat is not True or near_boundary)
        eps_knee, inv, per_inv = _metric_fields(knee, sub)
        cap_hit = bool(sub["stops_cap_hit"].any()) if "stops_cap_hit" in sub else None
        out.append(dict(
            uav=uk, plateau_safe=plateau, plateau_coverage=None,
            plateau_coverage_certified=False, saturation_proof="mechanism-observed-safe",
            coverage_threshold=None, sat_reached=sat, degenerate_knee=degenerate,
            selection_status=(("empty_route_pool" if _e1_route_space_is_empty(sub)
                               else "zero_positive_coverage") if plateau <= 0 else
                              ("unsafe_no_validated_candidate" if unsafe_no_candidate else
                               ("eligible" if not degenerate else
                                ("unsaturated_resource_axis" if sat is not True
                                 else "boundary_or_empty_knee")))),
            knee_resource_minimality_certified=False,
            knee_minimality_proof="mechanism-grid-rule",
            knee_global_certificate_available=(bool(_global_certificate_flag(knee.to_dict()))
                                                if knee is not None else False),
            knee_coverage_certificate_available=(bool(_coverage_certificate_flag(knee.to_dict()))
                                                  if knee is not None else False),
            knee_K=(int(knee["K"]) if knee is not None else None),
            knee_B=(int(knee["batteries"]) if knee is not None else None),
            knee_safe=(int(knee["safe_served"]) if knee is not None else None),
            knee_coverage_incumbent=(int(knee.get("coverage_incumbent", knee.get("covered", 0)))
                                     if knee is not None else None),
            knee_per_battery=(float(knee["per_battery"])
                              if knee is not None and pd.notna(knee["per_battery"]) else None),
            knee_energy_per_safe=eps_knee,
            knee_inventory_energy_kWh=inv,
            knee_safe_per_inventory_kWh=per_inv,
            knee_plan_holds=knee_holds,
            max_stops_requested=(int(sub["max_stops_requested"].iloc[0])
                                 if "max_stops_requested" in sub else None),
            stops_cap_spec=(str(sub["stops_cap_spec"].iloc[0]) if "stops_cap_spec" in sub else None),
            max_stops_effective=(int(sub["max_stops_effective"].iloc[0])
                                 if "max_stops_effective" in sub else
                                 (int(sub["stops_cap"].iloc[0]) if "stops_cap" in sub else None)),
            stops_cap=(int(sub["stops_cap"].iloc[0]) if "stops_cap" in sub else None),
            max_stops_observed=(int(sub["max_stops_observed"].max())
                                if "max_stops_observed" in sub else None),
            stops_cap_hit_any=cap_hit,
            coverable=(int(sub["coverable_note"].iloc[0]) if "coverable_note" in sub else None),
            knee_frac=frac, knee_order=str(order).upper(),
            formal_selection_contract="mechanism-observed-safe-v1"))
    return pd.DataFrame(out)


def _pick_selection(sel: pd.DataFrame, metric="safe_per_inventory_kWh"):
    """Select an eligible UAV using an explicit, physically comparable metric.

    ``safe_per_inventory_kWh`` is the default because a battery count is not a
    common resource unit across UAV profiles.  Legacy metrics remain available
    only when requested explicitly.
    """
    warns = []
    if "selection_status" in sel.columns:
        formal_mask = sel.get(
            "formal_selection_contract",
            pd.Series(index=sel.index, dtype=object)).astype(str).str.startswith(
                "discrete-coverage-threshold-min-resource")
        status_eligible = sel["selection_status"].astype(str).eq("eligible")
        # Final freeze re-checks structured proof fields instead of trusting the
        # summary status string.  A tampered/stale CSV cannot become selectable
        # merely by changing selection_status="eligible".
        def _flag(name, default=False):
            if name not in sel.columns:
                return pd.Series(bool(default), index=sel.index, dtype=bool)
            return sel[name].astype("boolean").fillna(False).astype(bool)
        provenance_ok = (
            sel.get("formal_selection_contract", pd.Series("", index=sel.index)).astype(str)
               .eq("discrete-coverage-threshold-min-resource-plus-lex-v3")
            & sel.get("source_result_contract", pd.Series("", index=sel.index)).astype(str).eq(RESULT_CONTRACT)
            & sel.get("source_formal_experiment_scheduler_contract", pd.Series("", index=sel.index)).astype(str)
               .eq(FORMAL_EXPERIMENT_SCHEDULER_CONTRACT)
            & sel.get("source_result_certificate_contract", pd.Series("", index=sel.index)).astype(str)
               .eq(BP.RESULT_CERTIFICATE_CONTRACT)
            & sel.get("source_formal_proof_contract", pd.Series("", index=sel.index)).astype(str)
               .eq(BP.FORMAL_PROOF_CONTRACT))
        formal_proof_ok = (
            _flag("knee_resource_minimality_certified")
            & _flag("knee_global_certificate_available")
            & _flag("knee_coverage_certificate_available")
            & _flag("resource_threshold_point_valid")
            & provenance_ok.astype(bool))
        formal_ok = formal_mask & status_eligible & formal_proof_ok
        legacy_ok = ((~formal_mask) & status_eligible
                     & (sel["degenerate_knee"] != True))  # noqa: E712
        ok = sel[formal_ok | legacy_ok].copy()
    else:
        ok = sel[sel["degenerate_knee"] != True].copy()  # noqa: E712
    if "knee_plan_holds" not in ok.columns:
        return None, ["选型表缺少 knee_plan_holds；缺失或不完整的结果合同 不得自动冻结方案"]
    holds_mask = ok["knee_plan_holds"].astype("boolean").fillna(False).astype(bool)
    ok = ok[holds_mask].copy()
    if not len(ok):
        return None, ["没有同时满足资源最小性/global lex 证书与 validation 风险门的正式阈值配置；方案保持未选择状态"]
    metric = str(metric)
    metric_map = {
        "safe_per_inventory_kWh": ("knee_safe_per_inventory_kWh", False),
        "per_battery": ("knee_per_battery", False),
        "energy_per_safe": ("knee_energy_per_safe", True),
        "max_safe": ("knee_safe", False),
    }
    if metric not in metric_map:
        raise SystemExit(f"未知 E1 selection metric={metric}; 可选 {sorted(metric_map)}")
    col, ascending = metric_map[metric]
    if col not in ok.columns or not ok[col].notna().any():
        return None, [f"可选档位缺少选型指标 {col}，请用新结果合同 重跑 E1"]
    ok = ok[ok[col].notna()]
    sort_cols = [col]
    sort_asc = [ascending]
    if col != "knee_energy_per_safe" and "knee_energy_per_safe" in ok.columns:
        sort_cols.append("knee_energy_per_safe")
        sort_asc.append(True)
    pick = ok.sort_values(sort_cols, ascending=sort_asc, na_position="last").iloc[0]
    pick = pick.copy()
    pick["selected_by_metric"] = metric
    if "sat_reached" in sel.columns and not bool(pick.get("sat_reached", True)):
        warns.append(f"{pick.uav} 档 B 轴触顶未饱和(sat_reached=False) —— 膝点为预算内最优, 建议补跑更大 --e1-b-cap")
    if not bool(pick.get("knee_plan_holds", False)):
        raise RuntimeError("内部错误：_pick_selection 选择了 validation 未通过的膝点")
    return pick, warns


def _resolve_e2_config(args):
    """【更新】E2/A 选型自动回填(导师要求: 不再手抄 <新选型>)。
    规则:
      --uav auto(新默认) ⇒ 三参数(uav,K,B)全部取自 E1 选型, 优先级:
        ① args._e1_sel_df(--exp all 同进程内存直通) ② E1_selection.csv ③ E1_frontier.csv 现算;
        皆无 ⇒ SystemExit(先跑 E1, 或显式三参数)。degenerate 选型【拒绝自动采用】(不静默采信废选型)。
        auto 下用户误给的 --k/--batteries 被忽略并警告(避免半自动歧义)。
      --uav 显式档位 ⇒ 全手动(K 默认 3, B 默认 2K, 旧语义)。

    Formal protocol note: manual values may be used for diagnostic E2 runs, but
    they are not an E1-certified freeze and therefore cannot authorize A-suite
    publication semantics or consumption of the one-shot final test.
    """
    setattr(args, "_e1_formal_freeze_verified", False)
    setattr(args, "_e1_formal_freeze_sha256", "")
    setattr(args, "_e1_config_source", "unresolved")
    if str(args.uav).lower() != "auto":
        setattr(args, "_e1_config_source", "manual-unfrozen")
        if args.k is None:
            args.k = 3
        return f"手动(--uav {args.uav}, K={args.k}, B={args.batteries if args.batteries is not None else 2*int(args.k)})"
    if args.k is not None or args.batteries is not None:
        log.warning("--uav auto 下忽略手给的 --k/--batteries(全自动语义); 需手动请显式 --uav <档位>。")
    sel, src = getattr(args, "_e1_sel_df", None), "内存(--exp all 直通)"
    if sel is None:
        p1 = RESULTS / "E1_frontier" / "E1_selection.csv"
        p2 = RESULTS / "E1_frontier" / "E1_frontier.csv"
        if p1.is_file():
            sel, src = pd.read_csv(p1, encoding="utf-8-sig"), f"{p1.name}"
        elif p2.is_file():
            sel, src = e1_select_from_df(
                pd.read_csv(p2, encoding="utf-8-sig"),
                frac=float(getattr(args, "knee_frac", 0.95)),
                order=str(getattr(args, "knee_order", "BK")),
                patience=max(1, int(getattr(args, "e1_sat_patience", 3)))), f"{p2.name}(现算选型)"
        else:
            raise SystemExit("[E2/A 自动选型] 找不到 E1_selection.csv 或 E1_frontier.csv —— "
                             "请先跑 `--exp E1_frontier`(及可选 `--exp E1_select`), "
                             "或显式给 --uav <档位> --k <..> --batteries <..>。")
    pick, warns = _pick_selection(sel, metric=getattr(args, "selection_metric",
                                                       "safe_per_inventory_kWh"))
    if pick is not None and str(getattr(args, "study_mode", "mechanism")).lower() == "formal":
        expected_e1 = str(getattr(args, "_expected_e1_resume_input_sha256", ""))
        got_e1 = str(pick.get("source_resume_input_sha256", ""))
        if not expected_e1 or got_e1 != expected_e1:
            pick = None
            warns = [
                "formal E1 freeze provenance does not match the current binary64 E1 instance; "
                "fresh --resume off E1 is required"]
    if pick is None:
        raise SystemExit("[E2/A 自动选型] " + warns[0] +
                         "; 或显式 --uav/--k/--batteries 强行继续(自担口径责任)。")
    for w in warns:
        log.warning("[E2/A 自动选型] %s", w)
    args.uav, args.k, args.batteries = str(pick.uav), int(pick.knee_K), int(pick.knee_B)
    setattr(args, "_e1_config_source", f"e1-selection:{src}")
    if str(getattr(args, "study_mode", "mechanism")).lower() == "formal":
        # _pick_selection has already revalidated all structured proof/provenance
        # fields; bind the exact selected resource point for downstream gates.
        setattr(args, "_e1_formal_freeze_verified", True)
        setattr(args, "_e1_formal_freeze_sha256", _e1_freeze_fingerprint(pick))
    msg = (f"自动消费 E1 选型[{src}]: uav={args.uav} K={args.k} B={args.batteries} "
           f"(metric={getattr(args, 'selection_metric', 'safe_per_inventory_kWh')}, "
           f"safe_per_inventory_kWh={pick.get('knee_safe_per_inventory_kWh')})")
    print(f"[E2/A 回填] {msg}")
    return msg


def E1_select(args):
    """只读 E1 frontier 生成可审计选型表；全退化时不再打印可误用的占位命令。"""
    csvp = Path(args.e1_csv) if args.e1_csv else (RESULTS / "E1_frontier" / "E1_frontier.csv")
    if not csvp.is_file():
        raise SystemExit(f"E1_select: 找不到 {csvp}(先跑 --exp E1_frontier, 或用 --e1-csv 指定)")
    df = pd.read_csv(csvp)
    sel = e1_select_from_df(df, frac=args.knee_frac, order=args.knee_order,
                            patience=max(1, int(args.e1_sat_patience)))
    metric = getattr(args, "selection_metric", "safe_per_inventory_kWh")
    pick, warns = _pick_selection(sel, metric=metric)
    sel = sel.copy()
    sel["selection_metric"] = metric
    sel["selected"] = False
    if pick is not None:
        sel.loc[sel["uav"].astype(str) == str(pick.uav), "selected"] = True
    target_dir = csvp.parent
    try:
        _save(sel, target_dir, "E1_selection.csv")
        EU.write_run_manifest(target_dir, "E1_select", args, input_paths=[csvp],
                              extra={"result_contract": RESULT_CONTRACT, "selection_metric": metric,
                                     "selection_valid": pick is not None})
    except OSError:
        target_dir = RESULTS / "E1_frontier"
        _save(sel, target_dir, "E1_selection.csv")
        EU.write_run_manifest(target_dir, "E1_select", args, input_paths=[csvp],
                              extra={"result_contract": RESULT_CONTRACT, "selection_metric": metric,
                                     "selection_valid": pick is not None})
    print("\n[E1_select] 逐 UAV 膝点、平台状态与统一资源指标:")
    print(sel.to_string(index=False))
    bad = sel[sel["selection_status"].astype(str) != "eligible"].copy()
    if len(bad):
        empty = bad[bad["selection_status"].astype(str) == "empty_route_pool"]
        nonempty_bad = bad[bad["selection_status"].astype(str) != "empty_route_pool"]
        if len(empty):
            print(f"\n⚠ 路线池为空的档位: {list(empty.uav)}。资源层未启动；扩大 --e1-b-cap 不会改变结果。")
        if len(nonempty_bad):
            hard = nonempty_bad[
                nonempty_bad["saturation_proof"].astype(str).eq("hard-coverable-cap")
            ] if "saturation_proof" in nonempty_bad.columns else nonempty_bad.iloc[0:0]
            other = nonempty_bad.drop(index=hard.index)
            if len(hard):
                print(f"\n⚠ hard-coverable-cap 已证明但资源 knee 未闭合: {list(hard.uav)}。"
                      "电池轴结构上已关闭；不要扩大 --e1-b-cap，运行 --exp E1_knee_refine "
                      "证明 threshold predecessor。")
            if len(other):
                print(f"\n⚠ 资源前沿未饱和/未证明的档位: {list(other.uav)}。"
                      "仅这些非 hard-cap 档位才需要检查资源网格边界。")
    capped = sel[sel["stops_cap_hit_any"] == True]      # noqa: E712
    if len(capped):
        print(f"⚠ stops_cap_hit: {list(capped.uav)}；需扩大 --stops-cap 后验证结论稳定性。")
    if pick is None:
        print("\n[E2/A 回填] 已拒绝：所有档位均未形成可信平台。E1_selection.csv 已保存诊断，未输出占位运行命令。")
        return sel
    for w in warns:
        print(f"⚠ {w}")
    print(f"\n[E2/A 回填] 按 {metric} 选出 uav={pick.uav}, K={int(pick.knee_K)}, "
          f"B={int(pick.knee_B)} (safe_per_inventory_kWh={pick.get('knee_safe_per_inventory_kWh')}, "
          f"energy_per_safe={pick.get('knee_energy_per_safe')})")
    print("  E2/A 默认 --uav auto 自动消费该表；显式命令仅供可追溯覆盖。")
    print(f"  python step13_experiment_model.py --exp E2_robust --uav {pick.uav} "
          f"--k {int(pick.knee_K)} --batteries {int(pick.knee_B)} "
          f"--e2-quantiles 0.2,0.35,0.5,0.65,0.8 --replay-n 400")
    print(f"  python step14_experiment_algorithm.py --exp A1_accuracy --n-list 20,40,70 "
          f"--uav {pick.uav} --k {int(pick.knee_K)} --batteries {int(pick.knee_B)}")
    print(f"  python step14_experiment_algorithm.py --exp A2_speed --a2-n 20,40 "
          f"--a2-dtau 15,10,5 --uav {pick.uav}")
    return sel


def _frozen_plan_fingerprint(chosen) -> str:
    """Collision-free binary64 identity of the frozen route/resource plan.

    The old v12 implementation rounded physical values to 9 decimals before
    hashing, so distinct binary64 plans could share one final-test identity.
    Formal freeze now binds exact route semantics plus exact assignment/service
    state through the same binary64 fingerprint machinery used by the solver.
    """
    rows = []
    for c in chosen or []:
        try:
            route_sig = BP._exact_route_signature(c)
        except Exception:
            route_sig = ("fallback-route", BP._state_fp({
                "tau": c.get("tau"), "h": c.get("h"),
                "tids": tuple(c.get("tids", ())),
                "route": c.get("route"), "ship": c.get("ship"), "wx": c.get("wx")}))
        try:
            semantics = BP._column_semantics_fp(c)
        except Exception:
            # Tests/legacy diagnostic objects may omit formal-only fields; the
            # fallback is still exact-as-binary64 and never rounds values.
            semantics = ("fallback-column-semantics", BP._state_fp({
                "E_plan_Wh": c.get("E_plan_Wh", c.get("E0")),
                "E_soc_required_Wh": c.get("E_soc_required_Wh"),
                "resource_intervals": c.get("resource_intervals"),
                "ordered_tids": c.get("ordered_tids", c.get("tids")),
            }))
        rows.append((
            route_sig,
            semantics,
            ("uav_id", int(c.get("uav_id", -1))),
            ("battery_group", int(c.get("battery_group", -1))),
            ("turnaround_before", BP._state_fp(c.get("turnaround_before"))),
            ("post_service_mode", str(c.get("post_service_mode", ""))),
            ("post_service_interval", BP._state_fp(c.get("post_service_interval"))),
        ))
    rows.sort(key=repr)
    raw = repr(BP._state_fp(tuple(rows)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _e1_freeze_fingerprint(pick) -> str:
    """Bind the structured E1 formal selection used to authorize E2/A."""
    payload = dict(
        result_contract=RESULT_CONTRACT,
        scheduler_contract=FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
        result_certificate_contract=BP.RESULT_CERTIFICATE_CONTRACT,
        formal_proof_contract=BP.FORMAL_PROOF_CONTRACT,
        model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
        uav=str(pick.get("uav")),
        K=int(pick.get("knee_K")),
        batteries=int(pick.get("knee_B")),
        coverage_threshold=pick.get("coverage_threshold"),
        plateau_coverage=pick.get("plateau_coverage"),
        knee_resource_minimality_certified=bool(pick.get("knee_resource_minimality_certified")),
        knee_global_certificate_available=bool(pick.get("knee_global_certificate_available")),
        knee_coverage_certificate_available=bool(pick.get("knee_coverage_certificate_available")),
        knee_plan_holds=bool(pick.get("knee_plan_holds")),
        source_resume_input_sha256=str(pick.get("source_resume_input_sha256", "")),
    )
    return hashlib.sha256(repr(BP._state_fp(payload)).encode("utf-8")).hexdigest()


# 更新(作者定案): E2 = 分布鲁棒 vs 其他鲁棒/随机方法。adaptive 已按作者要求从实验集移除
# (机制代码保留 dormant, 见 历史变更记录); 新增 box(经典 RO 支持集端点)。
# 统一形式(与 route_feasible_at_h 逐字对应): 同一 (a,b) 线性化 + 七种裕度算子 M_ρ(见 doc_model §15)。
_E2_CRITERIA = [
    ("nominal",   "drcc",   "nominal",     "non-robust"),
    ("gaussian",  "drcc",   "gaussian",    "parametric-CC"),
    ("SAA",       "saa",    None,          "sample-CC"),
    ("box",       "box",    None,          "RO-support"),
    ("budget_G2", "budget", None,          "RO-ellipsoid(G=2)"),
    ("cantelli",  "drcc",   "cantelli",    "moment-DRO"),
    ("vp",        "drcc",   "vp_unimodal", "moment-DRO-unimodal(main)"),
]

_E2_DRCC_RECOURSE_CRITERIA = {
    name for name, chance_mode, _kmode, _klass in _E2_CRITERIA
    if chance_mode == "drcc"
}


def _resolve_e2_criteria_for_recourse(args):
    """Resolve E2 methods without comparing different physical recourse policies.

    ``wait_and_speed`` is currently certified only for the DRCC-family oracle.
    SAA/box/budget remain valid wait-only baselines, but must never appear as
    zero-coverage rows caused merely by an unsupported speed-recourse formulation.
    """
    spec = str(getattr(args, "e2_criteria", "recourse_compatible")).strip()
    all_names = {name for name, *_ in _E2_CRITERIA}
    if spec.lower() in {"recourse_compatible", "auto"}:
        selected = (_E2_DRCC_RECOURSE_CRITERIA
                    if str(getattr(args, "time_recourse", "wait_and_speed")) == "wait_and_speed"
                    else all_names)
    elif spec.lower() == "all":
        selected = set(all_names)
    else:
        selected = {x.strip() for x in spec.split(",") if x.strip()}
    unknown = selected - all_names
    if unknown:
        raise SystemExit(
            f"E2 未知判据 {sorted(unknown)}; 可选 {sorted(all_names)} "
            "或 recourse_compatible/all")
    if str(getattr(args, "time_recourse", "wait_and_speed")) == "wait_and_speed":
        unsupported = selected - _E2_DRCC_RECOURSE_CRITERIA
        if unsupported:
            raise SystemExit(
                "E2 wait_and_speed 当前没有为以下基线认证 return-speed recourse: "
                f"{sorted(unsupported)}。请使用 --e2-criteria recourse_compatible "
                "(仅 nominal/gaussian/cantelli/vp)，或另跑 --time-recourse wait_only "
                "--e2-criteria all 作为统一 wait-only benchmark。")
    return set(selected)



def _binary64_equal(a, b) -> bool:
    """Exact equality for persisted formal scalar identities; never tolerance-based."""
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return False
    return bool(math.isfinite(fa) and math.isfinite(fb) and fa.hex() == fb.hex())


def _validate_formal_protocol_overrides(args) -> None:
    """Diagnostic overrides may never weaken the formal confirmatory protocol."""
    if str(getattr(args, "study_mode", "mechanism")).lower() != "formal":
        return
    if bool(getattr(args, "allow_incomplete_results", False)):
        raise SystemExit("formal protocol forbids --allow-incomplete-results; complete E2 validation matrix is mandatory before final-test freeze.")
    if bool(getattr(args, "allow_final_test_rerun", False)):
        raise SystemExit("formal protocol forbids --allow-final-test-rerun; the confirmatory final test is one-time only.")


def _formal_e2_final_test_authorized(args) -> bool:
    """True only when formal final-test consumption has E1 freeze + complete E2 matrix."""
    if str(getattr(args, "study_mode", "mechanism")).lower() != "formal":
        return True
    sha = str(getattr(args, "_e1_formal_freeze_sha256", ""))
    return bool(getattr(args, "_e1_formal_freeze_verified", False)
                and getattr(args, "_e2_matrix_completion_verified", False)
                and getattr(args, "_formal_sample_hashes_verified", False)
                and not bool(getattr(args, "allow_incomplete_results", False))
                and not bool(getattr(args, "allow_final_test_rerun", False))
                and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha))


def _load_final_test_metadata_pre_freeze(path, *, mmsi):
    """Load only provenance/timing metadata before the E2 candidate is frozen."""
    import step15_replay as RP
    before = EU.sha256_file(path)
    md = RP.load_sample_metadata(path, mmsi=mmsi, formal=True, expected_split="test")
    after = EU.sha256_file(path)
    if not before or before != after:
        raise SystemExit("final-test file changed while pre-freeze metadata were being validated; fail closed.")
    md.attrs["source_sha256"] = str(after)
    return md


def _bind_formal_sample_hashes(args, *, final_test_metadata=None) -> dict:
    """Freeze train/validation/test byte identities after no-leak metadata validation."""
    paths = {
        "train": getattr(args, "xi_train_samples", None),
        "validation": getattr(args, "validation_samples", None),
        "test": getattr(args, "final_test_samples", None),
    }
    hashes = {k: (EU.sha256_file(v) if v is not None else None) for k, v in paths.items()}
    if any(not h or len(str(h)) != 64 for h in hashes.values()):
        raise SystemExit("formal sample hash binding failed; train/validation/test must all be present and hashable.")
    if final_test_metadata is not None:
        meta_hash = str(final_test_metadata.attrs.get("source_sha256", ""))
        if meta_hash != str(hashes["test"]):
            raise SystemExit("final-test bytes changed after metadata validation; independent-holdout proof invalid.")
    setattr(args, "_formal_sample_hashes_sha256", dict(hashes))
    setattr(args, "_formal_sample_hashes_verified", True)
    return hashes


def _verify_formal_sample_hashes_unchanged(args) -> dict:
    """Reject any train/validation/test byte change after formal provenance validation."""
    if str(getattr(args, "study_mode", "mechanism")).lower() != "formal":
        return {}
    if not bool(getattr(args, "_formal_sample_hashes_verified", False)):
        raise SystemExit("formal sample hashes were not frozen before final-test authorization.")
    expected = dict(getattr(args, "_formal_sample_hashes_sha256", {}) or {})
    paths = {"train": getattr(args, "xi_train_samples", None),
             "validation": getattr(args, "validation_samples", None),
             "test": getattr(args, "final_test_samples", None)}
    for label, path in paths.items():
        want = str(expected.get(label, ""))
        got = str(EU.sha256_file(path) or "") if path is not None else ""
        if got != want:
            raise SystemExit(f"formal {label} sample file changed after provenance/freeze validation; refuse final test.")
    return expected


def _parse_e2_quantiles(spec) -> tuple[float, ...]:
    """Parse a predeclared finite [0,1] quantile grid with exact binary64 uniqueness."""
    qs = tuple(float(x) for x in str(spec).split(",") if str(x).strip())
    if not qs:
        raise SystemExit("E2 quantile grid is empty.")
    if any((not math.isfinite(q)) or q < 0.0 or q > 1.0 for q in qs):
        raise SystemExit(f"E2 quantiles must be finite values in [0,1], got {qs!r}.")
    if len({q.hex() for q in qs}) != len(qs):
        raise SystemExit(f"E2 quantile grid contains duplicate binary64 values: {qs!r}.")
    return qs


def _select_e2_validation_candidate(dfr: pd.DataFrame, quantiles) -> pd.Series | None:
    """Freeze one predeclared E2 candidate using validation only.

    The final test is not a leaderboard.  We first restrict to the harshest configured
    weather quantile, then require a complete validation pass, maximize safe coverage and
    total coverage, and finally minimize energy.  Criterion name is the deterministic tie-break.
    """
    if dfr is None or dfr.empty:
        return None
    q_target = max(float(q) for q in quantiles)
    sub = dfr.copy()
    ok = (sub.get("run_status", pd.Series(index=sub.index, dtype=object)).astype(str).eq("ok")
          & pd.to_numeric(sub.get("q"), errors="coerce").map(lambda q: _binary64_equal(q, q_target))
          & sub.get("holds", pd.Series(False, index=sub.index)).astype("boolean").fillna(False).astype(bool)
          & pd.to_numeric(sub.get("n_missing_replay", 1), errors="coerce").fillna(1).eq(0)
          & pd.to_numeric(sub.get("covered", 0), errors="coerce").fillna(0).gt(0))
    formal = bool("study_mode" in sub.columns and len(sub)
                  and sub["study_mode"].astype(str).str.lower().eq("formal").all())
    if formal:
        # Formal final-test freezing requires the complete lexicographic physical
        # certificate.  A time-limit incumbent at q_max is not an eligible plan.
        cert = sub.apply(lambda rr: _global_certificate_flag(rr.to_dict()), axis=1)
        freeze_ok = (
            sub.get("e1_formal_freeze_verified", pd.Series(False, index=sub.index))
               .astype("boolean").fillna(False).astype(bool)
            & sub.get("e1_formal_freeze_sha256", pd.Series("", index=sub.index)).astype(str)
               .str.fullmatch(r"[0-9a-f]{64}"))
        ok = ok & cert.astype(bool) & freeze_ok.astype(bool)
    sub = sub[ok].copy()
    if sub.empty:
        return None
    sub["_safe"] = pd.to_numeric(sub.get("safe_served", pd.Series(index=sub.index, dtype=float)), errors="coerce").fillna(-1)
    sub["_covered"] = pd.to_numeric(sub.get("covered", pd.Series(index=sub.index, dtype=float)), errors="coerce").fillna(-1)
    sub["_eps"] = pd.to_numeric(sub.get("energy_per_safe", pd.Series(index=sub.index, dtype=float)), errors="coerce").fillna(np.inf)
    sub["_energy"] = pd.to_numeric(sub.get("energy_Wh", pd.Series(index=sub.index, dtype=float)), errors="coerce").fillna(np.inf)
    return sub.sort_values(
        ["_safe", "_covered", "_eps", "_energy", "criterion"],
        ascending=[False, False, True, True, True]).iloc[0]


def _e2_final_test_record(candidate, turbines, lat0lon0, track_csv, xi_amb, wx_df,
                          p_u, wamb, args, K: int, B: int, cap_u: int,
                          t_swap: float, t_launch: float, frozen_chosen=None,
                          invocation_count: int = 1):
    """Audit the exact validation-frozen plan on final test exactly once.

    In a fresh run the in-memory chosen plan is reused directly.  On resume, the solver may
    reconstruct it from validation inputs, but the canonical route/resource fingerprint must
    match before any final-test row is read.
    """
    q = float(candidate["q"]); name = str(candidate["criterion"])
    criteria = {n: (cm, km, kl) for n, cm, km, kl in _E2_CRITERIA}
    if name not in criteria:
        raise RuntimeError(f"冻结的 E2 criterion 不存在: {name}")
    cmode, kmode, klass = criteria[name]
    opts, reach, kind, T_eff, wx0 = build_launch_options(
        turbines, lat0lon0, track_csv, xi_amb, wx_df, args.window_min, args.dtau_min,
        float(args._pair_radius_m), hs_quantile=q,
        track_start_min=args.track_start_min, allow_synth=args.allow_synth,
        infarm_radius_m=getattr(args, "infarm_radius", None),
        predictor=getattr(args, "recovery_predictor", "cv_noleak"),
        weather_alignment="representative_quantile",
        formal=(getattr(args, "study_mode", "mechanism") == "formal"))
    r = None
    if frozen_chosen is None:
        r = BP.solve_fleet_anytime(
            reach, opts, p_u, xi_amb, K, T_eff,
            deck_delta_min=args.deck_delta_min, t_swap_min=t_swap,
            landing_clear_min=args.landing_clear_min,
            swap_station_capacity=args.swap_stations,
            quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
            battery_reuse_mode=args.battery_reuse_mode,
            max_stops=cap_u, weather_unc=wamb,
            chance_mode=cmode, kappa_mode=(kmode or "vp_unimodal"),
            budget_gamma=2.0, batteries=B,
            deck_mode=args.deck_mode, t_launch_min=t_launch,
            pool_h_mode=getattr(args, "pool_h", "pareto"),
            solver_mode="exact-branch-price-cut",
            **_anytime_solver_kwargs(args))
        chosen = r["chosen"]
    else:
        chosen = list(frozen_chosen)
    expected_fp = str(candidate.get("plan_fingerprint", ""))
    actual_fp = _frozen_plan_fingerprint(chosen)
    if not expected_fp or expected_fp.lower() == "nan":
        raise RuntimeError("冻结候选缺少 plan_fingerprint；旧结果不得消费 final test，请重跑validation。")
    if actual_fp != expected_fp:
        raise RuntimeError(f"冻结方案指纹不一致: expected={expected_fp}, actual={actual_fp}")
    final_mode = ("real_joint_final_test"
                  if getattr(args, "final_weather_mode", "real") == "real"
                  else "real_xi_final_test")
    if getattr(args, "study_mode", "mechanism") == "formal":
        _verify_formal_sample_hashes_unchanged(args)
    rp = _replay_columns(
        chosen, p_u, xi_amb, p_u.eps_E,
        n_per=args.replay_n, seed=2707, wamb=wamb,
        validation_mode=final_mode,
        real_samples_csv=args.final_test_samples,
        weather_sample_mode=getattr(args, "final_weather_mode", "real"),
        holdout_disjointness_verified=bool(
            getattr(args, "_holdout_disjointness_verified", False)))
    budget = RM.mission_eps_budget(
        p_u, wamb is not None or getattr(args, "final_weather_mode", "real") == "real")
    test_hash = EU.sha256_file(args.final_test_samples) or "none"
    return dict(
        result_contract=RESULT_CONTRACT,
        formal_experiment_scheduler_contract=FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
        result_certificate_contract=BP.RESULT_CERTIFICATE_CONTRACT,
        formal_proof_contract=BP.FORMAL_PROOF_CONTRACT,
        model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
        resume_input_sha256=str(getattr(args, "_resume_input_sha256", "")),
        e1_formal_freeze_verified=bool(getattr(args, "_e1_formal_freeze_verified", False)),
        e1_formal_freeze_sha256=str(getattr(args, "_e1_formal_freeze_sha256", "")),
        e1_config_source=str(getattr(args, "_e1_config_source", "unresolved")),
        e2_matrix_completion_verified=bool(getattr(args, "_e2_matrix_completion_verified", False)),
        selection_rule=("validation-only; harshest configured q; holds=True; "
                        "max safe_served, max covered, min energy, criterion tie-break"),
        selected_criterion=name, selected_class=klass, selected_q=q,
        validation_safe_served=int(candidate["safe_served"]),
        validation_covered=int(candidate["covered"]),
        validation_upper95=float(candidate["emp_viol_upper95"])
        if pd.notna(candidate.get("emp_viol_upper95")) else None,
        validation_holds=bool(candidate["holds"]),
        project_name=PROJECT_NAME,
        train_samples_sha256=(EU.sha256_file(getattr(args, "xi_train_samples", None)) or "none"),
        validation_samples_sha256=(EU.sha256_file(getattr(args, "validation_samples", None)) or "none"),
        frozen_plan_fingerprint=actual_fp,
        frozen_plan_covered=int(candidate["covered"]),
        frozen_plan_flights=int(candidate["flights"]),
        frozen_plan_energy_Wh=float(candidate["energy_Wh"]),
        final_test_samples_sha256=str(test_hash),
        final_test_invocations=int(invocation_count),
        final_test_validation_type=rp.get("validation_type"),
        final_test_n_total=int(rp.get("n_test_total", 0)),
        final_test_n_viol=int(rp.get("n_viol_total", 0)),
        final_test_upper95=rp.get("upper95"),
        final_test_n_missing=int(rp.get("n_missing", 0)),
        final_test_ci_method=rp.get("ci_method"),
        final_test_formal_reliability_claim_eligible=bool(
            rp.get("formal_reliability_claim_eligible", False)),
        final_test_holds=bool(
            rp.get("formal_reliability_claim_eligible", False)
            and rp.get("allocation_budget_holds") is True
            and int(rp.get("n_missing", 0)) == 0),
        evidence_scope=rp.get("evidence_scope", "mechanism-or-partial-evidence"),
        uav=str(args.uav), K=int(K), batteries=int(B), max_stops=int(cap_u),
        track=str(kind), T_eff_min=float(T_eff),
        final_test_rerun_authorized=bool(getattr(args, "allow_final_test_rerun", False)))


def E2_robust(turbines, lat0lon0, track_csv, xi_amb, wx_df, p_base, wamb, outdir, args):
    """【E2·方法对照, 更新 重构】本文 VP-DRCC vs 六类鲁棒/随机基线, 7 判据 × 多起始窗(Hs 分位)。
    UAV/K/B 配置消费 E1 选型(--uav/--k/--batteries; E1 结果落地前用默认 S/3/2K 起步)。
    每行报 covered / safe_served / safe_ratio / emp_viol / plan_holds / max_col_viol +
    回放记账(n_missing>0 ⇒ holds 只覆盖部分计划, 不再静默)。"""
    _validate_formal_protocol_overrides(args)
    qs = _parse_e2_quantiles(args.e2_quantiles)
    K = int(args.k)
    p_u = M.apply_uav_profile(p_base, args.uav)
    t_swap, t_launch = _uav_deck(args, args.uav)
    # 更新: E2 与 E1 共用同一 stops_cap 解析(--stops-cap) —— 否则 E1 以 cap=9 选出的
    # (uav,K,B) 回填到 E2 时若退回 4, 两实验口径错位(E1 的档位能力在 E2 里被重新删失)。
    cap_u = _stops_cap(args.stops_cap, p_u, xi_amb, args.max_stops)
    B = int(args.batteries) if args.batteries is not None else 2 * K
    selected_criteria = _resolve_e2_criteria_for_recourse(args)
    expected_jobs = [(name, float(q)) for q in qs for name in selected_criteria]
    _resume_input_sha256 = _resume_context_sha256(
        turbines=turbines, lat0lon0=lat0lon0, track_csv=Path(track_csv) if track_csv else None,
        xi_ambiguity=xi_amb, weather_dataframe=wx_df, params=p_u,
        weather_uncertainty=wamb)
    setattr(args, "_resume_input_sha256", _resume_input_sha256)
    # ---- 更新 任务4: 断点续跑(键=criterion×q; q 键 + exact base inputs 共同绑定实例) ----
    _sig = dict(eps=p_u.eps_E, dtau_min=args.dtau_min, deck_delta_min=args.deck_delta_min,
                physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
                route_identity_contract=BP.ROUTE_IDENTITY_CONTRACT,
                model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
                resume_input_sha256=_resume_input_sha256,
                deck_mode=args.deck_mode, replay_n=args.replay_n, result_contract=RESULT_CONTRACT,
                formal_experiment_scheduler_contract=FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
                saa_source=RM.SAA_SOURCE, uav=args.uav, K=K, batteries=B,
                max_stops=cap_u,
                xi_source=getattr(args, "_xi_source", "unknown"),
                xi_mmsi=getattr(args, "_resolved_xi_mmsi", "ALL"),
                xi_predictor=getattr(args, "_resolved_xi_predictor", "unknown"),
                xi_predictor_contract=getattr(args, "_resolved_xi_predictor_contract", "unknown"),
                weather_uncertainty_source=getattr(args, "_weather_uncertainty_source", "off"),
                weather_risk_contract="vector-route-scalar-landing",
                route_airspeed_contract="per-leg-along-cross-projection",
                pair_radius_m=round(float(getattr(args, "_pair_radius_m", -1.0)), 1),
                soc_correction=getattr(p_u, "soc_correction", "none"),
                soc_risk_allocation=getattr(p_u, "soc_risk_allocation", "fixed"),
                time_recourse_mode=getattr(p_u, "time_recourse_mode", "wait_only"),
                time_contract_id=RM.time_contract_for(p_u),
                speed_is_recourse=bool(getattr(p_u, "speed_adjustable", False)),
                return_speed_recourse_contract=(RM.SPEED_RECOURSE_CONTRACT if getattr(p_u, "speed_adjustable", False) else None),
                energy_recourse_contract=(RM.ENERGY_SPEED_RECOURSE_CONTRACT if getattr(p_u, "speed_adjustable", False) else None),
                geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
                predictor=getattr(args, "recovery_predictor", "cv_noleak"),
                pool_h=getattr(args, "pool_h", "pareto"),
                validation_mode=getattr(args, "validation_mode", "real_validation"),
                validation_samples_hash=(EU.sha256_file(getattr(args, "validation_samples", None)) or "none"),
                study_mode=getattr(args, "study_mode", "mechanism"),
                e1_formal_freeze_verified=bool(getattr(args, "_e1_formal_freeze_verified", False)),
                e1_formal_freeze_sha256=str(getattr(args, "_e1_formal_freeze_sha256", "")),
                e1_config_source=str(getattr(args, "_e1_config_source", "unresolved")),
                quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                swap_station_capacity=int(getattr(args, "swap_stations", 1)),
                battery_reuse_mode=getattr(args, "battery_reuse_mode", "exact_soc"),
                recovery_target_model=str(getattr(p_u, "recovery_target_model", "discrete_horizon_ship_prediction")),
                terminal_sensor_error_mode=str(getattr(p_u, "terminal_sensor_error_mode", "out_of_scope")))
    plan_cache = {}
    raw, _done = _resume_load(outdir, "E2_robust_raw.csv", ["criterion", "q"],
                              _sig, getattr(args, "resume", "on"),
                              completed_status_col="run_status",
                              completed_values=("ok", "completed"))
    for q in qs:
        opts, reach, kind, T_eff, wx0 = build_launch_options(
            turbines, lat0lon0, track_csv, xi_amb, wx_df, args.window_min, args.dtau_min,
            float(args._pair_radius_m), hs_quantile=q,   # 更新/43: 解析后半径(main 必设; 缺失即炸, 不给魔数留门)
            track_start_min=args.track_start_min, allow_synth=args.allow_synth,
            infarm_radius_m=getattr(args, "infarm_radius", None),   # 更新
            predictor=getattr(args, "recovery_predictor", "cv_noleak"),
            weather_alignment="representative_quantile",
        formal=(getattr(args, "study_mode", "mechanism") == "formal"))   # E2 q 轴本身即天气严重度情景
        prov = _provenance(args, kind, T_eff, reach, p_u, t_swap, t_launch,
                           max_stops_val=cap_u, wx0=wx0)
        for name, cmode, kmode, klass in _E2_CRITERIA:
            if name not in selected_criteria:
                continue
            if (str(name), str(float(q))) in _done:               # 更新: 已完成 → 跳过
                log.info("E2 q=%.2f %-10s: resume 跳过(已完成)", q, name)
                continue
            # Remove a previous failed checkpoint for this key before retrying;
            # successful checkpoints are skipped above and therefore never duplicated.
            raw = [row for row in raw
                   if not (str(row.get("criterion")) == str(name)
                           and str(float(row.get("q"))) == str(float(q)))]
            started = time.time()
            try:
                r = BP.solve_fleet_anytime(
                    reach, opts, p_u, xi_amb, K, T_eff,
                    deck_delta_min=args.deck_delta_min, t_swap_min=t_swap,
                    landing_clear_min=args.landing_clear_min,
                    swap_station_capacity=args.swap_stations,
                    quick_inspection_capacity=int(getattr(args, "quick_inspection_capacity", 1)),
                    battery_reuse_mode=args.battery_reuse_mode,
                    max_stops=cap_u, weather_unc=wamb,
                    chance_mode=cmode, kappa_mode=(kmode or "vp_unimodal"),
                    budget_gamma=2.0, batteries=B,
                    deck_mode=args.deck_mode, t_launch_min=t_launch,
                    pool_h_mode=getattr(args, "pool_h", "pareto"),
                    solver_mode="exact-branch-price-cut",
                    **_e2_solver_kwargs(args, q, qs))
                _vmode = getattr(args, "validation_mode", "synthetic_stress")
                rp = _replay_columns(r["chosen"], p_u, xi_amb, p_u.eps_E,
                                     n_per=args.replay_n, seed=7, wamb=wamb,
                                     validation_mode=_vmode,
                                     real_samples_csv=getattr(args, "validation_samples", None),
                                     weather_sample_mode=("real" if _vmode in ("real_validation", "real_holdout")
                                                          else "synthetic"))
                safe = rp["safe_tids"]
                _bud = RM.mission_eps_budget(p_u, wamb is not None)
                holds = _formal_validation_selection_gate(rp)
                strict_holds = rp.get("mission_requirement_holds")
                all_hold = holds
                all_strict = strict_holds
                stat_union = holds
                stat_strict = strict_holds
                plan_fp = _frozen_plan_fingerprint(r["chosen"])
                plan_cache[(str(name), float(q))] = list(r["chosen"])
                _formal_e2 = str(getattr(args, "study_mode", "mechanism")) == "formal"
                _harshest_q = float(q) == max(float(x) for x in qs)
                _global_cert = bool(_global_certificate_flag(r))
                # Lower-q points are diagnostics.  At q_max, formal completion
                # means a full lexicographic physical certificate, not merely
                # that the solver returned an incumbent before its deadline.
                _run_status = ("unresolved"
                               if (_formal_e2 and _harshest_q and not _global_cert)
                               else "ok")
                raw.append(dict(q=q, Hs0=round(wx0["Hs"], 3), wind10_0=round(wx0["wind10"], 2),
                                criterion=name, klass=klass, run_status=_run_status,
                                e2_point_role=("formal-harshest-certification"
                                               if (_formal_e2 and _harshest_q)
                                               else "diagnostic-discovery"),
                                error_type=None, error_message=None,
                                elapsed_s=round(time.time() - started, 3),
                                plan_fingerprint=plan_fp,
                                e1_formal_freeze_verified=bool(
                                    getattr(args, "_e1_formal_freeze_verified", False)),
                                e1_formal_freeze_sha256=str(
                                    getattr(args, "_e1_formal_freeze_sha256", "")),
                                e1_config_source=str(getattr(args, "_e1_config_source", "unresolved")),
                                covered=r["covered"], safe_served=len(safe),
                                safe_ratio=(round(len(safe) / r["covered"], 3) if r["covered"] else None),
                                flights=r["flights"], mean_stops=r["mean_stops"],
                                multi_stop_ratio=r["multi_stop_ratio"],
                                energy_Wh=r["energy_Wh"],
                                energy_per_safe=(round(r["energy_Wh"] / len(safe), 1) if safe else None),
                                emp_viol=rp["emp"],
                                component_eps=round(float(p_u.eps_E), 6),
                                mission_eps_budget=round(float(_bud), 6),
                                # Deprecated alias retained for old plotting scripts.
                                bound_2eps=round(float(_bud), 6), eps_budget=round(float(_bud), 6),
                                strict_5pct_holds=strict_holds,
                                union_budget_holds=holds, holds=holds,
                                all_columns_strict_5pct=all_strict,
                                all_columns_union_budget=all_hold,
                                all_columns_hold=all_hold,
                                statistically_holds_strict_5pct=stat_strict,
                                statistically_holds_union_budget=stat_union,
                                max_col_viol=rp["max_col_viol"],
                                n_test_total=rp.get("n_test_total", 0),
                                n_viol_total=rp.get("n_viol_total", 0),
                                emp_viol_ci95_low=rp.get("ci95_low"),
                                emp_viol_ci95_high=rp.get("ci95_high"),
                                emp_viol_upper95=rp.get("upper95"),
                                ci_method=rp.get("ci_method"),
                                formal_reliability_claim_eligible=bool(
                                    rp.get("formal_reliability_claim_eligible", False)),
                                evidence_scope=rp.get("evidence_scope", "mechanism-or-partial-evidence"),
                                n_replayed_cols=rp["n_replayed"], n_missing_replay=rp["n_missing"],
                                n_realized=rp.get("n_realized", 0),
                                n_realized_viol=rp.get("n_realized_viol", 0),
                                realized_viol_ci95_low=rp.get("realized_ci95_low"),
                                realized_viol_ci95_high=rp.get("realized_ci95_high"),
                                realized_viol_upper95=rp.get("realized_upper95"),
                                validation_type=rp["validation_type"] + "+single-realized-audit",
                                disjoint_real_holdout=rp.get("disjoint_real_holdout", False),
                                independent_real_holdout=False,
                                uav=args.uav, K=K, batteries=B, solver=r["solver"],
                                formal_algorithm=r.get("algorithm"),
                                status=r.get("status"),
                                termination_reason=r.get("termination_reason"),
                                coverage_incumbent=r.get("coverage_incumbent"),
                                coverage_upper_bound=r.get("coverage_upper_bound"),
                                coverage_gap_abs=r.get("coverage_gap_abs"),
                                coverage_gap_pct=r.get("coverage_gap_pct"),
                                coverage_optimal=r.get("coverage_optimal"),
                                energy_incumbent_Wh=r.get("energy_incumbent_Wh"),
                                energy_lower_bound_Wh=r.get("energy_lower_bound_Wh"),
                                energy_gap_abs_Wh=r.get("energy_gap_abs_Wh"),
                                energy_gap_pct=r.get("energy_gap_pct"),
                                conditional_energy_gap_pct=r.get("conditional_energy_gap_pct"),
                                global_energy_gap_reason=r.get("global_energy_gap_reason"),
                                energy_optimal=r.get("energy_optimal"),
                                lexicographic_optimal=r.get("lexicographic_optimal"),
                                global_certificate_available=_global_certificate_flag(r),
                                global_route_space_certificate=_global_certificate_flag(r),
                                implicit_route_space_certified=_implicit_route_space_certificate(r),
                                physical_model_global_certificate=r.get("physical_model_global_certificate"),
                                coverage_global_certificate_available=_coverage_certificate_flag(r),
                                solve_scope=r.get("solve_scope", "lexicographic"),
                                pricing_complete=r.get("pricing_complete"),
                                pricing_bound_available=r.get("pricing_bound_available"),
                                resource_audit_complete=r.get("resource_audit_complete"),
                                branching_complete=r.get("branching_complete"),
                                farkas_pricing_complete=r.get("farkas_pricing_complete"),
                                bound_scope=r.get("bound_scope"),
                                bound_source=r.get("bound_source"),
                                open_nodes=r.get("open_nodes"),
                                processed_nodes=r.get("processed_nodes"),
                                generated_columns=r.get("generated_columns"),
                                pricing_calls=r.get("pricing_calls"),
                                exact_pricing_calls=r.get("exact_pricing_calls"),
                                resource_cuts_added=r.get("resource_cuts_added"), **prov))
                if _run_status == "ok":
                    _done.add((str(name), str(float(q))))
                log.info("E2 q=%.2f %-10s: covered=%d safe=%d viol=%s union=%s strict=%s miss=%d cert=%s status=%s",
                         q, name, r["covered"], len(safe), rp["emp"], holds, strict_holds,
                         rp["n_missing"], _global_cert, _run_status)
            except Exception as e:
                raw.append(dict(q=q, Hs0=round(wx0["Hs"], 3), wind10_0=round(wx0["wind10"], 2),
                                criterion=name, klass=klass, run_status="failed",
                                error_type=type(e).__name__, error_message=str(e)[:1000],
                                elapsed_s=round(time.time() - started, 3),
                                plan_fingerprint=None,
                                e1_formal_freeze_verified=bool(
                                    getattr(args, "_e1_formal_freeze_verified", False)),
                                e1_formal_freeze_sha256=str(
                                    getattr(args, "_e1_formal_freeze_sha256", "")),
                                e1_config_source=str(getattr(args, "_e1_config_source", "unresolved")),
                                covered=None, safe_served=None, safe_ratio=None, flights=None,
                                mean_stops=None, multi_stop_ratio=None, energy_Wh=None,
                                energy_per_safe=None, emp_viol=None,
                                component_eps=round(float(p_u.eps_E), 6),
                                mission_eps_budget=round(float(RM.mission_eps_budget(p_u, wamb is not None)), 6),
                                bound_2eps=round(float(RM.mission_eps_budget(p_u, wamb is not None)), 6),
                                eps_budget=round(float(RM.mission_eps_budget(p_u, wamb is not None)), 6),
                                strict_5pct_holds=None, union_budget_holds=None, holds=None,
                                all_columns_strict_5pct=None, all_columns_union_budget=None,
                                all_columns_hold=None, statistically_holds_strict_5pct=None,
                                statistically_holds_union_budget=None, max_col_viol=None,
                                n_test_total=0, n_viol_total=0, emp_viol_ci95_low=None,
                                emp_viol_ci95_high=None, emp_viol_upper95=None,
                                ci_method=None, formal_reliability_claim_eligible=False,
                                evidence_scope="failed-no-evidence",
                                n_replayed_cols=0, n_missing_replay=None,
                                n_realized=0, n_realized_viol=0,
                                realized_viol_ci95_low=None, realized_viol_ci95_high=None,
                                realized_viol_upper95=None,
                                validation_type=(("real-validation-not-final"
                                                  if getattr(args, "validation_mode", "synthetic_stress") in ("real_validation", "real_holdout")
                                                  else "synthetic-moment-matched-t3-stress")
                                                 + "+single-realized-audit"),
                                independent_real_holdout=False,
                                uav=args.uav, K=K, batteries=B, solver=None,
                                formal_algorithm="branch-price-and-cut-with-logic-benders",
                                status="solver_error", termination_reason=str(e)[:1000],
                                coverage_incumbent=None, coverage_upper_bound=None,
                                coverage_gap_abs=None, coverage_gap_pct=None,
                                coverage_optimal=False, energy_incumbent_Wh=None,
                                energy_lower_bound_Wh=None, energy_gap_abs_Wh=None,
                                energy_gap_pct=None, conditional_energy_gap_pct=None,
                                global_energy_gap_reason="solver failed before a valid result",
                                energy_optimal=False, lexicographic_optimal=False,
                                global_certificate_available=False, global_route_space_certificate=False,
                                implicit_route_space_certified=False, physical_model_global_certificate=False,
                                coverage_global_certificate_available=False, solve_scope="lexicographic",
                                e2_point_role=("formal-harshest-certification"
                                               if (str(getattr(args, "study_mode", "mechanism")) == "formal"
                                                   and float(q) == max(float(x) for x in qs))
                                               else "diagnostic-discovery"),
                                pricing_complete=False, pricing_bound_available=False,
                                resource_audit_complete=False, branching_complete=False,
                                farkas_pricing_complete=False, bound_scope=None,
                                bound_source=None, open_nodes=None, processed_nodes=None,
                                generated_columns=None, pricing_calls=None,
                                exact_pricing_calls=None, resource_cuts_added=None, **prov))
                log.exception("E2 q=%.2f %s 失败，已写入失败记录并保留续跑资格", q, name)
            _save(pd.DataFrame(raw), outdir, "E2_robust_raw.csv")
            EU.write_run_manifest(
                outdir, "E2_robust", args,
                input_paths=[x for x in [getattr(args, "xi_train_samples", None),
                                          getattr(args, "validation_samples", None),
                                          getattr(args, "final_test_samples", None),
                                          getattr(args, "_resolved_track_csv", None),
                                          getattr(args, "weather_moments_csv", None)]
                             if x is not None],
                extra={"completed_jobs": len(_done), "result_contract": RESULT_CONTRACT,
                       **_formal_instance_manifest_extra(args)})
    dfr = pd.DataFrame(raw); _save(dfr, outdir, "E2_robust_raw.csv")
    summ = []
    for name, _, _, klass in _E2_CRITERIA:
        all_sub = dfr[dfr.criterion == name]
        sub = all_sub[all_sub.run_status == "ok"] if "run_status" in all_sub else all_sub
        if not len(all_sub):
            continue
        if not len(sub):
            summ.append(dict(criterion=name, klass=klass, run_status="all_failed",
                             n_expected=int(sum(1 for n, _q in expected_jobs if n == name)),
                             n_completed=0, n_failed=int(len(all_sub))))
            continue
        low = sub[sub.covered < 3]
        miss = sub[(sub.covered > 0) & (sub.n_missing_replay > 0)]
        valid = sub[(sub.covered >= 3) & sub.holds.notna() & (sub.n_missing_replay == 0)]
        summ.append(dict(criterion=name, klass=klass, run_status="ok",
                         n_expected=int(sum(1 for n, _q in expected_jobs if n == name)),
                         n_completed=int(len(sub)),
                         n_failed=int((all_sub.run_status == "failed").sum()),
                         total_covered=int(sub.covered.sum()),
                         total_safe=int(sub.safe_served.sum()),
                         mean_safe_ratio=(round(float(sub.safe_ratio.dropna().mean()), 3)
                                          if sub.safe_ratio.notna().any() else None),
                         holds_rate=(round(float(valid.holds.mean()), 3) if len(valid) else None),
                         strict_5pct_holds_rate=(round(float(valid.strict_5pct_holds.mean()), 3)
                                                 if len(valid) else None),
                         statistically_holds_union_rate=(
                             round(float(valid.statistically_holds_union_budget.dropna().mean()), 3)
                             if valid.statistically_holds_union_budget.notna().any() else None),
                         n_all_columns_hold=int((sub.all_columns_hold == True).sum()),
                         n_valid=len(valid), n_low_coverage=len(low),
                         n_missing_replay=len(miss),
                         energy_Wh=round(float(sub.energy_Wh.sum()), 1),
                         energy_per_safe=(round(float(sub.energy_Wh.sum())
                                                / max(int(sub.safe_served.sum()), 1), 1)
                                          if int(sub.safe_served.sum()) else None)))
    dfs = pd.DataFrame(summ); _save(dfs, outdir, "E2_robust_summary.csv")
    successful_keys = [(str(r.criterion), float(r.q)) for r in dfr.itertuples()
                       if getattr(r, "run_status", None) == "ok"]
    completion = EU.matrix_completion(expected_jobs, successful_keys)
    completion_rows = []
    for name, q in sorted(expected_jobs, key=lambda x: (float(x[1]), str(x[0]))):
        hit = dfr[(dfr.criterion == name) & dfr.q.map(lambda v: _binary64_equal(v, q))]
        if len(hit):
            rr = hit.iloc[-1]
            st = str(rr.get("run_status", "unknown"))
            et = rr.get("error_type")
            em = rr.get("error_message")
        else:
            st, et, em = "missing", None, None
        completion_rows.append(dict(criterion=name, q=float(q), run_status=st,
                                    error_type=et, error_message=em))
    _save(pd.DataFrame(completion_rows), outdir, "E2_completion.csv")
    EU.write_run_manifest(
        outdir, "E2_robust", args,
        input_paths=[x for x in [getattr(args, "xi_train_samples", None),
                                  getattr(args, "validation_samples", None),
                                  getattr(args, "final_test_samples", None),
                                  getattr(args, "_resolved_track_csv", None),
                                  getattr(args, "weather_moments_csv", None)]
                     if x is not None],
        extra={"matrix_completion": completion, "result_contract": RESULT_CONTRACT,
               **_formal_instance_manifest_extra(args)})
    setattr(args, "_e2_matrix_completion_verified", bool(completion["complete"]))
    if not completion["complete"] and not bool(getattr(args, "allow_incomplete_results", False)):
        raise SystemExit("E2 实验矩阵不完整，已保存失败/缺失记录到 E2_completion.csv。"
                         f"缺失键: {completion['missing']}。修复后原命令 --resume on 续跑；"
                         "仅 mechanism 诊断可加 --allow-incomplete-results。")

    # test 不能参与方法比较或调参。完整 validation 矩阵完成后，只冻结一个预声明候选，
    # 并在独立 test 上审计一次。无 test 文件时保留接口但不生成真实可靠性结论。
    if getattr(args, "final_test_samples", None) is not None:
        if getattr(args, "study_mode", "mechanism") == "formal":
            _verify_formal_sample_hashes_unchanged(args)
        if not _formal_e2_final_test_authorized(args):
            raise SystemExit(
                "formal E2 final test requires an auto-resolved, structured E1 resource-minimal "
                "threshold + global-lex + validation freeze; manual --uav/--k/--batteries "
                "is diagnostic only and may not consume final test.")
        candidate = _select_e2_validation_candidate(dfr, qs)
        if candidate is None:
            msg = "E2 在最严天气分位没有 validation 风险门通过的候选，禁止消费 final test。"
            if getattr(args, "study_mode", "mechanism") == "formal":
                raise SystemExit(msg)
            log.warning(msg)
        else:
            final_path = outdir / "E2_final_test.csv"
            test_hash = EU.sha256_file(args.final_test_samples) or "none"
            expected_fp = str(candidate.get("plan_fingerprint", ""))
            if not expected_fp or expected_fp.lower() == "nan":
                raise SystemExit("E2冻结候选缺少 plan_fingerprint；为保证方案真正冻结，必须重跑validation。")
            expected_key = (
                str(candidate["criterion"]), float(candidate["q"]), expected_fp, str(test_hash),
                str(getattr(args, "_resume_input_sha256", "")),
                str(getattr(args, "_e1_formal_freeze_sha256", "")),
                str(EU.sha256_file(getattr(args, "xi_train_samples", None)) or "none"),
                str(EU.sha256_file(getattr(args, "validation_samples", None)) or "none"),
                RESULT_CONTRACT, FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
                BP.RESULT_CERTIFICATE_CONTRACT, BP.FORMAL_PROOF_CONTRACT)
            reuse = False
            prior_invocations = 0
            if final_path.is_file():
                prev = pd.read_csv(final_path, encoding="utf-8-sig")
                if len(prev) == 1:
                    row = prev.iloc[0]
                    try:
                        prior_invocations = int(row.get("final_test_invocations", 1))
                    except Exception:
                        prior_invocations = 1
                    got_key = (
                        str(row.get("selected_criterion")), float(row.get("selected_q")),
                        str(row.get("frozen_plan_fingerprint")),
                        str(row.get("final_test_samples_sha256")),
                        str(row.get("resume_input_sha256")),
                        str(row.get("e1_formal_freeze_sha256")),
                        str(row.get("train_samples_sha256")),
                        str(row.get("validation_samples_sha256")),
                        str(row.get("result_contract")),
                        str(row.get("formal_experiment_scheduler_contract")),
                        str(row.get("result_certificate_contract")),
                        str(row.get("formal_proof_contract")))
                    reuse = (got_key == expected_key)
                if not reuse and not bool(getattr(args, "allow_final_test_rerun", False)):
                    raise SystemExit("E2_final_test.csv 已存在且冻结候选/测试哈希不一致。"
                                     "为避免反复查看 test，默认拒绝重跑；确需审计性重跑时显式加 "
                                     "--allow-final-test-rerun，并记录为授权重跑。")
            if not reuse:
                rec = _e2_final_test_record(
                    candidate, turbines, lat0lon0, track_csv, xi_amb, wx_df,
                    p_u, wamb, args, K, B, cap_u, t_swap, t_launch,
                    frozen_chosen=plan_cache.get((str(candidate["criterion"]), float(candidate["q"]))),
                    invocation_count=int(prior_invocations + 1))
                _save(pd.DataFrame([rec]), outdir, "E2_final_test.csv")
            else:
                log.info("E2 final test 已按相同冻结候选和测试哈希执行过，本次复用结果，不重复消费 test。")
    return dfr, dfs



def _run_algorithm_full_suite(args):
    """无 shell 运算符地串行运行 A1→A2；任一步失败立即停止。"""
    script = Path(__file__).resolve().with_name("step14_experiment_algorithm.py")
    cmd = [
        sys.executable, str(script), "--exp", "all",
        "--farm", str(args.farm),
        "--pair-radius", str(args.pair_radius),
        "--max-stops", str(int(args.algorithm_max_stops)),
        "--window-min", str(float(args.window_min)),
        "--dtau-min", str(float(args.dtau_min)),
        "--deck-delta-min", str(float(args.deck_delta_min)),
        "--deck-mode", str(args.deck_mode),
        "--landing-clear-min", str(float(args.landing_clear_min)),
        "--quick-inspection-capacity", str(int(args.quick_inspection_capacity)),
        "--swap-stations", str(int(args.swap_stations)),
        "--uav", str(args.uav),
        "--k", str(int(args.k)),
        "--batteries", str(int(args.batteries)),
        "--selection-metric", str(args.selection_metric),
        "--hs-quantile", str(float(args.hs_quantile)),
        "--weather-drcc", str(args.weather_drcc),
        "--weather-alignment", str(args.weather_alignment),
        "--recovery-predictor", str(args.recovery_predictor),
        "--pool-h", str(args.pool_h),
        "--soc-correction", str(args.soc_correction),
        "--soc-risk-allocation", str(args.soc_risk_allocation),
        "--time-recourse", str(args.time_recourse),
        "--study-mode", str(args.study_mode),
        "--resume", str(args.resume),
    ]
    # full_suite must preserve the exact real-data provenance when invoking step14.
    for flag, value in (("--turbines-csv", args.turbines_csv), ("--wind-csv", args.wind_csv),
                        ("--wave-csv", args.wave_csv), ("--xi-moments-csv", args.xi_moments_csv),
                        ("--xi-train-samples", args.xi_train_samples),
                        ("--weather-moments-csv", args.weather_moments_csv),
                        ("--recovery-scenarios-csv", args.recovery_scenarios_csv)):
        if value is not None:
            cmd += [flag, str(value)]
    cmd += ["--solver-mode", str(args.solver_mode), "--pricing-mode", str(args.pricing_mode),
            "--coverage-gap-target-abs", str(int(args.coverage_gap_target_abs)),
            "--energy-gap-target-rel", str(float(args.energy_gap_target_rel)),
            "--energy-gap-target-abs-wh", str(float(args.energy_gap_target_abs_wh))]
    if args.time_limit_s is not None:
        cmd += ["--time-limit-s", str(float(args.time_limit_s))]

    if args.t_swap_min is not None:
        cmd += ["--t-swap-min", str(float(args.t_swap_min))]
    if args.t_launch_min is not None:
        cmd += ["--t-launch-min", str(float(args.t_launch_min))]
    if args.track_csv:
        cmd += ["--track-csv", str(args.track_csv)]
    if args.track_mmsi:
        cmd += ["--track-mmsi", str(args.track_mmsi)]
    if args.track_start_min is not None:
        cmd += ["--track-start-min", str(float(args.track_start_min))]
    if args.allow_synth:
        cmd.append("--allow-synth")

    print("\n[full_suite] 模型实验完成，开始串行运行 A1_accuracy → A2_speed")
    print("[full_suite] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=True)
    print("\n[full_suite] E1_frontier → E1_select → E2_robust → A1_accuracy → A2_speed 全部完成。")


def _register_saa_baseline(args, saa_csv: Path, forbidden_hashes: set[str], *, mmsi: str = "ALL") -> tuple[int, bool]:
    """Register SAA train samples with fail-closed formal semantics.

    In mechanism mode only, an *auto-discovered* legacy/incompatible default
    ``tracks/xi_samples_caseB.csv`` is treated as unavailable and the caller may
    use the documented moment-Gaussian synthetic baseline.  Explicitly supplied
    samples and all formal runs remain fail-closed.
    """
    saa_explicit = bool(getattr(args, "xi_train_samples", None) is not None
                        or getattr(args, "saa_samples", None))
    if not saa_csv.is_file():
        return 0, False
    if EU.sha256_file(saa_csv) in forbidden_hashes:
        raise SystemExit("SAA 样本与 validation/final test 相同，已拒绝泄漏；SAA 只能使用 train。")
    try:
        _formal = (str(getattr(args, "study_mode", "formal")) == "formal")
        return int(RM.load_saa_empirical(
            saa_csv, mmsi=str(mmsi),
            require_current_contract=_formal,
            allow_pooled_fallback=not _formal)), True
    except ValueError as exc:
        if str(getattr(args, "study_mode", "formal")) == "formal" or saa_explicit:
            raise
        RM.SAA_EMPIRICAL.clear()
        RM.SAA_SOURCE = (
            f"moment-gaussian(auto SAA rejected: {saa_csv.name}; "
            f"{type(exc).__name__}: {exc})")
        log.warning(
            "自动探测的 SAA 经验样本与当前合同不兼容，mechanism 模式将拒绝该旧样本并"
            "回退矩匹配高斯对照；不会把旧样本当作当前经验数据。file=%s; reason=%s",
            saa_csv, exc)
        return 0, False

def main():
    # Windows consoles commonly default to GBK. argparse --help contains a few
    # Unicode math/arrows not representable in GBK; render them with replacement
    # instead of crashing before argument parsing. Solver/model semantics are unchanged.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    ap = argparse.ArgumentParser(description=f"{PROJECT_NAME} 时空耦合机队巡检：E1/E2 模型实验")
    ap.add_argument("--exp", default="all",
                    choices=["all", "full_suite", "E1_frontier", "E1_knee_refine",
                             "E1_lex_certify", "E1_select", "E2_robust", "E2_safety"],
                    help="full_suite=E1_frontier→E1_knee_refine→E1_select→E2/A；"
                         "all=本脚本 E1_frontier→E1_knee_refine→E2_robust；"
                         "E1_knee_refine=复用当前实例/provenance 一致的 E1_frontier 做 threshold predecessor "
                         "exact decision + knee-only full lex；"
                         "E1_lex_certify=对显式 --uav/--k/--batteries 资源点直接执行完整 coverage→energy 词典序证明；"
                         "E2_safety=E2_robust 兼容旧名。")
    ap.add_argument("--n-turbines", type=int, default=None)
    ap.add_argument("--farm", default="Rodsand_II", choices=["Rodsand_II", "Nysted", "Anholt"])
    ap.add_argument("--pair-radius", type=str, default="auto",
                    help="reach 可达集的作业半径(米)。'auto'(更新 新默认)="
                         "由 UAV 档位物理推导的最大作业半径(step9.max_flight_radius_m, "
                         "取本批档位外包络, 正式口径); 数字=显式覆盖(仅敏感性分析; "
                         "旧 8km 口径已废弃, 旧结果作废重跑)。更新 起【不再】用于"
                         "窗起点/选船评分(改用 --infarm-radius, 两语义解耦)。")
    ap.add_argument("--infarm-radius", type=float, default=INFARM_RADIUS_M_DEFAULT,
                    help="更新: '船在风场作业'判据半径(米, 默认 3000=距最近风机≤3km)。"
                         "仅用于选船评分/进场窗自动起点/不在场警告; reach 与列级判据不受影响。"
                         "实际生效值=min(本值, pair_radius)。更新-44 误用外包络(≈21km)作此判据, "
                         "导致真航迹窗口对到泊港时段、E1 列池全零(见 build_launch_options 复盘注)。")
    ap.add_argument("--soc-correction", choices=["none", "geo2d"], default="geo2d",
                    help="SOC 线性化校正。geo2d(更新 起为默认=正式口径)=2-D 精确几何界"
                         "(消除一阶泰勒对返程距离的系统性低估); none=旧一阶口径(仅消融对照)。"
                         "全部实验须同一取值整批出结果, 不可混排(resume 签名强制)。")
    ap.add_argument("--soc-risk-allocation", choices=["fixed", "optimized"], default="optimized",
                    help="geo2d 内部 Bonferroni 风/沿向/横向份额。optimized 逐路线从有效拆分中"
                         "选择最紧证书，始终保留旧 fixed 拆分为候选；总风险预算不变。")
    ap.add_argument("--time-recourse", choices=["wait_only", "wait_and_speed"], default="wait_and_speed",
                    help="固定接地时刻的时间 recourse。wait_only=旧固定巡航速度，仅等待可压缩；"
                         "wait_and_speed=等待与返程空速共同调整，仍受 v_air_max 和原风险预算约束。")
    ap.add_argument("--resume", choices=["on", "off"], default="on",
                    help="断点续跑(更新)。on(默认)=若输出 CSV 已存在且口径签名一致, "
                         "跳过已完成键继续; 口径不一致直接报错拒绝混排。off=整跑覆盖。"
                         "实现: 每个单元求解后立即整表落盘, 结果 CSV 本身即检查点, "
                         "崩溃/中断后原命令重跑即续。")
    ap.add_argument("--max-stops", type=int, default=4,
                    help="--stops-cap 无法解析某档时的回退值；并非 auto 模式下的实际求解上限")
    ap.add_argument("--algorithm-max-stops", type=int, default=4,
                    help="--exp full_suite 调用 step14 时 A1/A2 使用的停靠上限；须与正式模型一致，默认 4")
    ap.add_argument("--window-min", type=float, default=360.0, help="窗 T=船在场时间(分钟); 超航迹时按航迹截断")
    ap.add_argument("--dtau-min", type=float, default=5.0, help="起飞时隙步长 Δτ(作者定案 5min)")
    ap.add_argument("--deck-delta-min", type=float, default=2.5, help="甲板事件槽宽 Δ(须整除 h 步长 5)")
    ap.add_argument("--deck-mode", default="interval", choices=["interval", "slot"],
                    help="更新: interval=起降区间占用甲板(新默认, 物理自洽+架构A割集); slot=旧瞬时槽(消融)")
    ap.add_argument("--t-swap-min", type=float, default=None,
                    help="非着陆区换电服务时长(分钟); 默认按 UAV 档位(S/M=4, L=5)")
    ap.add_argument("--landing-clear-min", type=float, default=1.0,
                    help="接地、停桨并移出着陆区的清场时间(分钟); 默认 1.0, 待实测标定")
    ap.add_argument("--swap-stations", type=int, default=1,
                    help="非着陆区并行换电工位数; 默认 1")
    ap.add_argument("--quick-inspection-capacity", type=int, default=1,
                    help="保留原电池时的并行快速检查工位数; 默认 1")
    ap.add_argument("--battery-reuse-mode", choices=["exact_soc", "legacy_count"], default="exact_soc",
                    help="exact_soc=实体电池组分配+剩余SOC复用(正式); legacy_count=一架次消耗一组(消融)")
    ap.add_argument("--t-launch-min", type=float, default=None,
                    help="起飞准备甲板占用(分钟); 默认 None=按 UAV 档位(S=2.5, M/L=3)")
    ap.add_argument("--uav", default="auto", choices=["auto"] + sorted(M.UAV_PROFILES),
                    help="E2/detail 的 UAV 档位(E1 选型后回填; 默认 S=DJI M30T)")
    ap.add_argument("--e1-uavs", type=str, default="S,M,L", help="E1 的 UAV 种类轴")
    ap.add_argument("--fleet-ks", type=str, default="1,2,3,4,5,6,7,8",
                    help="E1 的 K 轴(更新 默认加 1,2: 首跑 3..8 全平坦=零信息量, "
                         "需 K=1,2 定位 K 何时真正绑定; --fleet-ks 3,...,8 复现 更新)")
    ap.add_argument("--e1-batteries", type=str, default="0,1,2,3,4,5,6,7,8",
                    help="E1 的电池【基础】网格(0 递增到 8; B=0 为退化锚点); "
                         "更新 起默认在此之上按饱和自动延伸(见 --e1-b-auto)")
    ap.add_argument("--e1-b-auto", default="on", choices=["on", "off"],
                    help="更新: B 轴饱和自动延伸 —— K_max 曲线连续 --e1-sat-patience 次"
                         "边际增益=0 即停(首跑 0..8 内严格线性无饱和 ⇒ 膝点规则退化, 故加); "
                         "off=固定网格(复现 更新 口径)")
    ap.add_argument("--e1-b-cap", type=int, default=8,
                    help="B 自动延伸请求上限；formal 实际额外取 min(本值, |I|)。"
                         "当前 8 台实例 B>8 结构冗余，hard-coverable-cap 后禁止继续扩 B。")
    ap.add_argument("--e1-sat-patience", type=int, default=3,
                    help="判饱和所需的连续零边际次数(默认 3，降低边界假平台风险)")
    ap.add_argument("--e1-frontier-time-limit-s", type=float, default=120.0,
                    help="formal exact E1 每个资源格的 Stage-1 coverage-only discovery 时限(秒，默认120)。"
                         "只影响调度：未闭合格保留严格 incumbent/UB，不会获得 exact certificate；"
                         "<=0 表示继承 --time-limit-s。")
    ap.add_argument("--e1-certify-time-limit-s", type=float, default=None,
                    help="formal E1 膝点候选的完整两阶段 exact resolve 时限；默认继承 --time-limit-s。"
                         "最多只对少数候选使用，不再让全部网格都支付 Stage-2 成本。")
    ap.add_argument("--formal-warmstart-seconds", type=float, default=60.0,
                    help="formal 大规模回退路径的 heuristic multi-stop candidate discovery 软时限。"
                         "v12 small-n 完整路线宇宙启用时自动跳过该 warm start。")
    ap.add_argument("--formal-route-universe", choices=["auto", "off", "force"], default="auto",
                    help="v12 formal E1 完整物理路线宇宙加速。auto=仅 small-n/低 stops 时启用；"
                         "off=使用原隐式 exact BPC；force=无视 small-n 阈值强制尝试完整物化。")
    ap.add_argument("--formal-route-universe-max-turbines", type=int, default=8,
                    help="--formal-route-universe auto 的最大风机数，默认 8。")
    ap.add_argument("--formal-route-universe-max-stops", type=int, default=4,
                    help="--formal-route-universe auto 的最大 stops，默认 4。")
    ap.add_argument("--formal-route-universe-time-limit-s", type=float, default=7200.0,
                    help="每 UAV 一次性构建完整物理路线宇宙的时限。<=0 表示不限时；"
                         "若 small-n auto 已启用但未闭合，正式实验 fail-closed，不把残缺列池当证书。")
    ap.add_argument("--stops-cap", type=str, default="auto",
                    help="更新: 停靠数【生成上界】, E1/E2 共用, 逐 UAV 解析(_stops_cap)。"
                         "auto=max(--max-stops, ⌊h_max/τ_insp⌋)=时间预算逻辑上界(默认网格下=9, "
                         "作者扩 horizons 后自动=12) —— 消除 更新 对 L 档的删失; "
                         "整数=全档统一(传 4 逐位复现 更新); 逐档映射如 'S:4,M:5,L:6'")
    ap.add_argument("--e1-csv", default=None,
                    help="E1_select 的输入(默认 results/model_experiments/E1_frontier/E1_frontier.csv)")
    ap.add_argument("--knee-frac", type=float, default=0.95,
                    help="膝点阈值: safe_served ≥ frac·plateau(doc_experiments §2.3)")
    ap.add_argument("--knee-order", default="BK", choices=["BK", "KB"],
                    help="膝点'最小 (K,B)'的字典序: BK=电池优先(默认; 电池是绑定/稀缺资源)")
    ap.add_argument("--selection-metric", default="safe_per_inventory_kWh",
                    choices=["safe_per_inventory_kWh", "per_battery", "energy_per_safe", "max_safe"],
                    help="跨 UAV 自动选型指标。默认按安全覆盖/库存kWh，避免不同容量电池按块数硬比较；"
                         "per_battery 仅用于复现旧口径。")
    ap.add_argument("--k", type=int, default=None,
                    help="E2 机队规模; --uav auto(默认)时忽略并由 E1 选型自动回填, 显式 --uav 时默认 3")
    ap.add_argument("--batteries", type=int, default=None, help="E2 的电池数(默认 2K; E1 选型后回填)")
    ap.add_argument("--hs-quantile", type=float, default=0.5, help="E1 天气序列起始分位")
    ap.add_argument("--e2-quantiles", type=str, default="0.2,0.5,0.8")
    ap.add_argument("--e2-criteria", type=str, default="recourse_compatible",
                    help="E2 判据。默认 recourse_compatible: wait_and_speed 仅比较 "
                         "nominal/gaussian/cantelli/vp；wait_only 可用 all 比较全部7个。")
    ap.add_argument("--e2-discovery-time-limit-s", type=float, default=120.0,
                    help="formal E2 非最严天气分位的诊断 discovery 时限(秒，默认120；<=0继承 --time-limit-s)。")
    ap.add_argument("--e2-certify-time-limit-s", type=float, default=None,
                    help="formal E2 最严天气分位每个方法的完整 lexicographic 认证时限；默认继承 --time-limit-s。")
    ap.add_argument("--allow-incomplete-results", action="store_true",
                    help="仅 mechanism 诊断用：允许 E2 矩阵存在失败/缺失键并返回；formal 明确拒绝。")
    ap.add_argument("--allow-final-test-rerun", action="store_true",
                    help="仅 mechanism 诊断用：允许重跑 test；formal confirmatory final test 严格一次性，明确拒绝。")
    ap.add_argument("--replay-n", type=int, default=400, help="合成压力测试每 (h,c) 格样本数")
    ap.add_argument("--study-mode", choices=["formal", "mechanism"], default="formal",
                    help="formal=正式 train/validation/test 协议并 fail-closed；"
                         "mechanism=允许显式合成压力测试，但不得形成真实可靠性声明。")
    ap.add_argument("--validation-mode",
                    choices=["synthetic_stress", "real_validation", "real_holdout"],
                    default="real_validation",
                    help="E1 前沿/选型使用 validation。synthetic_stress 仅允许 --study-mode mechanism；"
                         "real_holdout 为 real_validation 的兼容旧名。")
    ap.add_argument("--xi-train-samples", type=Path, default=None,
                    help="仅用于估计 ξ 矩和 SAA 的 purged train CSV")
    ap.add_argument("--validation-samples", type=Path, default=None,
                    help="可参与选型的 purged validation CSV；不得与 train/test 重叠")
    ap.add_argument("--final-test-samples", type=Path, default=None,
                    help="冻结 selected knee 后一次性审计的 purged test CSV；不得参与选型")
    ap.add_argument("--holdout-purge-min", type=float, default=None,
                    help="train/validation/test 最小时间隔离；默认=max(h_min)")
    ap.add_argument("--final-weather-mode", choices=["real", "synthetic"], default="real",
                    help="real 要求 test CSV 含 wind_error_e_ms/wind_error_n_ms/hs_error_m，"
                         "此时才可标记完整真实联合留出；synthetic 仅为真实 ξ + 合成天气的部分审计。")
    ap.add_argument("--solver-mode",
                    choices=["exact-branch-price-cut", "research-baseline"],
                    default="exact-branch-price-cut",
                    help="正式实验使用按需列生成的精确 Branch-Price-and-Cut；research-baseline 仅作受限列池研究对照且不发全局证书。")
    ap.add_argument("--pricing-mode",
                    choices=["exact-implicit-dfs", "exact-discovery-shadow",
                             "exact-dual-guided-shadow",
                             "exact-layered-guided-shadow",
                             "exact-layered-batch-shadow",
                             "exact-layered-batch-primal-shadow",
                             "exact-layered-batch-primal-diagnostic-shadow",
                             "exact-layered-batch-primal-target9-diagnostic-shadow",
                             "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
                             "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow",
                             "exact-layered-batch-primal-battery-halfcap-formal",
                             "exact-layered-batch-primal-battery-halfcap-depth-fair-formal",
                             "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal",
                             "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                             "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                             "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                             "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                             "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                             "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                             "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                             "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal",
                             "r-bpc",
                             "exact-mip"],
                    default="r-bpc",
                    help="定价/求解策略。论文正式算法使用 r-bpc：resource-aware exact "
                         "Branch-and-Price-and-Cut，包含正式 battery half-cap strengthening、"
                         "heuristic-first layered exact pricing、resource-aware singleton timing "
                         "enrichment、exact entity-resource audit/cuts，以及 generated-column "
                         "exact primal recovery。Timing enrichment 只负责发现合法列，不能证明 "
                         "pricing closure、界、剪枝或不可行；restricted archive recovery 只可在"
                         " unchanged exact resource audit 通过后提升 incumbent/LB，其 restricted "
                         "UB/certificate 永不进入 full-space proof。其余模式保留用于历史复现、"
                         "消融与回归；exact-mip 在缺少与物理/DRCC严格等价编码时 fail-closed。")
    ap.add_argument("--time-limit-s", type=float, default=None,
                    help="覆盖枚举、定价、MILP、资源 DFS、割循环和能耗阶段的统一墙钟预算。")
    ap.add_argument("--archive-diagnostic-time-limit-s", type=float, default=30.0,
                    help="正式求解时钟结束后的冻结 archive 诊断额外秒数；"
                         "V8/V20.1 求 restricted maximum coverage，V9/V10/V11 做 coverage>=9 target decision；"
                         "V20.1 另用最多 min(10s,该预算) 做 variant/blocker 诊断；都绝不升级 full-space 证书。")
    ap.add_argument("--archive-shadow-diagnostic-time-limit-s", type=float, default=30.0,
                    help="V10/V11：target-9 decision 已结束后，对其已拒 patterns 做 certificate-safe "
                         "resource-ladder/core shadow 分析的独立额外秒数；不改变 target decision。")
    ap.add_argument("--archive-clique-diagnostic-time-limit-s", type=float, default=30.0,
                    help="V11 专用：在 baseline target-9 与 resource shadow 后，用严格安全的 "
                         "battery-energy clique rows 对同一 frozen archive 做独立 target-9 A/B rerun；"
                         "结果仅为 diagnostic，不改变正式 certificate。")
    ap.add_argument("--archive-primal-recovery", choices=["on", "off"], default="off",
                    help="研究开关：在其它兼容 mode 中手工启用 generated-archive exact primal recovery；"
                         "V21 archive-recovery-formal mode 会强制启用。只导入经 unchanged exact resource audit "
                         "重新确认的更好 primal witness；archive 内部界/证书绝不用于 full-space pruning/UB/optimality。")
    ap.add_argument("--archive-primal-recovery-time-limit-s", type=float, default=2.0,
                    help="每次 V21 generated-current-archive exact primal recovery 的局部墙钟预算秒数；"
                         "results(4) 的 62-column exact archive optimum 用约 0.04 秒即证明，默认 2 秒留有充足余量。")
    ap.add_argument("--fullspace-target-diagnostic-time-limit-s", type=float, default=0.0,
                    help="V21 evidence build：正式 solver clock 结束后，用独立额外预算从最强 archive witness "
                         "开始做 full physical route-space coverage target ladder (LB+1,...)；只写 diagnostic "
                         "telemetry，不回写正式 LB/UB/剪枝/证书。")
    ap.add_argument("--coverage-gap-target-abs", type=int, default=0)
    ap.add_argument("--energy-gap-target-rel", type=float, default=0.0)
    ap.add_argument("--energy-gap-target-abs-wh", type=float, default=1e-6)
    ap.add_argument("--recovery-predictor", choices=["cv_noleak", "true_track"],
                    default="cv_noleak", dest="recovery_predictor",
                    help="更新: 回收点名义预测口径。cv_noleak=正式口径(仅 t≤τ 后向窗 CV, "
                         "与 step7 ξ 矩同源, 无泄漏); true_track=沿真实未来航迹取回收点"
                         "(泄漏消融/『已知计划航线』解释, 正式结果禁用)")
    ap.add_argument("--pool-h", choices=["pareto", "first"], default="pareto", dest="pool_h",
                    help="更新: 列池 h 保留策略。pareto=每 (τ,集合) 留 (h,E0) 非支配前沿"
                         "(正式口径, 修复最早可行 h 对能量层的截断); first=旧口径(消融)")
    ap.add_argument("--weather-drcc", default="on", choices=["on", "off"])
    ap.add_argument("--saa-samples", default=None,
                    help="SAA 经验样本 csv(step7 --dump-samples 的 xi_samples_caseB.csv); "
                         "默认自动探测 tracks/xi_samples_caseB.csv。机制模式缺失时可使用矩匹配高斯并如实标注；"
                         "正式E2必须提供train经验样本，缺失即拒绝。")
    ap.add_argument("--turbines-csv", type=Path, default=None, help="本地风机 CSV；正式实验必须显式提供。")
    ap.add_argument("--wind-csv", type=Path, default=None, help="本地风场 CSV；正式实验必须显式提供。")
    ap.add_argument("--wave-csv", type=Path, default=None, help="本地浪场 CSV；正式实验必须显式提供。")
    ap.add_argument("--xi-moments-csv", type=Path, default=None, help="可选本地 ξ 矩 CSV；train 样本会覆盖其统计量。")
    ap.add_argument("--weather-moments-csv", type=Path, default=None,
                    help="真实 ERA5/CMEMS 经 step7 内置 no-leak 预测生成的 weather_moments_caseB.csv；formal+weather-drcc 必需。")
    ap.add_argument("--recovery-scenarios-csv", type=Path, default=None)
    ap.add_argument("--track-csv", default=None, help="显式指定 AIS 航迹(如 tracks/track_219028973.csv)")
    ap.add_argument("--track-mmsi", default=None, help="等价于 --track-csv tracks/track_<mmsi>.csv")
    ap.add_argument("--track-start-min", type=float, default=None, help="窗起点=航迹第几分钟(默认自动取进场段)")
    ap.add_argument("--allow-synth", action="store_true", help="允许合成航迹(仅调试; 默认拒跑)")
    ap.add_argument("--weather-scale", type=float, default=1.0)
    ap.add_argument("--weather-alignment", choices=["timestamp", "representative_quantile"],
                    default="timestamp",
                    help="E1 天气起点口径：timestamp=按真实 AIS UTC 时间匹配再分析天气(正式默认)；"
                         "representative_quantile=按 Hs 分位构造情景。E2 的 q 轴固定采用后者并明确标注。")
    ap.add_argument("--insp-min", type=float, default=None,
                    help="单台巡检时长(分钟); 作者问题2定案=5(默认值即 5, 无须传)。物理参数, 改动即换口径。")
    args = ap.parse_args()
    args._source_tree_sha256 = M.source_tree_sha256(Path(__file__).resolve().parent)
    full_suite = (args.exp == "full_suite")
    if full_suite:
        # full_suite 复用本脚本 all 流程，之后以受检子进程串行执行 step14。
        # IDE 的脚本参数栏只需填写 --exp full_suite，无需 && 或反斜杠。
        args.exp = "all"
    if args.exp == "E2_safety":
        args.exp = "E2_robust"
    if args.validation_mode == "real_holdout":
        args.validation_mode = "real_validation"
    if args.exp == "E1_select":            # 纯 CSV 后处理，不触数据/求解
        E1_select(args)
        return
    _preflight_e2_criteria = None
    if args.exp in ("all", "E2_robust"):
        # Fail before any expensive model construction if the requested E2 methods
        # do not share the selected physical time-recourse policy.
        _preflight_e2_criteria = _resolve_e2_criteria_for_recourse(args)
    formal_protocol = (args.study_mode == "formal")
    _validate_formal_protocol_overrides(args)
    if formal_protocol:
        if args.allow_synth:
            raise SystemExit("--study-mode formal 禁止 --allow-synth；合成数据只能用于 mechanism。")
        required_local = {
            "--turbines-csv": args.turbines_csv,
            "--wind-csv": args.wind_csv,
            "--wave-csv": args.wave_csv,
        }
        missing_local = [name for name, value in required_local.items()
                         if value is None or not Path(value).is_file()]
        if args.track_csv is None and args.track_mmsi is None:
            missing_local.append("--track-csv/--track-mmsi")
        if args.track_start_min is None:
            missing_local.append("--track-start-min")
        if missing_local:
            raise SystemExit("正式实验必须通过本地路径显式提供输入: " + ", ".join(missing_local))
        if args.validation_mode != "real_validation":
            raise SystemExit("正式协议必须使用 --validation-mode real_validation。")
        if args.final_weather_mode != "real":
            raise SystemExit("正式协议必须使用 --final-weather-mode real。")
        required = [args.xi_train_samples, args.validation_samples, args.final_test_samples]
        if any(x is None or not Path(x).is_file() for x in required):
            raise SystemExit("正式协议必须同时提供存在的 --xi-train-samples、--validation-samples、"
                             "--final-test-samples。")
        if args.recovery_predictor != "cv_noleak" or args.pool_h != "pareto":
            raise SystemExit("正式协议要求 --recovery-predictor cv_noleak 和 --pool-h pareto。")
        if args.weather_drcc != "on" or args.soc_correction != "geo2d":
            raise SystemExit("正式协议要求 --weather-drcc on 和 --soc-correction geo2d。")
        if args.weather_moments_csv is None or not Path(args.weather_moments_csv).is_file():
            raise SystemExit("正式协议 weather-drcc=on 必须显式提供 --weather-moments-csv；"
                             "请先运行 step7_compute_xi.py 并显式提供 --weather-csv，禁止使用相邻再分析差分代理。")
        if args.battery_reuse_mode != "exact_soc" or args.deck_mode != "interval":
            raise SystemExit("正式协议要求 exact_soc 电池模型与 interval 甲板语义。")
    else:
        if args.validation_mode == "real_validation":
            if args.validation_samples is None or not args.validation_samples.is_file():
                raise SystemExit("--validation-mode real_validation 必须提供存在的 --validation-samples。")
        elif args.validation_mode == "synthetic_stress":
            log.warning("机制实验使用合成压力测试：结果不得标记为真实可靠性结论。")
    if args.final_test_samples is not None:
        required = [args.xi_train_samples, args.validation_samples, args.final_test_samples]
        if any(x is None or not Path(x).is_file() for x in required):
            raise SystemExit("最终独立审计必须同时提供 train、validation、test 三份样本。")
        if args.validation_mode != "real_validation":
            raise SystemExit("提供 final test 时必须使用 real_validation 完成选型。")

    if args.track_csv is None and args.track_mmsi:
        args.track_csv = str(
            Path(__file__).resolve().parent / "tracks" / f"track_{args.track_mmsi}.csv")
    turbines, wx_df, xi_amb, lat0lon0, sc_csv, src, track_csv = load_all(
        args.n_turbines, farm=args.farm, allow_synth=args.allow_synth,
        turbines_csv=args.turbines_csv, wind_csv=args.wind_csv, wave_csv=args.wave_csv,
        xi_moments_csv=args.xi_moments_csv,
        recovery_scenarios_csv=args.recovery_scenarios_csv,
        track_csv=args.track_csv,
        require_xi_moments=(args.xi_train_samples is None))
    if args.track_csv:
        track_csv = Path(args.track_csv)
    args._resolved_track_csv = (
        str(track_csv)
        if track_csv is not None and not isinstance(track_csv, (list, tuple))
        else None)
    preselected_mmsi = _infer_concrete_track_mmsi(
        track_csv, getattr(args, "track_mmsi", None),
        formal=(args.study_mode == "formal"))
    args._preselected_track_mmsi = str(preselected_mmsi or "ALL")

    # 正式独立协议：train 估矩、validation 选型、test 仅在冻结方案后审计。
    import step15_replay as RP
    train_df = validation_df = final_test_df = None
    args._holdout_disjointness_verified = False
    args._holdout_independence_verified = False
    formal_samples = (args.study_mode == "formal")
    if args.xi_train_samples is not None:
        train_df, xi_amb = _xi_ambiguity_from_train_samples(
            args.xi_train_samples, preselected_mmsi, formal=formal_samples)
        src = f"purged-train:{args.xi_train_samples.name}:{EU.sha256_file(args.xi_train_samples)}"
    if args.validation_samples is not None:
        validation_df = RP.load_samples(
            args.validation_samples, mmsi=(preselected_mmsi if formal_samples else "ALL"),
            formal=formal_samples,
            expected_split=("validation" if formal_samples else None))
    if args.final_test_samples is not None:
        if formal_samples:
            final_test_df = _load_final_test_metadata_pre_freeze(
                args.final_test_samples, mmsi=preselected_mmsi)
        else:
            final_test_df = RP.load_samples(
                args.final_test_samples, mmsi="ALL", formal=False, expected_split=None)
        purge = (float(args.holdout_purge_min) if args.holdout_purge_min is not None
                 else float(max(train_df["h_min"].max(), validation_df["h_min"].max(),
                                final_test_df["h_min"].max())))
        RP.validate_holdout_disjointness(
            train_df, validation_df, final_test_df, purge_min=purge,
            require_real_weather=(args.final_weather_mode == "real"),
            require_real_recovery_state=True)
        args._holdout_disjointness_verified = True
        args._holdout_independence_verified = False
        # 文件哈希相同是最明显的误用；行级/时间区间检查由上函数进一步防护。
        hashes = [EU.sha256_file(x) for x in (args.xi_train_samples,
                                               args.validation_samples,
                                               args.final_test_samples)]
        if len(set(hashes)) != 3:
            raise SystemExit("train/validation/test 文件哈希重复，已拒绝数据泄漏。")
        if formal_samples:
            _bind_formal_sample_hashes(args, final_test_metadata=final_test_df)
    # SAA 经验样本登记：机制模式无样本时允许矩匹配高斯并如实标注；
    # 正式 E2 必须来自 train 经验样本，禁止回退。
    saa_csv = (Path(args.xi_train_samples) if args.xi_train_samples is not None else
               (Path(args.saa_samples) if args.saa_samples else
                (Path(__file__).resolve().parent / "tracks" / "xi_samples_caseB.csv")))
    forbidden = {EU.sha256_file(x) for x in (args.validation_samples, args.final_test_samples) if x is not None}
    saa_cells_loaded, saa_input_usable = _register_saa_baseline(
        args, saa_csv, forbidden,
        mmsi=(preselected_mmsi if formal_samples else "ALL"))
    if (args.study_mode == "formal" and args.exp in ("all", "E2_robust")
            and _preflight_e2_criteria is not None and "SAA" in _preflight_e2_criteria
            and saa_cells_loaded <= 0):
        raise SystemExit("正式 E2 选择了 SAA 时必须使用 train 经验样本；禁止回退矩匹配高斯。")
    p = M.Params()
    p.quick_inspection_capacity = int(getattr(args, "quick_inspection_capacity", 1))
    if args.insp_min is not None:
        p.tau_insp = float(args.insp_min) * 60.0
        log.warning("场景参数 tau_insp=%.1f min/台(--insp-min): 物理口径改变, 全部实验须同一取值。", args.insp_min)
    # SOC 校正口径（默认 none；geo2d 开启即切换口径）
    p.soc_correction = args.soc_correction
    p.soc_risk_allocation = getattr(args, "soc_risk_allocation", "optimized")
    p.time_recourse_mode = str(getattr(args, "time_recourse", "wait_and_speed"))
    p.speed_adjustable = (p.time_recourse_mode == "wait_and_speed")
    p.validate_contract(formal=(args.study_mode == "formal"))
    log.info("固定接地时间 recourse=%s (wait=%s, speed=%s)", p.time_recourse_mode, True, p.speed_adjustable)
    if args.soc_correction != "none":
        log.warning("SOC 校正=%s: 判据口径改变(更保守), 与 soc_correction=none 的结果不可混排。",
                    args.soc_correction)
    # 先用 ALL 的 horizon 支持构造并选择真实航迹；选定 MMSI 后立即切换为具体船的分层 ξ。
    # 这样窗口选择仍不依赖未来误差结果，而路线风险不再无条件混用另一艘船的操纵统计。
    _pr, _pr_mode = _resolve_pair_radius(args, p, xi_amb, turbines)
    args._pair_radius_m, args._pair_radius_mode = _pr, _pr_mode
    opts, reach, kind, T_eff, wx0 = build_launch_options(
        turbines, lat0lon0, track_csv, xi_amb, wx_df, args.window_min, args.dtau_min,
        _pr, hs_quantile=args.hs_quantile,
        track_start_min=args.track_start_min, allow_synth=args.allow_synth,
        infarm_radius_m=args.infarm_radius,
        predictor=args.recovery_predictor,
        weather_alignment=args.weather_alignment,
        formal=(args.study_mode == "formal"),
        bound_track_mmsi=(preselected_mmsi if args.study_mode == "formal" else None))
    # build_launch_options() may receive a candidate list in mechanism mode.
    # Once it selects a concrete track, pin every later E1/E2/A stage and the
    # full_suite Step14 subprocess to that exact file.  This also prevents
    # provenance/hash code from later receiving a list where a path is required.
    selected_track_csv = wx0.get("selected_track_csv")
    if selected_track_csv:
        track_csv = Path(selected_track_csv)
        args.track_csv = str(track_csv)
        args._resolved_track_csv = str(track_csv)
    elif isinstance(track_csv, (list, tuple)):
        # No candidate could be loaded and build_launch_options() fell back to
        # a synthetic track.  There is therefore no real track path to hash.
        track_csv = None
        args._resolved_track_csv = None

    selected_mmsi = str(wx0.get("selected_track_mmsi") or "").strip()
    if args.study_mode == "formal":
        if not selected_mmsi or selected_mmsi != str(preselected_mmsi):
            raise SystemExit(
                "正式轨迹窗口解析出的 MMSI 与预绑定样本 MMSI 不一致: "
                f"track_window={selected_mmsi!r}, samples={preselected_mmsi!r}")
        if str(getattr(xi_amb, "selected_mmsi", "")) != selected_mmsi:
            raise SystemExit(
                "formal Xi ambiguity 未绑定当前轨迹 MMSI: "
                f"xi={getattr(xi_amb, 'selected_mmsi', None)!r}, track={selected_mmsi!r}")
        if bool(getattr(xi_amb, "cross_vessel_pooling", False)):
            raise SystemExit("formal Xi ambiguity 检测到 cross_vessel_pooling=True；禁止签发/运行正式实验。")
        src = (f"purged-train:{Path(args.xi_train_samples).name}:"
               f"mmsi={selected_mmsi}:{EU.sha256_file(args.xi_train_samples)}")
        if saa_input_usable and saa_csv.is_file():
            RM.load_saa_empirical(
                saa_csv, mmsi=selected_mmsi,
                require_current_contract=True, allow_pooled_fallback=False)
    else:
        xi_source_path = (Path(getattr(xi_amb, "source_path", ""))
                          if getattr(xi_amb, "source_path", None) else None)
        if selected_mmsi and xi_source_path is not None and xi_source_path.is_file():
            old_h = tuple(RM.decision_horizons_of(xi_amb))
            xi_selected = M.XiAmbiguity.from_csv_hierarchical(
                xi_source_path, selected_mmsi, formal=False)
            new_h = tuple(RM.decision_horizons_of(xi_selected))
            xi_amb = xi_selected
            src = f"分层 ξ 矩 {xi_source_path.name}:mmsi={selected_mmsi}"
            if new_h != old_h:
                opts, reach, kind, T_eff, wx0 = build_launch_options(
                    turbines, lat0lon0, track_csv, xi_amb, wx_df, args.window_min, args.dtau_min,
                    _pr, hs_quantile=args.hs_quantile,
                    track_start_min=args.track_start_min, allow_synth=args.allow_synth,
                    infarm_radius_m=args.infarm_radius, predictor=args.recovery_predictor,
                    weather_alignment=args.weather_alignment,
        formal=(args.study_mode == "formal"))
    args._xi_source = src
    args._resolved_xi_mmsi = str(getattr(xi_amb, "selected_mmsi", "ALL"))
    args._resolved_xi_predictor = str(getattr(xi_amb, "predictor", "unknown"))
    args._resolved_xi_predictor_contract = str(getattr(xi_amb, "predictor_contract", "unknown"))
    expected_predictor = str(args.recovery_predictor)
    expected_predictor_contract = str(M.XI_PREDICTOR_CONTRACTS.get(expected_predictor, "unknown"))
    if args._resolved_xi_predictor != expected_predictor:
        raise SystemExit(
            f"ξ predictor={args._resolved_xi_predictor!r} 与运行预测器={expected_predictor!r} 不一致；"
            "禁止混用预测器生成的 ξ。请用当前预测器重新运行 step7。")
    if expected_predictor_contract != "unknown" and args._resolved_xi_predictor_contract != expected_predictor_contract:
        raise SystemExit(
            f"ξ predictor_contract={args._resolved_xi_predictor_contract!r} 与运行合同="
            f"{expected_predictor_contract!r} 不一致；禁止加载旧 ξ 统计。")
    resolved_epoch_contract = str(getattr(xi_amb, "timestamp_epoch_contract", "unknown"))
    if resolved_epoch_contract != str(M.XI_TIMESTAMP_EPOCH_CONTRACT):
        raise SystemExit(
            f"ξ timestamp_epoch_contract={resolved_epoch_contract!r} 与运行合同="
            f"{M.XI_TIMESTAMP_EPOCH_CONTRACT!r} 不一致；旧 ξ 可能受 pandas 时间精度缩放影响，"
            "请重新运行 step7。")
    if args.study_mode != "formal" and not bool(getattr(xi_amb, "valid_for_formal_data", False)):
        log.warning("当前 ξ 数据合同 valid_for_formal=False（例如 overlap=all 或 purge=0）；"
                    "仅可用于机制诊断，不得形成正式可靠性结论。")
    log.info("ξ 预测器合同通过: predictor=%s predictor_contract=%s epoch_contract=%s mmsi=%s",
             args._resolved_xi_predictor, args._resolved_xi_predictor_contract,
             resolved_epoch_contract, args._resolved_xi_mmsi)
    if args.weather_drcc == "on":
        if args.weather_moments_csv is not None:
            wamb = RM.weather_ambiguity_from_moments_csv(
                args.weather_moments_csv, RM.decision_horizons_of(xi_amb), formal=True)
        else:
            if args.study_mode == "formal":
                raise SystemExit("formal 模式必须提供 --weather-moments-csv；禁止使用相邻再分析差分天气代理。")
            wamb = RM.weather_ambiguity_from_series(
                wx_df, RM.decision_horizons_of(xi_amb), scale=args.weather_scale)
    else:
        wamb = None
    _assert_weather_xi_train_binding(
        wamb, args.xi_train_samples, formal=(args.study_mode == "formal"))
    args._weather_uncertainty_source = str(getattr(wamb, "source", "off")) if wamb is not None else "off"
    args._weather_formal_eligible = bool(getattr(wamb, "formal_eligible", False)) if wamb is not None else True
    if args.study_mode == "formal" and wamb is not None and not args._weather_formal_eligible:
        raise SystemExit("formal 模式天气歧义集未通过 real-history no-leak 正式合同。")
    _record_formal_instance_provenance(
        args, mmsi=(preselected_mmsi if args.study_mode == "formal" else selected_mmsi),
        track_csv=track_csv, xi_train_samples=args.xi_train_samples,
        weather_moments_csv=args.weather_moments_csv, weather_uncertainty=wamb,
        launch_formal=(args.study_mode == "formal"))
    args._wx0 = wx0
    print(f"就绪: {kind} | 窗 T={T_eff:.0f}min Δτ={args.dtau_min:.0f} | reach={len(reach)} 台 | "
          f"天气DRCC={'ON' if wamb is not None else 'OFF'} | 甲板={args.deck_mode} | "
          f"SAA={RM.SAA_SOURCE} | tau_insp={p.tau_insp/60:.1f}min/台")
    # Standalone E2/A must prove that an on-disk E1 selection belongs to the
    # current binary64 base instance, not merely to the same software version.
    args._expected_e1_resume_input_sha256 = _resume_context_sha256(
        reach=reach, launch_options=opts, params=p, xi_ambiguity=xi_amb,
        weather_uncertainty=wamb, track_kind=kind, T_eff_min=float(T_eff))

    if args.exp == "E1_lex_certify":
        print("\n[E1_lex_certify] direct exact lexicographic certification for one fixed resource point")
        lex = E1_lex_certify(
            reach, opts, p, xi_amb, wamb, RESULTS / "E1_lex_certify",
            args, kind, T_eff)
        print(
            "coverage: %s <= C* <= %s | coverage_optimal=%s\n"
            "energy: incumbent=%s Wh lower_bound=%s Wh | energy_optimal=%s\n"
            "lexicographic_optimal=%s global_certificate_available=%s | termination=%s"
            % (
                lex.get("coverage_incumbent"), lex.get("coverage_upper_bound"),
                lex.get("coverage_optimal"),
                lex.get("energy_incumbent_Wh"), lex.get("energy_lower_bound_Wh"),
                lex.get("energy_optimal"),
                lex.get("lexicographic_optimal"),
                _global_certificate_flag(lex),
                lex.get("termination_reason"),
            ))
        return

    if args.exp in ("all", "E1_frontier"):
        print("\n[E1_frontier] 三轴主实验: UAV(%s) × K(%s) × B(%s 基础网格, b_auto=%s cap=%d) "
              "stops_cap=%s(判据=VP, 词典序 L1覆盖→L2能耗)"
              % (args.e1_uavs, args.fleet_ks, args.e1_batteries,
                 args.e1_b_auto, args.e1_b_cap, args.stops_cap))
        df1 = E1_frontier(reach, opts, p, xi_amb, wamb, RESULTS / "E1_frontier", args, kind, T_eff)
        args._e1_sel_df = e1_select_from_df(
            df1, frac=args.knee_frac, order=args.knee_order,
            patience=max(1, int(args.e1_sat_patience)))   # 同进程 E2/A 自动回填直通
        with pd.option_context("display.max_columns", 16, "display.width", 220):
            print(df1[["uav", "K", "batteries", "covered", "safe_served", "per_battery",
                       "flights", "mean_stops", "max_stops_requested", "max_stops_effective",
                       "max_stops_observed", "stops_cap_hit", "energy_Wh",
                       "emp_viol", "plan_holds"]].to_string(index=False))
        print("\n(提示: 跑完可用 `python step13_experiment_model.py --exp E1_select` 出选型表与 E2/A 回填命令)")
        if full_suite:
            print("\n[full_suite] E1_frontier 完成；进入 v12 direct targeted resource closure")

    if args.exp in ("all", "E1_knee_refine"):
        print("\n[E1_knee_refine] exact target predecessor certificates + knee-only full lex")
        df1, sel1 = E1_knee_refine(
            reach, opts, p, xi_amb, wamb, RESULTS / "E1_frontier",
            args, kind, T_eff)
        args._e1_sel_df = sel1
        with pd.option_context("display.max_columns", 40, "display.width", 260):
            print(sel1.to_string(index=False))
        if full_suite:
            print("\n[full_suite] targeted E1 closure 完成，刷新 E1_select")
            E1_select(args)

    if args.exp in ("all", "E2_robust"):
        _resolve_e2_config(args)                   # 更新: 选型自动回填(auto)或手动直通
        print("\n[E2_robust] 分布鲁棒 vs 其他鲁棒/随机方法(7 判据 × 窗分位; uav=%s K=%d B=%s)"
              % (args.uav, args.k, args.batteries))
        dfr, dfs = E2_robust(turbines, lat0lon0, track_csv, xi_amb, wx_df, p, wamb,
                             RESULTS / "E2_robust_comparison", args)
        print(dfs.to_string(index=False))

    if full_suite:
        if not _formal_e2_final_test_authorized(args):
            raise SystemExit("formal A1/A2 full-suite requires the same structured E1 freeze as E2 final-test")
        _run_algorithm_full_suite(args)


if __name__ == "__main__":
    main()
