#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step15_replay.py — 阶段5无泄漏联合回放验证。

正式协议与当前模型逐项对齐：
  1. 使用阶段3输出的逐轨迹 purged train/validation/test；训练段只估计船位误差矩，
     测试段只用于冻结方案后的回放。
  2. 外层船位预测误差严格按起飞时可观测状态 c(tau) 和精确预测时长 h 取样；预测
     回收状态只负责判断是否允许船尾伴飞/着舰，二者不得混用。
  3. 联合失败事件覆盖能量、时间、波浪门、风速门、航段空速、对接储备和船尾伴飞。
     回收时长 h 是优化决策，计划回收点为该 h 的预测船位；实现回收位置不确定性由
     同一 h 的真实 AIS/CV 船位预测误差 xi_h 描述。传感器级 acquisition error 不属于
     当前有限模型，也不允许通过填零或合成样本伪装成正式实测通道。
  4. 每个架次分别计算联合失败率。正式统计证书使用 Bonferroni 同时修正后的逐架次
     单侧上置信界最大值；路线×样本的合并二项区间仅保留为非正式诊断，避免同一历史
     样本被多条路线复用时置信区间虚假变窄。
  5. 只有真实天气误差、真实回收状态、完整模型内事件通道和已验证的独立留出协议同时
     成立时，结果才具备当前有限模型范围内的真实联合可靠性声明资格。传感器级终端
     获取/定位可靠性明确不在该声明范围内。

无额外脚本依赖；正式数据由 step7 --dump-samples 生成，独立样本协议由本文件现有
validate_holdout_independence() 检查。对应 doc_model、doc_experiments 和 doc_process。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

import step9_model as M
import step10_model_routing as RM
import step11_algorithm_route_drcc as RA
import step7_compute_xi as S7

log = logging.getLogger("replay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS = Path(__file__).resolve().parent / "results" / "validation" / "replay"


def _stable_scalar_fp(value):
    """Stable text fingerprint for holdout-event auditing, not model semantics."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "+inf" if v > 0 else "-inf"
        return v.hex()
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return int(value)
    if pd.isna(value):
        return "nan"
    return str(value)


def _holdout_cell_fingerprint(cell_df: pd.DataFrame) -> str:
    """Hash the actual ordered validation rows used by one route.

    This is an audit-only identity.  It is intentionally not used to relax the
    formal Bonferroni selection gate: two routes that happen to share an
    empirical failure mask are not assumed to represent the same population
    hypothesis unless structural equivalence is proved separately.
    """
    preferred = [
        "h_min", "h", "c_state", "mmsi", "split",
        "xi_e_m", "xi_n_m",
        "wind_error_e_ms", "wind_error_n_ms", "hs_error_m",
        "actual_recovery_state", "recovery_state_actual", "timestamp",
    ]
    cols = [c for c in preferred if c in cell_df.columns]
    if not cols:
        cols = list(cell_df.columns)
    h = hashlib.sha256()
    h.update(("cols:" + "|".join(map(str, cols))).encode("utf-8"))
    for row in cell_df[cols].itertuples(index=False, name=None):
        payload = [_stable_scalar_fp(v) for v in row]
        h.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _failure_event_fingerprint(cell_sha256: str, mask, required_events) -> tuple[str, str]:
    arr = np.asarray(mask, dtype=np.uint8).reshape(-1)
    packed = np.packbits(arr, bitorder="little").tobytes()
    mask_sha = hashlib.sha256(
        len(arr).to_bytes(8, "little", signed=False) + packed).hexdigest()
    payload = dict(
        cell_sha256=str(cell_sha256),
        mask_sha256=str(mask_sha),
        n=int(len(arr)),
        required_events=sorted(map(str, required_events)),
    )
    event_sha = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return mask_sha, event_sha


# =============================================================================
# 1. 样本载入 / 时间切分 / 由样本建模糊集
# =============================================================================
SAMPLE_KEY_COLUMNS = ["mmsi", "h_min", "c_state", "t0_epoch"]
REAL_WEATHER_ERROR_COLUMNS = ["wind_error_e_ms", "wind_error_n_ms", "hs_error_m"]
REAL_WEATHER_PROVENANCE_COLUMNS = [
    "weather_predictor", "weather_predictor_contract",
    "weather_timestamp_epoch_contract", "weather_truth_contract",
    "weather_data_contract", "weather_valid_for_formal",
    "weather_source_sha256", "weather_train_source_sha256",
]
ACTUAL_RECOVERY_STATE_COLUMNS = (
    "actual_recovery_state",
    "recovery_state_actual",
)
ACTUAL_RECOVERY_DIAGNOSTIC_COLUMNS = (
    "actual_recovery_speed_ms",
    "actual_recovery_turn_rate_deg_min",
    "actual_recovery_displacement_window_m",
)


def _strict_true_column(df: pd.DataFrame, column: str) -> bool:
    vals = df[column].astype(str).str.strip().str.lower()
    return bool(vals.isin({"true", "1", "yes"}).all())


def _single_text_value(df: pd.DataFrame, column: str, default: str = "unknown") -> str:
    if column not in df.columns:
        return str(default)
    vals = sorted({str(x).strip() for x in df[column].dropna() if str(x).strip()})
    if len(vals) > 1:
        raise ValueError(f"replay ξ samples {column} mixes multiple values: {vals}")
    return vals[0] if vals else str(default)


def _validate_formal_xi_sample_contract(df: pd.DataFrame, expected_split: str | None) -> dict:
    """Validate formal Xi sample provenance before any dtype normalization.

    The returned metadata may be attached to an ``XiAmbiguity`` rebuilt from the
    samples.  We deliberately do not manufacture missing provenance: a missing or
    mixed contract is a formal-data error, not an invitation to fill in cv_noleak.
    """
    required = {
        "predictor", "predictor_contract", "timestamp_epoch_contract",
        "sample_overlap_policy", "purge_min", "moments_source",
        "valid_for_formal", "split", "h_min",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"formal replay ξ samples missing contract columns {missing}")

    predictor = _single_text_value(df, "predictor")
    predictor_contract = _single_text_value(df, "predictor_contract")
    epoch_contract = _single_text_value(df, "timestamp_epoch_contract")
    overlap_policy = _single_text_value(df, "sample_overlap_policy")
    moments_source = _single_text_value(df, "moments_source")
    split = _single_text_value(df, "split")

    if predictor != "cv_noleak":
        raise ValueError(f"formal replay ξ samples predictor mismatch: {predictor!r}")
    if predictor_contract != M.XI_PREDICTOR_CONTRACTS["cv_noleak"]:
        raise ValueError("formal replay ξ samples predictor_contract mismatch")
    if epoch_contract != M.XI_TIMESTAMP_EPOCH_CONTRACT:
        raise ValueError("formal replay ξ samples epoch contract mismatch")
    if overlap_policy != "nonoverlap":
        raise ValueError("formal replay ξ samples require sample_overlap_policy=nonoverlap")
    if moments_source != "train":
        raise ValueError("formal replay ξ samples require moments_source=train")
    if expected_split is not None and split != str(expected_split):
        raise ValueError(
            f"formal replay ξ samples split mismatch: expected={expected_split!r}, found={split!r}")
    if not _strict_true_column(df, "valid_for_formal"):
        raise ValueError("formal replay ξ samples contain valid_for_formal=False")

    h_raw = pd.to_numeric(df["h_min"], errors="raise").to_numpy(float)
    allowed = {float(h).hex() for h in M.XI_FORMAL_HORIZON_GRID_MIN}
    bad = [float(h) for h in h_raw if float(h).hex() not in allowed]
    if bad:
        raise ValueError(f"formal replay ξ samples contain off-grid h_min={bad[0]!r}")
    purge = pd.to_numeric(df["purge_min"], errors="raise").to_numpy(float)
    if not bool(np.isfinite(purge).all()):
        raise ValueError("formal replay ξ samples purge_min contains non-finite values")
    max_h = float(np.max(h_raw))
    if not bool((purge >= max_h).all()):
        raise ValueError("formal replay ξ samples require purge_min >= max horizon")

    return {
        "predictor": predictor,
        "predictor_contract": predictor_contract,
        "timestamp_epoch_contract": epoch_contract,
        "sample_overlap_policy": overlap_policy,
        "moments_source": moments_source,
        "valid_for_formal_data": True,
        "purge_min": float(np.min(purge)),
        "split": split,
    }


def _validate_weather_provenance_metadata(df: pd.DataFrame, available_columns) -> None:
    """Validate weather provenance without requiring outcome values to be materialized."""
    available = set(map(str, available_columns))
    has_weather_schema = all(c in available for c in REAL_WEATHER_ERROR_COLUMNS)
    if not has_weather_schema:
        return
    miss_weather = set(REAL_WEATHER_PROVENANCE_COLUMNS) - available
    if miss_weather:
        raise ValueError(f"真实天气误差列存在但 provenance 缺列 {sorted(miss_weather)}")
    if set(df["weather_predictor"].astype(str)) != {"weather_speed_primary_coherent_noleak"}:
        raise ValueError("weather predictor mismatch")
    if set(df["weather_predictor_contract"].astype(str)) != {RM.WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"]}:
        raise ValueError("weather predictor contract mismatch")
    if set(df["weather_timestamp_epoch_contract"].astype(str)) != {RM.WEATHER_TIMESTAMP_EPOCH_CONTRACT}:
        raise ValueError("weather timestamp contract mismatch")
    if set(df["weather_truth_contract"].astype(str)) != {RM.WEATHER_TRUTH_CONTRACT}:
        raise ValueError("weather truth contract mismatch")
    if set(df["weather_data_contract"].astype(str)) != {RM.WEATHER_FORMAL_DATA_CONTRACT}:
        raise ValueError("weather data contract mismatch")
    _wvalid = df["weather_valid_for_formal"].astype(str).str.strip().str.lower()
    if not bool((_wvalid == "true").all()):
        raise ValueError("weather holdout contains weather_valid_for_formal=False")
    for _c in ("weather_source_sha256", "weather_train_source_sha256"):
        _vals = set(df[_c].astype(str))
        if len(_vals) != 1:
            raise ValueError(f"weather holdout {_c} must be constant")
        _v = next(iter(_vals))
        if len(_v) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in _v):
            raise ValueError(f"weather holdout {_c} is not SHA-256")


def load_sample_metadata(path: Path, mmsi="ALL", *, formal: bool = False,
                         expected_split: str | None = None) -> pd.DataFrame:
    """Load provenance/timing identity without materializing holdout outcome values.

    This is the only formal pre-freeze loader for the final-test file.  It reads the
    CSV header plus columns needed for split/provenance/sample-key/temporal-independence
    checks.  Xi errors, weather errors, and actual recovery outcomes are deliberately
    absent from the returned frame; their *schema presence* is retained in attrs.
    """
    path = Path(path)
    header = list(pd.read_csv(path, nrows=0).columns)
    available = set(map(str, header))
    contract_cols = {
        "predictor", "predictor_contract", "timestamp_epoch_contract",
        "sample_overlap_policy", "purge_min", "moments_source",
        "valid_for_formal", "split", "h_min",
    }
    identity_candidates = {
        "mmsi", "h_min", "c_state", "t0_epoch", "t0_iso", "t1_epoch",
        "source_track_id", "source_track", "segment_id", "sample_id",
    }
    provenance = set(REAL_WEATHER_PROVENANCE_COLUMNS)
    usecols = sorted((contract_cols | identity_candidates | provenance) & available)
    df = pd.read_csv(path, usecols=usecols)
    formal_meta = (_validate_formal_xi_sample_contract(df, expected_split)
                   if bool(formal) else None)
    if "predictor" in df.columns and set(df["predictor"].dropna().astype(str)) != {"cv_noleak"}:
        raise ValueError("replay ξ samples predictor mismatch")
    if ("predictor_contract" in df.columns
            and set(df["predictor_contract"].dropna().astype(str))
            != {M.XI_PREDICTOR_CONTRACTS["cv_noleak"]}):
        raise ValueError("replay ξ samples predictor_contract mismatch")
    if ("timestamp_epoch_contract" in df.columns
            and set(df["timestamp_epoch_contract"].dropna().astype(str))
            != {M.XI_TIMESTAMP_EPOCH_CONTRACT}):
        raise ValueError("replay ξ samples epoch contract mismatch")
    if "t0_epoch" not in df.columns and "t0_iso" in df.columns:
        ts = pd.to_datetime(df["t0_iso"], utc=True, errors="coerce")
        df["t0_epoch"] = S7.to_epoch_seconds_utc(ts)
    need = {"h_min", "c_state", "t0_epoch"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"样本文件 {path} 缺少 metadata 列 {sorted(miss)}")
    if "mmsi" not in df.columns:
        df["mmsi"] = "UNKNOWN"
    _mmsi_filter = str(mmsi).strip()
    if _mmsi_filter.upper() != "ALL":
        df = df[df["mmsi"].astype(str) == _mmsi_filter].copy()
        if df.empty:
            raise ValueError(f"样本文件 {path} 中没有 mmsi={_mmsi_filter} 的样本；禁止跨船回退。")
    for c in ("h_min", "t0_epoch", "t1_epoch", "purge_min"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["h_min", "c_state", "t0_epoch"]).copy()
    if formal:
        allowed = {float(h).hex() for h in M.XI_FORMAL_HORIZON_GRID_MIN}
        bad = [float(h) for h in df["h_min"].to_numpy(float)
               if float(h).hex() not in allowed]
        if bad:
            raise ValueError(f"formal replay ξ metadata contain off-grid h_min={bad[0]!r}")
    df["h_min"] = df["h_min"].astype(int)
    if "t1_epoch" not in df.columns:
        df["t1_epoch"] = df["t0_epoch"] + 60.0 * df["h_min"]
    else:
        df["t1_epoch"] = pd.to_numeric(df["t1_epoch"], errors="coerce")
        df["t1_epoch"] = df["t1_epoch"].fillna(df["t0_epoch"] + 60.0 * df["h_min"])
    _validate_weather_provenance_metadata(df, available)
    df = df.sort_values(["t0_epoch", "mmsi", "h_min", "c_state"]).reset_index(drop=True)
    if formal_meta is not None:
        df.attrs["formal_xi_sample_contract"] = dict(formal_meta)
        df.attrs["source_path"] = str(path.expanduser().resolve())
    df.attrs["selected_mmsi_filter"] = _mmsi_filter
    df.attrs["cross_vessel_pooling"] = bool(_mmsi_filter.upper() == "ALL")
    df.attrs["available_columns"] = tuple(sorted(available))
    df.attrs["outcomes_materialized"] = False
    return df


def load_samples(path: Path, mmsi="ALL", *, formal: bool = False,
                 expected_split: str | None = None) -> pd.DataFrame:
    """读取误差样本并规范时间字段。

    ``mmsi='ALL'`` 表示保留所有真实船舶样本，而不是筛选一个字面值为 ALL 的行。
    旧实现会把 step7 导出的真实样本全部筛空，导致 real_holdout 静默变成 n=0。

    In formal mode the full predictor/timestamp/split/nonoverlap/purge contract is
    validated *before* h_min is normalized to integer minutes.
    """
    df = pd.read_csv(path)
    formal_meta = (_validate_formal_xi_sample_contract(df, expected_split)
                   if bool(formal) else None)
    if "predictor" in df.columns and set(df["predictor"].dropna().astype(str)) != {"cv_noleak"}:
        raise ValueError("replay ξ samples predictor mismatch")
    if ("predictor_contract" in df.columns
            and set(df["predictor_contract"].dropna().astype(str))
            != {M.XI_PREDICTOR_CONTRACTS["cv_noleak"]}):
        raise ValueError("replay ξ samples predictor_contract mismatch")
    if ("timestamp_epoch_contract" in df.columns
            and set(df["timestamp_epoch_contract"].dropna().astype(str))
            != {M.XI_TIMESTAMP_EPOCH_CONTRACT}):
        raise ValueError("replay ξ samples epoch contract mismatch")
    if "t0_epoch" not in df.columns and "t0_iso" in df.columns:
        ts = pd.to_datetime(df["t0_iso"], utc=True, errors="coerce")
        df["t0_epoch"] = S7.to_epoch_seconds_utc(ts)
    need = {"h_min", "c_state", "t0_epoch", "xi_e_m", "xi_n_m"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"样本文件 {path} 缺少列 {sorted(miss)}")
    if "mmsi" not in df.columns:
        df["mmsi"] = "UNKNOWN"
    _mmsi_filter = str(mmsi).strip()
    if _mmsi_filter.upper() != "ALL":
        df = df[df["mmsi"].astype(str) == _mmsi_filter].copy()
        if df.empty:
            raise ValueError(f"样本文件 {path} 中没有 mmsi={_mmsi_filter} 的样本；禁止跨船回退。")
    numeric_cols = ["h_min", "t0_epoch", "xi_e_m", "xi_n_m", "t1_epoch"]
    numeric_cols += [c for c in REAL_WEATHER_ERROR_COLUMNS if c in df.columns]
    _has_weather = all(c in df.columns for c in REAL_WEATHER_ERROR_COLUMNS)
    if _has_weather:
        _validate_weather_provenance_metadata(df, set(df.columns))
    numeric_cols += [c for c in ACTUAL_RECOVERY_DIAGNOSTIC_COLUMNS if c in df.columns]
    for c in dict.fromkeys(numeric_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["h_min", "c_state", "t0_epoch", "xi_e_m", "xi_n_m"]).copy()
    df["h_min"] = df["h_min"].astype(int)
    if "t1_epoch" not in df.columns:
        df["t1_epoch"] = df["t0_epoch"] + 60.0 * df["h_min"]
    else:
        df["t1_epoch"] = pd.to_numeric(df["t1_epoch"], errors="coerce")
        df["t1_epoch"] = df["t1_epoch"].fillna(df["t0_epoch"] + 60.0 * df["h_min"])
    df = df.sort_values(["t0_epoch", "mmsi", "h_min", "c_state"]).reset_index(drop=True)
    if formal_meta is not None:
        df.attrs["formal_xi_sample_contract"] = dict(formal_meta)
        df.attrs["source_path"] = str(Path(path).expanduser().resolve())
    df.attrs["selected_mmsi_filter"] = _mmsi_filter
    df.attrs["cross_vessel_pooling"] = bool(_mmsi_filter.upper() == "ALL")
    return df


def temporal_split(df: pd.DataFrame, train_frac=0.6):
    """向后兼容的两段时间切分；新实验应使用 ``purged_temporal_split``。"""
    d = df.sort_values("t0_epoch")
    k = int(len(d) * train_frac)
    return d.iloc[:k].copy(), d.iloc[k:].copy()


def _source_group_column(df: pd.DataFrame) -> str:
    """选择逐轨迹切分键；优先使用阶段3的绝对路径标识。"""
    for c in ("source_track_id", "source_track", "mmsi"):
        if c in df.columns:
            return c
    return "__single_source__"


def purged_temporal_split(df: pd.DataFrame, train_frac=0.60, validation_frac=0.20,
                          purge_min=None):
    """逐轨迹生成 train/validation/test，并剔除跨边界预测区间。

    每个误差样本使用区间 ``[t0,t1]``。不同轨迹各自计算分位点，避免一条长轨迹
    主导另一条短轨迹的切分；边界两侧保留 ``purge_min`` 隔离带。返回的 meta 同时
    包含逐轨迹边界与合计丢弃行数。
    """
    d = df.copy()
    if "t1_epoch" not in d.columns:
        d["t1_epoch"] = d["t0_epoch"] + 60.0 * d["h_min"]
    if not (0 < train_frac < 1 and 0 < validation_frac < 1
            and train_frac + validation_frac < 1):
        raise ValueError("train_frac/validation_frac 必须为正且和小于 1")
    if d.empty:
        raise ValueError("样本为空，无法切分")
    purge_s = 60.0 * (float(purge_min) if purge_min is not None
                      else float(d["h_min"].max()))
    key_col = _source_group_column(d)
    if key_col == "__single_source__":
        d[key_col] = "ALL"
    train_parts, val_parts, test_parts, per_source = [], [], [], []
    for source, g in d.groupby(d[key_col].astype(str), sort=True):
        g = g.sort_values("t0_epoch").copy()
        t = np.sort(g["t0_epoch"].dropna().to_numpy(float))
        if len(t) < 3:
            raise ValueError(f"轨迹 {source!r} 样本量过小，无法做三段时间切分")
        train_end = float(np.quantile(t, train_frac))
        validation_end = float(np.quantile(t, train_frac + validation_frac))
        validation_start = train_end + purge_s
        test_start = validation_end + purge_s
        tr = g[g["t1_epoch"] <= train_end].copy()
        va = g[(g["t0_epoch"] >= validation_start)
               & (g["t1_epoch"] <= validation_end)].copy()
        te = g[g["t0_epoch"] >= test_start].copy()
        train_parts.append(tr); val_parts.append(va); test_parts.append(te)
        per_source.append(dict(source=str(source), train_end_epoch=train_end,
                               validation_start_epoch=validation_start,
                               validation_end_epoch=validation_end,
                               test_start_epoch=test_start,
                               train_rows=int(len(tr)), validation_rows=int(len(va)),
                               test_rows=int(len(te)),
                               dropped_rows=int(len(g)-len(tr)-len(va)-len(te))))
    train = pd.concat(train_parts, ignore_index=True) if train_parts else d.iloc[:0].copy()
    validation = pd.concat(val_parts, ignore_index=True) if val_parts else d.iloc[:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else d.iloc[:0].copy()
    meta = dict(purge_min=purge_s / 60.0,
                source_key=key_col,
                per_source=per_source,
                dropped_rows=int(len(d)-len(train)-len(validation)-len(test)))
    return train, validation, test, meta

def _sample_keys(df: pd.DataFrame):
    """构造正式样本键；轨迹身份必须使用路径/内容派生的 ``source_track_id``。"""
    if "source_track_id" not in df.columns:
        raise ValueError("正式独立性审计缺少 source_track_id；禁止退回同名文件或 MMSI 合并。")
    cols = ["source_track_id"]
    if "segment_id" in df.columns:
        cols.append("segment_id")
    if "sample_id" in df.columns:
        cols.append("sample_id")
    else:
        cols.extend(c for c in SAMPLE_KEY_COLUMNS if c in df.columns)
    if len(cols) <= 1:
        raise ValueError("正式独立性审计缺少 sample_id 或时间/时长样本键。")
    return set(map(tuple, df[cols].astype(str).itertuples(index=False, name=None)))

def validate_holdout_disjointness(train_df: pd.DataFrame, validation_df: pd.DataFrame,
                                  test_df: pd.DataFrame, purge_min=0.0,
                                  require_real_weather=False,
                                  require_real_recovery_state=False):
    """Fail-closed check for disjoint, purged temporal holdouts.

    This validates sample/key separation and the requested temporal embargo. It
    deliberately does *not* claim stochastic independence: serially correlated
    AIS/weather failures can remain dependent after a finite purge.
    """
    named = {"train": train_df.copy(), "validation": validation_df.copy(), "test": test_df.copy()}
    for name, d in named.items():
        if d.empty:
            raise ValueError(f"{name} 样本为空，不能声明有效留出")
        if "t1_epoch" not in d.columns:
            d["t1_epoch"] = d["t0_epoch"] + 60.0 * d["h_min"]
        named[name] = d
    max_h = max(float(named[x]["h_min"].max()) for x in named)
    if float(purge_min) < max_h:
        raise ValueError(f"purge_min={float(purge_min):g} 小于最大预测时长 {max_h:g} min。")
    kt, kv, ks = (_sample_keys(named[x]) for x in ("train", "validation", "test"))
    if kt & kv or kt & ks or kv & ks:
        raise ValueError("train/validation/test 存在重复样本键，已拒绝数据泄漏")
    gap = 60.0 * float(purge_min)
    key_col = "source_track_id"
    if not all(key_col in named[x].columns for x in named):
        raise ValueError("train/validation/test 必须全部提供 source_track_id。")
    groups = sorted(set().union(*(set(named[x][key_col].astype(str)) for x in named)))
    checked_sources = 0
    for source in groups:
        parts = {name: d[d[key_col].astype(str) == str(source)] for name, d in named.items()}
        if any(v.empty for v in parts.values()):
            raise ValueError(f"轨迹 {source!r} 在 train/validation/test 中至少一段为空")
        if float(parts["train"]["t1_epoch"].max()) + gap > float(parts["validation"]["t0_epoch"].min()):
            raise ValueError(f"轨迹 {source!r}: train 与 validation 区间或 purge gap 重叠")
        if float(parts["validation"]["t1_epoch"].max()) + gap > float(parts["test"]["t0_epoch"].min()):
            raise ValueError(f"轨迹 {source!r}: validation 与 test 区间或 purge gap 重叠")
        checked_sources += 1
    for split in ("validation", "test"):
        available = set(named[split].attrs.get("available_columns", named[split].columns))
        if require_real_weather:
            miss = set(REAL_WEATHER_ERROR_COLUMNS) - available
            if miss:
                raise ValueError(f"{split} 真实联合样本缺少天气误差列 {sorted(miss)}")
        if require_real_recovery_state:
            if not any(c in available for c in ACTUAL_RECOVERY_STATE_COLUMNS):
                raise ValueError(f"{split} 真实联合样本缺少 actual_recovery_state 真实回收状态列")
    return {
        "train_rows": len(named["train"]),
        "validation_rows": len(named["validation"]),
        "test_rows": len(named["test"]),
        "purge_min": float(purge_min),
        "sources_checked": int(checked_sources),
        "disjoint": True,
        "purge_verified": True,
        "independence_verified": False,
        "independent": False,  # legacy field: never overclaim stochastic independence
    }


def validate_holdout_independence(train_df: pd.DataFrame, validation_df: pd.DataFrame,
                                  test_df: pd.DataFrame, purge_min=0.0,
                                  require_real_weather=False,
                                  require_real_recovery_state=False):
    """Deprecated compatibility alias; stochastic independence is never certified."""
    return validate_holdout_disjointness(
        train_df, validation_df, test_df, purge_min=purge_min,
        require_real_weather=require_real_weather,
        require_real_recovery_state=require_real_recovery_state)


def required_cells_present(df: pd.DataFrame, cells):
    have = set(zip(df["h_min"].astype(int), df["c_state"].astype(str)))
    need = {(int(h), str(c)) for h, c in cells}
    return sorted(need - have)


def ambiguity_from_samples(df: pd.DataFrame, horizons, states,
                           min_cell_n: int = 30,
                           merge_policy: str = "low_speed_pair",
                           *, formal: bool = False) -> M.XiAmbiguity:
    """由 train 样本按阶段3数据契约估计矩模糊集。

    Formal provenance is propagated from already validated sample metadata.  This
    is intentionally different from assigning a hard-coded predictor: if the
    source frame did not pass the formal sample contract, the rebuilt ambiguity
    cannot acquire a physical certificate merely by reconstruction.
    """
    d = df.copy()
    d.attrs.update(df.attrs)
    if "mmsi" not in d.columns:
        d["mmsi"] = "UNKNOWN"

    meta = dict(d.attrs.get("formal_xi_sample_contract", {}))
    if formal and not meta:
        # Support callers that constructed a DataFrame directly, but validate it
        # with the same fail-closed contract before propagating provenance.
        meta = _validate_formal_xi_sample_contract(d, "train")
    overlap_policy = str(meta.get("sample_overlap_policy",
                                  _single_text_value(d, "sample_overlap_policy", "legacy-unspecified")))
    purge_min = meta.get("purge_min", None)
    concrete_mmsi = sorted({
        str(x).strip() for x in d["mmsi"].dropna().astype(str)
        if str(x).strip() not in {"", "ALL", "UNKNOWN", "nan", "None"}
    })
    if formal:
        if len(concrete_mmsi) != 1:
            raise ValueError(
                "formal ξ train 样本必须精确绑定一个具体 MMSI；"
                f"当前样本包含 concrete_mmsi={concrete_mmsi!r}。禁止 mmsi=ALL 跨船统计。")
        target_mmsi = concrete_mmsi[0]
        requested = str(d.attrs.get("selected_mmsi_filter", target_mmsi)).strip()
        if requested.upper() == "ALL" or requested != target_mmsi:
            raise ValueError(
                "formal ξ 样本 provenance 未绑定具体 MMSI；"
                f"requested={requested!r}, data={target_mmsi!r}。")
    else:
        target_mmsi = (concrete_mmsi[0] if len(concrete_mmsi) == 1 else "ALL")

    accepted, _rejected = S7.summarize_with_contract(
        d, int(min_cell_n), str(merge_policy), "train",
        overlap_policy=overlap_policy, purge_min=purge_min)
    cells = {}
    wanted_h = {int(h) for h in horizons}; wanted_s = {str(c) for c in states}
    for row in accepted:
        if str(row.get("mmsi")) != str(target_mmsi):
            continue
        h, c = int(row["h_min"]), str(row["c_state"])
        if h not in wanted_h or c not in wanted_s:
            continue
        mu = np.array([float(row["mu_e_m"]), float(row["mu_n_m"])])
        cov = np.array([[float(row["sigma_ee"]), float(row["sigma_en"])],
                        [float(row["sigma_en"]), float(row["sigma_nn"])]])
        cells[(h, c)] = M.XiCell(
            h_min=h, c_state=c, n=int(row["n"]), mu=mu, Sigma=cov,
            support_radius=float(row["max_norm_m"]),
            p95_norm=float(row["p95_norm_m"]),
            rms_norm=float(row["rms_norm_m"]))
    if not cells:
        raise ValueError("train 样本没有任何统计格通过阶段3最小样本契约")

    obj = M.XiAmbiguity(cells, sorted({h for h, _ in cells}))
    obj.predictor = str(meta.get("predictor", _single_text_value(d, "predictor")))
    obj.predictor_contract = str(meta.get(
        "predictor_contract", _single_text_value(d, "predictor_contract")))
    obj.timestamp_epoch_contract = str(meta.get(
        "timestamp_epoch_contract", _single_text_value(d, "timestamp_epoch_contract")))
    obj.sample_overlap_policy = str(meta.get("sample_overlap_policy", overlap_policy))
    obj.moments_source = str(meta.get("moments_source",
                                      _single_text_value(d, "moments_source", "unknown")))
    obj.valid_for_formal_data = bool(meta.get("valid_for_formal_data", False))
    obj.formal_validated = bool(formal and obj.valid_for_formal_data)
    obj.selected_mmsi = str(target_mmsi)
    obj.cross_vessel_pooling = bool(str(target_mmsi).upper() == "ALL")
    obj.sample_source_path = str(d.attrs.get("source_path", ""))
    obj.formal_horizon_grid_contract = (M.XI_FORMAL_GRID_CONTRACT
                                        if formal else "legacy-nonformal")
    obj.covariance_contract = (M.XI_COVARIANCE_CONTRACT
                               if formal else "legacy-nonformal")
    return obj

def samples_df_for_cell(df: pd.DataFrame, h: int, c: str) -> pd.DataFrame:
    return df[(df["h_min"] == int(h)) & (df["c_state"].astype(str) == str(c))].copy()


def samples_for_cell(df: pd.DataFrame, h: int, c: str) -> np.ndarray:
    sub = samples_df_for_cell(df, h, c)
    return sub[["xi_e_m", "xi_n_m"]].to_numpy(float)


# =============================================================================
# 2. 回放: 在 test 样本上统计某方案的实测违反率
# =============================================================================
def _weather_deltas(n, weather_unc, dist="t3", seed=0):
    """更新: 与 ξ 同尾形(t3 标准化)的天气扰动样本 —— 返回 (Δw[n,2] 10m 风矢量增量 m/s,
    ΔHs[n] 有效浪高增量 m)。二阶矩取 weather_unc(该 h 的 WeatherUnc: wind_cov/hs_std),
    偏置 wind_bias/hs_bias 一并加入 —— 与规划侧风/浪 DRCC 的同一不确定性模型对拍。"""
    rng = np.random.default_rng(seed)
    cov = np.asarray(weather_unc.wind_cov, float)
    cov = 0.5 * (cov + cov.T)
    try:
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(2))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov), 0.0)))
    z = rng.standard_normal((n, 2))
    hz = rng.standard_normal(n)
    if dist in ("t5", "t3"):
        dfree = 5.0 if dist == "t5" else 3.0
        g = rng.chisquare(dfree, size=n) / dfree
        s = np.sqrt((dfree - 2.0) / dfree) / np.sqrt(g)
        z = z * s[:, None]
        g2 = rng.chisquare(dfree, size=n) / dfree
        hz = hz * np.sqrt((dfree - 2.0) / dfree) / np.sqrt(g2)
    dw = z @ L.T + np.asarray(weather_unc.wind_bias, float)[None, :]
    dhs = hz * float(weather_unc.hs_std) + float(getattr(weather_unc, "hs_bias", 0.0))
    return dw, dhs


def _actual_recovery_state_column(df: pd.DataFrame):
    for c in ACTUAL_RECOVERY_STATE_COLUMNS:
        if c in df.columns:
            return c
    return None


def _actual_recovery_state_samples(cell_df: pd.DataFrame, planned_state: str,
                                   mode: str, real_sample_context: bool):
    """Resolve per-sample realized recovery states without using prediction errors as labels.

    ``real`` requires an explicit categorical state column. ``planned`` is allowed only for
    mechanism/synthetic replay and cannot support a formal real-world claim. ``auto`` chooses
    real data when present and otherwise fails closed in a real-sample context.
    """
    requested = str(mode).strip().lower()
    col = _actual_recovery_state_column(cell_df)
    if requested == "auto":
        requested = "real" if col is not None else ("unavailable" if real_sample_context else "planned")
    if requested == "real":
        if col is None:
            return None, "real-missing", False, False
        vals = cell_df[col].astype("string").str.strip()
        bad = vals.isna() | vals.eq("") | vals.str.lower().isin({"nan", "none", "unknown"})
        if bool(bad.any()):
            return None, "real-invalid", False, False
        return vals.astype(str).to_numpy(), f"real:{col}", True, True
    if requested in ("planned", "prediction", "model"):
        vals = np.repeat(str(planned_state), len(cell_df))
        return vals, "planned-prediction", True, False
    if requested in ("none", "unavailable"):
        return None, "unavailable", False, False
    raise ValueError("recovery_state_sample_mode 必须为 auto|real|planned|unavailable")


def _route_recovery_state(route, fallback_ship, h: float):
    source_ship = getattr(route, "ship", None) or fallback_ship
    try:
        return source_ship.recovery_state_at(float(h))
    except Exception as exc:
        return None, f"missing:{type(exc).__name__}"


def _zero_xi_for_terminal(h: int, state: str) -> M.XiAmbiguity:
    cell = M.XiCell(h_min=int(h), c_state=str(state), n=10**6,
                    mu=np.zeros(2), Sigma=np.zeros((2, 2)), support_radius=0.0,
                    p95_norm=0.0, rms_norm=0.0)
    return M.XiAmbiguity({(int(h), str(state)): cell}, [int(h)])


def _planned_terminal_diag(route, h: int, p: M.Params, wx: dict, weather_unc, xi_state: str):
    """Recover the planned robust dock reserve under the discrete recovery-target model."""
    try:
        d = RM.route_feasible_at_h(route, int(h), p, wx,
                                   _zero_xi_for_terminal(int(h), str(xi_state)),
                                   weather_unc=weather_unc, chance_mode="drcc")
        return dict(E_dock_Wh=float(d.get("E_dock_Wh", np.nan)),
                    t_dock_s=float(d.get("t_dock_s", np.nan)),
                    available=bool(np.isfinite(d.get("E_dock_Wh", np.nan))
                                   and np.isfinite(d.get("t_dock_s", np.nan))))
    except Exception as exc:
        return dict(E_dock_Wh=np.nan, t_dock_s=np.nan, available=False,
                    error=f"{type(exc).__name__}:{exc}")


def _gate_components(Hs: float, Tp: float, wave_dir: float, ship_heading: float,
                     w10: float, p: M.Params):
    motion = M.deck_motion(Hs, Tp, wave_dir - ship_heading, p)
    wave_fail = bool(motion["heave"] > p.s_heave_max
                     or motion["roll"] > p.s_roll_max
                     or motion["pitch"] > p.s_pitch_max
                     or Hs > p.Hs_op)
    wind_fail = bool(w10 > p.w_land_max)
    return wave_fail, wind_fail, motion



def _minimum_return_time_s(displacement: np.ndarray, wind_vec: np.ndarray,
                           v_air_max: float) -> float:
    """Exact minimum straight-line return time under a constant wind and airspeed cap."""
    r = np.asarray(displacement, float); w = np.asarray(wind_vec, float)
    a = float(r @ r)
    if a == 0.0:
        return 0.0
    b = float(r @ w)
    c = float(w @ w) - float(v_air_max) ** 2
    disc = b * b - a * c
    if disc < 0.0:
        return float("inf")
    x_hi = (b + math.sqrt(max(disc, 0.0))) / a
    return (1.0 / x_hi) if x_hi > 0.0 else float("inf")


def _realized_return_energy_policy(displacement: np.ndarray, wind_vec: np.ndarray,
                                    available_s: float, p: M.Params,
                                    escort_power_W: float) -> dict:
    """Choose the lowest-energy feasible split between return flight and stern waiting."""
    T_avail = float(available_s)
    min_time = _minimum_return_time_s(displacement, wind_vec, float(p.v_air_max))
    if not math.isfinite(min_time) or min_time > T_avail or T_avail <= 0.0:
        if T_avail > 0.0:
            needed = float(np.linalg.norm(np.asarray(displacement, float) / T_avail
                                          - np.asarray(wind_vec, float)))
            energy = M.leg_power(p, float(p.v_air_max)) * T_avail / 3600.0
        else:
            needed = float("inf"); energy = 0.0
        return dict(feasible=False, minimum_return_time_s=float(min_time),
                    return_time_s=float(min_time), wait_s=0.0,
                    required_airspeed_ms=float(needed), energy_Wh=float(energy))
    lo = max(float(min_time), 1e-6); hi = max(lo, T_avail)

    def _eval(T):
        vg = np.asarray(displacement, float) / max(float(T), 1e-12)
        req = float(np.linalg.norm(vg - np.asarray(wind_vec, float)))
        if req > float(p.v_air_max):
            return float("inf"), req
        E = (M.leg_power(p, req) * float(T)
             + float(escort_power_W) * max(T_avail - float(T), 0.0)) / 3600.0
        return float(E), req

    if hi <= lo + 1e-9:
        E, req = _eval(hi)
        return dict(feasible=math.isfinite(E), minimum_return_time_s=float(min_time),
                    return_time_s=float(hi), wait_s=0.0,
                    required_airspeed_ms=float(req), energy_Wh=float(E))
    grid = np.linspace(lo, hi, 513)
    vals = []
    for T in grid:
        E, req = _eval(float(T)); vals.append((E, float(T), req))
    E0, T0, req0 = min(vals, key=lambda z: z[0])
    # Refine the best grid interval without assuming convexity outside that local bracket.
    try:
        from scipy.optimize import minimize_scalar
        j = int(np.argmin([z[0] for z in vals]))
        a = float(grid[max(0, j - 1)]); bnd = float(grid[min(len(grid) - 1, j + 1)])
        if bnd > a + 1e-9:
            opt = minimize_scalar(lambda T: _eval(float(T))[0], bounds=(a, bnd),
                                  method="bounded", options={"xatol": 1e-7})
            if bool(getattr(opt, "success", False)) and math.isfinite(float(opt.fun)):
                E0 = float(opt.fun); T0 = float(opt.x); req0 = _eval(T0)[1]
    except Exception:
        pass
    return dict(feasible=math.isfinite(E0), minimum_return_time_s=float(min_time),
                return_time_s=float(T0), wait_s=float(max(T_avail - T0, 0.0)),
                required_airspeed_ms=float(req0), energy_Wh=float(E0))


def _realized_speed_recourse(route, h: float, p: M.Params, wx: dict,
                              xi: np.ndarray, wind_delta, t_dock_s: float,
                              E_dock_Wh: float) -> dict:
    """Replay the fixed-touchdown wait-and-return-speed policy for one realised sample."""
    h_s = 60.0 * float(h)
    wd = None if wind_delta is None else np.asarray(wind_delta, float)
    nom = RM.route_nominal_ET(route, h, p, wx, wind_delta=wd, t_dock_s=float(t_dock_s))
    d0 = float(nom["d_ret0"])
    T_ret_nom = d0 / max(float(nom["v_ret"]), 1e-12)
    T_nonreturn = float(nom["T0"]) - T_ret_nom
    T_budget = h_s - float(t_dock_s) - T_nonreturn
    q_last = np.asarray(route.turbines[-1].local, float)
    P_real = np.asarray(route.ship.predicted_at(float(h)), float) + np.asarray(xi, float)
    r = P_real - q_last
    w_nom, alpha = RM._return_leg_wind_at_height(route, p, wx)
    w_real = np.asarray(w_nom, float) + (alpha * wd if wd is not None else 0.0)
    escort = RM.escort_state(route, h, p, wx, wind_delta=wd)
    policy = _realized_return_energy_policy(
        r, w_real, T_budget, p, float(escort["power_W"]))
    time_violation = not bool(policy["feasible"])
    E_total = (float(nom.get("E_fixed_nonreturn_Wh", 0.0))
               + float(policy["energy_Wh"]) + float(E_dock_Wh))
    min_time = float(policy["minimum_return_time_s"])
    overrun = (max(min_time - T_budget, 0.0)
               if math.isfinite(min_time) else float("inf"))
    realized_core = (T_nonreturn + float(policy.get("return_time_s", min_time))
                     + float(t_dock_s))
    return dict(E=E_total, E_return_wait_Wh=float(policy["energy_Wh"]),
                T_budget_s=float(T_budget), minimum_return_time_s=min_time,
                realized_return_time_s=float(policy.get("return_time_s", min_time)),
                required_airspeed_ms=float(policy["required_airspeed_ms"]),
                time_violation=time_violation,
                realized_core_time_s=float(realized_core),
                realized_wait_s=float(policy.get("wait_s", 0.0)),
                scheduled_touchdown_s=float(h_s),
                time_overrun_s=float(overrun),
                speed_feasible=bool(nom.get("speed_ok_outbound", True) and not time_violation),
                escort_speed_feasible=bool(escort.get("feasible", True)),
                time_contract=RM.SPEED_RECOURSE_TIME_CONTRACT,
                return_speed_recourse_contract=RM.SPEED_RECOURSE_CONTRACT,
                energy_recourse_contract="minimum_energy_return_plus_escort",
                speed_is_recourse=True)

def replay_routes(chosen_routes_with_h, ship, p, wx, test_df,
                  weather_unc=None, weather_dist="t3", weather_seed=0,
                  include_dock=True, recheck_gate=True,
                  weather_sample_mode="synthetic",
                  recovery_state_sample_mode="auto",
                  require_complete=False,
                  holdout_disjointness_verified=False,
                  confirmatory=False,
                  holdout_independence_verified=None) -> dict:
    r"""Replay the exact per-sortie union event declared by the model.

    Each route uses its own launch-time observable state ``c(tau)`` for the exact ``(h,c)``
    outer-error holdout cell.  The discrete recovery horizon h is a route decision and determines
    the planned recovery target ``ship.predicted_at(h)``.  The same cell's AIS/CV position error
    realizes the ship position at recovery; there is no additional acquisition random variable.
    The separately predicted recovery state is used only to allow or forbid stern escort/landing.

    The full weather-on union is energy, time, wave gate, wind gate, route airspeed,
    dock-reserve exceedance and stern escort.  The realized recovery-state
    gate is audited as a dedicated diagnostic and any forbidden state is included in the union
    through stern-escort failure as well.  ``viol_rate_any`` and its
    confidence bounds are returned only when all required event channels and all selected-route
    cells are observed; otherwise ``viol_rate_any_observed`` remains available but
    ``validation_complete=False`` prevents a 95% reliability claim.
    """
    if holdout_independence_verified is not None:
        # Compatibility with v16 callers: the old flag actually established
        # disjointness/purge, not stochastic independence.
        holdout_disjointness_verified = bool(
            holdout_disjointness_verified or holdout_independence_verified)
    chosen_routes_with_h = list(chosen_routes_with_h)
    n_selected_sorties = len(chosen_routes_with_h)
    simultaneous_confidence = 1.0 - 0.05 / max(1, n_selected_sorties)
    weather_sample_mode = str(weather_sample_mode).strip().lower()
    if weather_sample_mode not in ("synthetic", "real"):
        raise ValueError("weather_sample_mode 必须为 synthetic 或 real")
    weather_on = bool(weather_unc is not None or weather_sample_mode == "real")
    if weather_sample_mode == "real":
        miss = set(REAL_WEATHER_ERROR_COLUMNS) - set(test_df.columns)
        if miss:
            raise ValueError(f"真实天气回放缺少列 {sorted(miss)}")
        pmiss = set(REAL_WEATHER_PROVENANCE_COLUMNS) - set(test_df.columns)
        if pmiss:
            raise ValueError(f"真实天气回放缺少 provenance 列 {sorted(pmiss)}")
        if weather_unc is None or not isinstance(weather_unc, RM.WeatherAmbiguity):
            raise ValueError("真实天气回放必须绑定 formal WeatherAmbiguity")
        wsrc = set(test_df["weather_source_sha256"].astype(str))
        wtrain = set(test_df["weather_train_source_sha256"].astype(str))
        if wsrc != {str(getattr(weather_unc, "weather_source_sha256", ""))}:
            raise ValueError("真实天气 holdout 与规划 WeatherAmbiguity 的 weather source SHA 不一致")
        if wtrain != {str(getattr(weather_unc, "xi_train_source_sha256", ""))}:
            raise ValueError("真实天气 holdout 与规划 WeatherAmbiguity 的 train source SHA 不一致")
        if set(test_df["weather_predictor_contract"].astype(str)) != {str(getattr(weather_unc, "predictor_contract", ""))}:
            raise ValueError("真实天气 holdout 与规划 WeatherAmbiguity predictor contract 不一致")

    allocation = RM.mission_risk_allocation(p, weather_on)
    required_events = set(allocation)
    event_keys = ("energy", "time", "wave_gate", "wind_gate", "landing_gate",
                  "route_airspeed", "dock_reserve", "stern_escort",
                  "recovery_state_gate")
    counts = {k: 0 for k in event_keys}
    n_total = 0; n_any_observed = 0
    per_route, missing_reasons = [], []
    realized_time_records = []
    heading = float(wx.get("ship_heading", 0.0) or 0.0)
    Hs0 = float(wx.get("Hs", 0.5) if wx.get("Hs") is not None else 0.5)
    w10_0 = wx.get("wind10", 6.7)
    w10_0 = 6.7 if (w10_0 is None or (isinstance(w10_0, float) and math.isnan(w10_0))) else float(w10_0)
    wdir_0 = wx.get("wind_dir_from", 230.0)
    wdir_0 = 230.0 if (wdir_0 is None or (isinstance(wdir_0, float) and math.isnan(wdir_0))) else float(wdir_0)

    for ri, item in enumerate(chosen_routes_with_h):
        if len(item) < 2:
            raise ValueError("chosen_routes_with_h 每项至少为 (Route,h)")
        r = item[0]
        h_raw = float(item[1])
        if not math.isfinite(h_raw) or h_raw != float(int(h_raw)):
            raise ValueError(f"formal replay horizon must be an exact integer-minute cell, got {item[1]!r}")
        h = int(h_raw)
        try:
            route_wx_launch = (r.ship.weather_at_h(0.0, fallback=wx)
                               if hasattr(r.ship, "weather_at_h") else dict(wx))
            route_wx_recovery = (r.ship.weather_at_h(float(h), fallback=route_wx_launch)
                                 if hasattr(r.ship, "weather_at_h") else dict(route_wx_launch))
        except Exception as exc:
            missing_reasons.append(f"route[{ri}]:missing_recovery_weather:{type(exc).__name__}")
            per_route.append(dict(route_index=ri, stops=r.n_stops(), h=h, n_test=0,
                                  viol_rate=None, validation_complete=False,
                                  reason="missing_recovery_weather"))
            continue
        route_heading = float(route_wx_recovery.get("ship_heading", heading) or heading)
        route_Hs0 = float(route_wx_recovery.get("Hs", Hs0)
                          if route_wx_recovery.get("Hs") is not None else Hs0)
        route_w10_0 = route_wx_recovery.get("wind10", w10_0)
        route_w10_0 = (w10_0 if route_w10_0 is None or
                       (isinstance(route_w10_0, float) and math.isnan(route_w10_0))
                       else float(route_w10_0))
        route_wdir_0 = float(route_wx_recovery.get("wind_dir_from", wdir_0) or wdir_0)
        state, state_source = _route_recovery_state(r, ship, h)
        if state is None:
            missing_reasons.append(f"route[{ri}]:missing_recovery_state")
            per_route.append(dict(route_index=ri, stops=r.n_stops(), h=h, recovery_state=None,
                                  n_test=0, viol_rate=None,
                                  validation_complete=False,
                                  reason="missing_recovery_state_support"))
            continue
        if str(state) in set(getattr(p, "recovery_forbidden_states", ())):
            missing_reasons.append(f"route[{ri}]:forbidden_recovery_state:{state}")
            per_route.append(dict(route_index=ri, stops=r.n_stops(), h=h,
                                  recovery_state=str(state), recovery_state_source=state_source,
                                  n_test=0, viol_rate=None, validation_complete=False,
                                  reason="recovery_state_forbidden"))
            continue
        xi_state = str(getattr(r.ship, "c_state", getattr(ship, "c_state", "unknown")))
        cell_df = samples_df_for_cell(test_df, h, xi_state)
        xs = cell_df[["xi_e_m", "xi_n_m"]].to_numpy(float)
        if len(xs) == 0:
            missing_reasons.append(f"route[{ri}]:missing_cell:({h},{xi_state})")
            per_route.append(dict(route_index=ri, stops=r.n_stops(), h=h,
                                  xi_state=xi_state, recovery_state=str(state),
                                  recovery_state_source=state_source,
                                  n_test=0, viol_rate=None, validation_complete=False,
                                  reason="missing_holdout_cell"))
            continue

        route_weather_unc = RM._resolve_weather_unc(weather_unc, h)
        if weather_sample_mode == "real":
            dw = cell_df[["wind_error_e_ms", "wind_error_n_ms"]].to_numpy(float)
            dhs = cell_df["hs_error_m"].to_numpy(float)
        elif route_weather_unc is not None:
            dw, dhs = _weather_deltas(len(xs), route_weather_unc, dist=weather_dist,
                                      seed=weather_seed + 101 * ri)
        else:
            dw = np.zeros((len(xs), 2)); dhs = np.zeros(len(xs))
        actual_states, actual_state_mode, actual_state_complete, actual_state_real = (
            _actual_recovery_state_samples(
                cell_df, str(state), recovery_state_sample_mode,
                real_sample_context=(weather_sample_mode == "real")))
        route_observed = {"energy", "time", "route_airspeed"}
        if actual_state_complete:
            route_observed.add("stern_escort")
            route_observed.add("recovery_state_gate")
        else:
            missing_reasons.append(f"route[{ri}]:actual_recovery_state_missing")
        if weather_on and recheck_gate:
            route_observed.update(("wave_gate", "wind_gate", "landing_gate"))
        elif weather_on:
            missing_reasons.append(f"route[{ri}]:gate_recheck_disabled")
        planned_terminal = _planned_terminal_diag(r, h, p, route_wx_launch, route_weather_unc, xi_state)
        if weather_on and include_dock and planned_terminal.get("available", False):
            route_observed.add("dock_reserve")
        elif weather_on:
            missing_reasons.append(f"route[{ri}]:planned_dock_reserve_unavailable")

        P_rec_pred = r.ship.predicted_at(float(h))
        h_s = 60.0 * float(h)
        route_core_times, route_waits, route_overruns = [], [], []
        local = {k: 0 for k in event_keys}; local_any = 0
        failure_mask = []
        cell_sha256 = _holdout_cell_fingerprint(cell_df)
        for i, xi in enumerate(xs):
            _dw = dw[i] if weather_on else None
            P_rec_real = P_rec_pred + np.asarray(xi, float)
            gwx = RM.recovery_gate_wx(r, route_wx_recovery, P_rec_real)
            gTp = float(gwx.get("Tp", 2.1)); gwave = float(gwx.get("wave_dir", 200.0))
            gHs0 = float(gwx.get("Hs", route_Hs0) if gwx.get("Hs") is not None else route_Hs0)
            gw10_nom = gwx.get("wind10", route_w10_0)
            gw10_nom = route_w10_0 if (gw10_nom is None or (isinstance(gw10_nom, float) and math.isnan(gw10_nom))) else float(gw10_nom)
            gwdir = float(gwx.get("wind_dir_from", route_wdir_0) or route_wdir_0)
            if weather_on:
                gv = RM._wind_vec(gw10_nom, gwdir) + dw[i]
                gw10_s, _ = RM._speed_dir_from_vec(gv)
                gHs_s = max(gHs0 + float(dhs[i]), 0.0)
            else:
                gw10_s, gHs_s = gw10_nom, gHs0
            waveV, windV, motion = _gate_components(gHs_s, gTp, gwave, route_heading, gw10_s, p)
            if include_dock:
                t_d, E_d = M.dock_reserve(p, motion, gw10_s)
            else:
                t_d, E_d = 0.0, 0.0
            if getattr(p, "speed_adjustable", False):
                det = _realized_speed_recourse(r, h, p, route_wx_launch, xi, _dw, t_d, E_d)
                time_realized = dict(
                    realized_core_time_s=float(det["realized_core_time_s"]),
                    realized_wait_s=float(det["realized_wait_s"]),
                    scheduled_touchdown_s=float(det["scheduled_touchdown_s"]),
                    time_overrun_s=float(det["time_overrun_s"]),
                    time_violation=bool(det["time_violation"]))
            else:
                det = RM.route_energy_time(r, h, xi, p, route_wx_launch, detail=True,
                                           wind_delta=_dw, t_dock_s=t_d)
                realized_core_time_s = float(det["T"]) + float(t_d)
                time_realized = RM.realized_fixed_touchdown_time(h_s, realized_core_time_s)
            actual_state = (str(actual_states[i]) if actual_state_complete else None)
            actual_state_forbidden = bool(
                actual_state_complete and actual_state in set(getattr(p, "recovery_forbidden_states", ())))
            route_core_times.append(float(time_realized["realized_core_time_s"]))
            route_waits.append(float(time_realized["realized_wait_s"]))
            route_overruns.append(float(time_realized["time_overrun_s"]))
            realized_time_records.append(dict(
                route_index=int(ri), sample_index=int(i), h_min=float(h),
                realized_core_time_s=float(time_realized["realized_core_time_s"]),
                realized_wait_s=float(time_realized["realized_wait_s"]),
                scheduled_touchdown_s=float(time_realized["scheduled_touchdown_s"]),
                time_overrun_s=float(time_realized["time_overrun_s"]),
                time_violation=bool(time_realized["time_violation"]),
                time_contract=(RM.SPEED_RECOURSE_TIME_CONTRACT
                               if getattr(p, "speed_adjustable", False) else RM.WAIT_ONLY_TIME_CONTRACT),
                wait_is_recourse=True, speed_is_recourse=bool(getattr(p, "speed_adjustable", False)),
                realized_required_airspeed_ms=det.get("required_airspeed_ms")))
            flags = {
                "energy": bool((det["E"] if getattr(p, "speed_adjustable", False)
                                 else det["E"] + E_d) > p.B_use),
                "time": bool(time_realized["time_violation"]),
                "route_airspeed": not bool(det["speed_feasible"]),
                "stern_escort": bool(actual_state_forbidden
                                      or not bool(det.get("escort_speed_feasible", True))),
                "recovery_state_gate": bool(actual_state_forbidden),
                "wave_gate": bool(waveV),
                "wind_gate": bool(windV),
                "landing_gate": bool(waveV or windV),
                "dock_reserve": bool(planned_terminal.get("available", False)
                                     and (E_d > float(planned_terminal["E_dock_Wh"])
                                          or t_d > float(planned_terminal["t_dock_s"]))),
            }
            observed_union = any(flags[k] for k in route_observed)
            failure_mask.append(bool(observed_union))
            local_any += int(observed_union)
            for k in event_keys:
                if k in route_observed:
                    local[k] += int(flags[k])
        n = len(xs); n_total += n; n_any_observed += local_any
        for k in event_keys: counts[k] += local[k]
        route_complete = required_events.issubset(route_observed)
        if not route_complete:
            missing_reasons.append(f"route[{ri}]:unobserved_events:{sorted(required_events-route_observed)}")
        # IID/binomial intervals are retained for diagnosis only. The formal
        # gate uses a Hoeffding-Azuma bound for average conditional failure risk,
        # which does not require iid Bernoulli observations or a common p.
        route_ci_lo, route_ci_hi = M.binomial_interval(local_any, n, confidence=0.95)
        iid_route_upper95 = M.binomial_upper_bound(local_any, n, confidence=0.95)
        route_upper95 = M.martingale_conditional_risk_upper_bound(failure_mask, confidence=0.95)
        route_simultaneous_upper95 = M.martingale_conditional_risk_upper_bound(
            failure_mask, confidence=simultaneous_confidence)
        mask_sha256, event_sha256 = _failure_event_fingerprint(
            cell_sha256, failure_mask, required_events)
        per_route.append(dict(
            route_index=ri, stops=r.n_stops(), h=h,
            xi_state=xi_state, recovery_state=str(state),
            recovery_state_source=str(state_source), n_test=n, n_viol_any_observed=int(local_any),
            viol_rate_observed=round(local_any/n, 6),
            observed_ci95_low=(round(float(route_ci_lo), 6) if route_ci_lo is not None else None),
            observed_ci95_high=(round(float(route_ci_hi), 6) if route_ci_hi is not None else None),
            observed_upper95=(round(float(route_upper95), 6) if route_upper95 is not None else None),
            conditional_risk_upper95=(round(float(route_upper95), 6) if route_upper95 is not None else None),
            iid_diagnostic_upper95=(round(float(iid_route_upper95), 6)
                                    if iid_route_upper95 is not None else None),
            viol_rate=(round(local_any/n, 6) if route_complete else None),
            ci95_low=(round(float(route_ci_lo), 6) if route_complete and route_ci_lo is not None else None),
            ci95_high=(round(float(route_ci_hi), 6) if route_complete and route_ci_hi is not None else None),
            upper95=(round(float(route_upper95), 6) if route_complete and route_upper95 is not None else None),
            simultaneous_confidence=float(simultaneous_confidence),
            simultaneous_upper95=(float(route_simultaneous_upper95)
                                  if route_complete and route_simultaneous_upper95 is not None else None),
            recovery_state_sample_mode=actual_state_mode,
            recovery_target_model=str(getattr(p, "recovery_target_model", "discrete_horizon_ship_prediction")),
            terminal_sensor_error_mode=str(getattr(p, "terminal_sensor_error_mode", "out_of_scope")),
            actual_recovery_state_channel_real=bool(actual_state_real),
            n_actual_recovery_state_mismatch=(int(np.sum(np.asarray(actual_states, dtype=str) != str(state)))
                                              if actual_state_complete else None),
            observed_events=sorted(route_observed),
            required_events=sorted(required_events),
            validation_complete=bool(route_complete),
            holdout_cell_sha256=cell_sha256,
            observed_failure_mask_sha256=mask_sha256,
            validation_event_fingerprint=event_sha256,
            validation_event_fingerprint_scope="audit-only-no-bonferroni-relaxation",
            time_contract=(RM.SPEED_RECOURSE_TIME_CONTRACT
                           if getattr(p, "speed_adjustable", False) else RM.WAIT_ONLY_TIME_CONTRACT),
            wait_is_recourse=True, speed_is_recourse=bool(getattr(p, "speed_adjustable", False)),
            realized_core_time_s=(max(route_core_times) if route_core_times else None),
            realized_wait_s=(min(route_waits) if route_waits else None),
            scheduled_touchdown_s=float(h_s),
            time_overrun_s=(max(route_overruns) if route_overruns else None),
            median_realized_core_time_s=(float(np.median(route_core_times)) if route_core_times else None),
            median_realized_wait_s=(float(np.median(route_waits)) if route_waits else None),
            planned_E_dock_Wh=(float(planned_terminal["E_dock_Wh"])
                               if planned_terminal.get("available") else None),
            planned_t_dock_s=(float(planned_terminal["t_dock_s"])
                              if planned_terminal.get("available") else None),
            **{f"n_viol_{k}": int(local[k]) for k in event_keys},
        ))

    validation_complete = bool(not missing_reasons and n_total > 0)
    # Route holdout cells may reuse the same historical samples.  Pooling all route×sample
    # rows as independent Bernoulli trials would therefore make the confidence interval
    # artificially narrow.  The formal requirement is per sortie, so the certificate is the
    # largest one-sided 95% upper bound among selected sorties.  Pooled quantities below are
    # descriptive only.
    rate_observed = (n_any_observed / n_total) if n_total else None
    pooled_ci_lo, pooled_ci_hi = M.binomial_interval(n_any_observed, n_total, confidence=0.95)
    pooled_upper95 = M.binomial_upper_bound(n_any_observed, n_total, confidence=0.95)
    observed_route_uppers = [d.get("observed_upper95") for d in per_route
                             if d.get("observed_upper95") is not None]
    complete_route_uppers = [d.get("upper95") for d in per_route
                             if d.get("validation_complete") and d.get("upper95") is not None]
    simultaneous_route_uppers = [d.get("simultaneous_upper95") for d in per_route
                                 if d.get("validation_complete")
                                 and d.get("simultaneous_upper95") is not None]
    observed_upper95 = max(observed_route_uppers) if observed_route_uppers else None
    ordinary_max_upper95 = (max(complete_route_uppers)
                            if validation_complete and len(complete_route_uppers) == len(per_route)
                            else None)
    upper95 = (max(simultaneous_route_uppers)
               if validation_complete and len(simultaneous_route_uppers) == len(per_route) else None)
    route_rates = [d.get("viol_rate") for d in per_route if d.get("viol_rate") is not None]
    full_rate = (max(route_rates) if validation_complete and len(route_rates) == len(per_route)
                 else None)
    budget = RM.mission_eps_budget(p, weather_on)
    result = dict(
        n_test_total=int(n_total), n_viol_any_observed=int(n_any_observed),
        viol_rate_any_observed=(round(rate_observed, 6) if rate_observed is not None else None),
        observed_ci95_low=None,
        observed_ci95_high=None,
        observed_upper95=(round(float(observed_upper95), 6)
                          if observed_upper95 is not None else None),
        pooled_naive_ci95_low=(round(float(pooled_ci_lo), 6) if pooled_ci_lo is not None else None),
        pooled_naive_ci95_high=(round(float(pooled_ci_hi), 6) if pooled_ci_hi is not None else None),
        pooled_naive_upper95=(round(float(pooled_upper95), 6)
                              if pooled_upper95 is not None else None),
        ordinary_max_per_sortie_upper95=(round(float(ordinary_max_upper95), 6)
                                           if ordinary_max_upper95 is not None else None),
        simultaneous_confidence_per_sortie=float(simultaneous_confidence),
        ci_method="bonferroni-simultaneous-max-per-sortie-hoeffding-azuma-conditional-risk-upper95",
        validation_complete=validation_complete,
        missing_reasons=sorted(set(missing_reasons)),
        required_events=sorted(required_events), eps_allocation=allocation,
        viol_rate_any=(round(full_rate, 6) if full_rate is not None else None),
        n_viol_any=(int(n_any_observed) if validation_complete else None),
        ci95_low=None,
        ci95_high=None,
        upper95=(float(upper95) if upper95 is not None else None),
        holds_point=(bool(full_rate <= budget) if full_rate is not None else None),
        holds_upper95=(bool(upper95 <= budget)
                       if upper95 is not None else None),
        include_dock=bool(include_dock), recheck_gate=bool(recheck_gate),
        weather=("real" if weather_sample_mode == "real" else
                 ("synthetic" if weather_unc is not None else "off")),
        weather_sample_mode=weather_sample_mode,
        recovery_state_sample_mode=str(recovery_state_sample_mode),
        recovery_target_model=str(getattr(p, "recovery_target_model", "discrete_horizon_ship_prediction")),
        terminal_sensor_error_mode=str(getattr(p, "terminal_sensor_error_mode", "out_of_scope")),
        holdout_disjointness_verified=bool(holdout_disjointness_verified),
        holdout_independence_verified=False,
        stochastic_independence_assumed=False,
        confirmatory_evaluation=bool(confirmatory),
        iid_binomial_intervals_are_diagnostic_only=True,
        planning_weather_model_present=bool(weather_unc is not None),
        joint_real_channels_complete=bool(
            weather_unc is not None and weather_on and weather_sample_mode == "real" and per_route
            and all(d.get("actual_recovery_state_channel_real", False) for d in per_route)),
        formal_reliability_claim_eligible=bool(
            validation_complete and confirmatory and holdout_disjointness_verified
            and weather_unc is not None and weather_on and weather_sample_mode == "real" and per_route
            and all(d.get("actual_recovery_state_channel_real", False) for d in per_route)),
        evidence_scope=(
            "confirmatory-purged-disjoint-real-joint-holdout-with-terminal-sensor-error-out-of-scope"
            if validation_complete and confirmatory and holdout_disjointness_verified
            and weather_unc is not None and weather_on and weather_sample_mode == "real"
            and per_route and all(d.get("actual_recovery_state_channel_real", False) for d in per_route)
            else ("validation-selection-only-no-formal-inference" if not confirmatory
                  else "mechanism-or-partial-evidence")),
        eps_budget=round(float(budget), 6), protocol="joint-route-replay",
        time_contract=RM.time_contract_for(p), time_contract_id=RM.time_contract_for(p),
        geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
        soc_risk_allocation=str(getattr(p, "soc_risk_allocation", "fixed")),
        wait_is_recourse=RM.WAIT_IS_RECOURSE, dock_risk_contract=RM.DOCK_RISK_CONTRACT,
        realized_time_records=realized_time_records,
        per_route=per_route,
    )
    aliases = {
        "E": "energy", "T": "time", "gate": "landing_gate", "speed": "route_airspeed",
        "escort": "stern_escort", "dock": "dock_reserve",
        "wave_gate": "wave_gate", "wind_gate": "wind_gate",
        "recovery_state_gate": "recovery_state_gate",
    }
    for alias, key in aliases.items():
        if key is None:
            nval = counts["wave_gate"] + counts["wind_gate"]
            # Combined gate count needs sample-level union; conservative sum is only an alias.
            result[f"n_viol_{alias}"] = int(nval)
            result[f"viol_rate_{alias}"] = round(nval/n_total, 6) if n_total else None
        else:
            result[f"n_viol_{alias}"] = int(counts[key])
            result[f"viol_rate_{alias}"] = round(counts[key]/n_total, 6) if n_total else None
    if require_complete and not validation_complete:
        raise ValueError("联合回放不完整: " + "; ".join(result["missing_reasons"][:12]))
    return result

# =============================================================================
# 3. ε 扫描下的无泄漏回放(核心实验)
# =============================================================================
def _params_for_mission_budget(base: M.Params, mission_budget: float,
                               weather_on: bool) -> M.Params:
    """Scale the declared event split so component budgets sum to mission_budget."""
    mission_budget = float(mission_budget)
    if not (0.0 < mission_budget < 1.0):
        raise ValueError("mission_budget 必须位于 (0,1)")
    alloc0 = RM.mission_risk_allocation(base, weather_on)
    total0 = float(sum(alloc0.values()))
    if total0 <= 0:
        raise ValueError("基础风险分配为空")
    scaled = {k: mission_budget * float(v) / total0 for k, v in alloc0.items()}
    field = {"energy": "eps_E", "time": "eps_T",
             "wave_gate": "eps_cap", "wind_gate": "eps_gate",
             "route_airspeed": "eps_air", "dock_reserve": "eps_dock",
             "stern_escort": "eps_escort"}
    import copy
    out = copy.deepcopy(base)
    for fname in field.values():
        setattr(out, fname, 0.0)
    for event, value in scaled.items():
        setattr(out, field[event], float(value))
    out.mission_failure_budget = mission_budget
    # Binary64 scaling can overshoot the requested mission budget by one or a
    # few ulps even when the real-valued proportions sum exactly.  Formal
    # feasibility must never gain risk budget from that rounding.  Move one
    # active component downward until the exact binary64-as-real sum is safe.
    from fractions import Fraction
    active_fields = [field[event] for event in scaled]
    budget_exact = Fraction.from_float(float(mission_budget))
    def exact_total():
        return sum((Fraction.from_float(float(getattr(out, name)))
                    for name in active_fields), Fraction(0))
    if exact_total() > budget_exact:
        target = max(active_fields, key=lambda name: float(getattr(out, name)))
        value = float(getattr(out, target))
        while exact_total() > budget_exact:
            value = math.nextafter(value, -math.inf)
            if not value > 0.0:
                raise ValueError("无法在 binary64 下构造安全的任务风险分配")
            setattr(out, target, value)
    return out


def replay_vs_eps(turbines, ship, wx, train_df, evaluation_df, horizons, states,
                  eps_list=(0.01, 0.03, 0.05, 0.10), max_stops=4,
                  use_exact=False, outdir=None, min_cell_n=30,
                  base_params=None,
                  holdout_disjointness_verified=False,
                  evaluation_role="validation-diagnostic",
                  holdout_independence_verified=None):
    """Scan the *mission-level* failure budget, not independent component epsilons.

    For each budget the default event proportions are normalized to sum exactly to that budget.
    The formal verdict uses the dependency-robust conditional-risk bound, not
    only the point rate. Validation rows remain selection/diagnostic only; only
    ``evaluation_role=frozen-final-test`` is confirmatory.
    """
    if holdout_independence_verified is not None:
        holdout_disjointness_verified = bool(
            holdout_disjointness_verified or holdout_independence_verified)
    xi_train = ambiguity_from_samples(train_df, horizons, states,
                                      min_cell_n=min_cell_n,
                                      merge_policy="low_speed_pair")
    rows = []
    base_params = base_params or M.Params()
    for mission_eps in eps_list:
        p = _params_for_mission_budget(base_params, float(mission_eps), weather_on=False)
        if use_exact:
            import step12_branch_price as BP
            res = BP.solve_route_drcc_exact(turbines, ship, p, wx, xi_train,
                                            max_stops=max_stops, verbose=False)
            chosen = [(r, RM.route_drcc_feasible(r, p, wx, xi_train)["h"])
                      for r in _routes_from_exact(res, turbines, ship)]
        else:
            res = RA.solve_route_drcc(turbines, ship, p, wx, xi_train,
                                      strategy="full", max_stops=max_stops)
            chosen = [(_route_obj(d, turbines, ship), d["h"]) for d in res["route_diag"]]
        rep = replay_routes(
            chosen, ship, p, wx, evaluation_df,
            holdout_disjointness_verified=holdout_disjointness_verified,
            confirmatory=(str(evaluation_role) == "frozen-final-test"))
        if not rep["validation_complete"]:
            holds = "不可判定(联合事件缺失)"
        elif rep["upper95"] is not None and rep["upper95"] <= float(mission_eps):
            holds = "是"
        else:
            holds = "否"
        alloc = RM.mission_risk_allocation(p, False)
        rows.append(dict(
            eps_mission=float(mission_eps), eps_nominal=float(mission_eps),
            eps_E=alloc.get("energy"), eps_T=alloc.get("time"),
            kappa_E=round(RM.kappa(float(alloc["energy"])), 3),
            n_sorties=len(chosen), emp_viol_E=rep["viol_rate_E"],
            emp_viol_T=rep["viol_rate_T"],
            emp_viol_any=rep["viol_rate_any"],
            emp_viol_any_observed=rep["viol_rate_any_observed"],
            ci95_low=rep["ci95_low"], ci95_high=rep["ci95_high"],
            upper95=rep["upper95"], n_test=rep["n_test_total"],
            eps_budget=round(float(rep["eps_budget"]), 6),
            validation_complete=bool(rep["validation_complete"]), holds=holds,
            evaluation_role=str(evaluation_role),
            evidence_scope=rep.get("evidence_scope"),
            formal_reliability_claim_eligible=bool(
                rep.get("formal_reliability_claim_eligible", False)),
            formal_holds=(holds if rep.get("formal_reliability_claim_eligible", False)
                          else None),
            ci_method=rep.get("ci_method"),
            optimizer_algorithm=("historical-route-enumeration-research-baseline"
                                 if use_exact else "historical-route-heuristic-research-baseline"),
            optimizer_bound_scope="route-subproblem-replay-only",
            global_certificate_available=False,
            global_route_space_certificate=False,
            implicit_route_space_certified=False,
            global_gap_available=False))
    df = pd.DataFrame(rows)
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "replay_vs_eps.csv", index=False, encoding="utf-8-sig")
        log.info("写 %s", outdir / "replay_vs_eps.csv")
    return df

def _route_obj(diag, turbines, ship):
    """从 route_diag 的 turbine id 还原 Route 对象。"""
    by_id = {t.tid: t for t in turbines}
    seq = [by_id[tid] for tid in diag["turbines"]]
    r = RM.Route(rid=-1, turbines=seq, ship=ship); r.fixed_h = diag["h"]
    return r


def _routes_from_exact(res, turbines, ship):
    by_id = {t.tid: t for t in turbines}
    out = []
    for d in res["route_diag"]:
        seq = [by_id[tid] for tid in d["turbines"]]
        r = RM.Route(rid=-1, turbines=seq, ship=ship); r.fixed_h = d["h"]
        out.append(r)
    return out


# =============================================================================
# 4. 合成重尾样本(无真实样本时的自检夹具)
# =============================================================================
def replay_with_real_samples(chosen_routes_with_h, ship, p, wx, samples_csv, xi_amb=None):
    r"""【PR-6 真实留出样本回放】用 step7 --dump-samples 生成的【真实 ξ 样本】(而非合成)做 out-of-sample 回放。
    这是把"合成 t-df3 验证"升级为"真实数据验证"的关键: 真实 DP ξ 峰度 91-131, 结构为"紧核+偶发大漂移",
    与任何参数分布都不同。按每条航线的 (h, c(τ)) 匹配真实样本(更新 复盘要点: 不能全局用一个状态)。

    samples_csv: 列 [mmsi,h_min,c_state,t0_epoch,xi_e_m,xi_n_m](step7 --dump-samples 输出)。
    若某 (h,c) 格真实样本不足, 该航线记 n_test=0(不可判定), 不虚报。返回同 replay_routes 的字典。"""
    import pandas as _pd
    try:
        real = _pd.read_csv(samples_csv)
    except Exception as exc:
        return dict(error=f"无法读取真实样本 {samples_csv}: {type(exc).__name__}", viol_rate_any=None, n_test_total=0)
    need_cols = {"h_min", "c_state", "xi_e_m", "xi_n_m"}
    if not need_cols.issubset(real.columns):
        return dict(error=f"真实样本缺列(需 {need_cols})", viol_rate_any=None, n_test_total=0)
    # 直接复用 replay_routes: 它按每条航线的 h 与 ship.c_state 取样本子集
    return replay_routes(chosen_routes_with_h, ship, p, wx, real)


def _synthetic_samples_legacy_note():
    """占位: 保持向后兼容标记。真实回放请用 replay_with_real_samples。"""
    return None


def _dist_samples(horizons, states, dist="t3", n_per=4000, seed=0, xi_amb=None):
    r"""矩-匹配的 out-of-sample 误差样本, 支持多种分布(全部 **同一阶/二阶矩**, 仅尾部形状不同),
    用于【分布鲁棒压力测试】(更新): 检验各方法方案在整个矩模糊集上的稳健性, 而非单一分布。
      dist:
        'gaussian'  轻尾正态(SAA/高斯的"主场");
        't5'/'t3'   多元 t(df=5/3), 对称重尾(t3=旧 _synthetic_samples, 但此处修正为真·协方差匹配);
        'mixture'   尺度混合: (1-p) 紧核 + p 大偏移 —— 贴近真实船位"多数时刻厘米级 + 偶发数百米漂移"
                    (真实 DP ξ: p50≈5m 但 p99 达数百米, 峰度 91-131), 是最贴近现实的重尾情形;
        'twopoint'  矩-最坏二点(Cantelli 紧): 沿随机方向放置, 只有分布无关 DRCC(Cantelli)能挡 —— 理论最坏。
    更新 修复(审计 P0-回放同源): 新增 xi_amb —— 给定时, 每个 (h,c) 格的 (μ, Σ) 直接取
    【本实例模糊集】xi_amb.get_interp(h,c) 的矩(尾形仍按 dist), 回放分布与规划歧义集同一阶/
    二阶矩 ⇒ “违反率 ≤ 任务联合预算 ε_mission”的检验才与 DRCC 声明对得上。缺省 None 保留旧合成量级
    (base5×(h/5)^1.38, μ=(20,-15)m), 仅供 standalone 自检/无 xi_amb 场景。"""
    rng = np.random.default_rng(seed)
    base5 = {"低速": 405.0, "动力定位": 186.0, "直航": 727.0, "转弯": 989.0}
    mu_legacy = np.array([20.0, -15.0])
    recs = []
    t0 = 1.7e9
    for c in states:
        for h in horizons:
            if xi_amb is not None:
                cell = xi_amb.get_interp(int(h), c)
                mu = np.asarray(cell.mu, float)
                Sig = np.asarray(cell.Sigma, float)
                Sig = 0.5 * (Sig + Sig.T)
                try:
                    L = np.linalg.cholesky(Sig + 1e-9 * np.eye(2))
                except np.linalg.LinAlgError:
                    L = np.diag(np.sqrt(np.maximum(np.diag(Sig), 0.0)))
            else:
                mu = mu_legacy
                sl = base5.get(c, 400.0) * (h / 5.0) ** 1.38
                L = np.array([[sl, 0.0], [0.18 * sl, 0.6 * sl]])   # 尺度矩阵; Σ = L Lᵀ
            for k in range(n_per):
                z = rng.standard_normal(2)
                if dist == "gaussian":
                    xi = L @ z + mu
                elif dist in ("t5", "t3"):
                    dfree = 5.0 if dist == "t5" else 3.0
                    g = rng.chisquare(dfree) / dfree
                    # 修正: 标准化使协方差 = L Lᵀ (t 的 cov = scale²·df/(df-2))
                    xi = (L @ z) / math.sqrt(g) * math.sqrt((dfree - 2.0) / dfree) + mu
                elif dist == "mixture":
                    # 尺度混合(真实型): p 概率大偏移放大 F 倍, 其余紧核缩小 f 倍; 保持 cov=L Lᵀ
                    p = 0.05
                    F = 1.0 / math.sqrt(p)                      # 大偏移 ≈ 4.47σ (贴近真实 p99/p50)
                    f2 = (1.0 - p * F * F) / (1.0 - p)
                    fac = F if rng.random() < p else math.sqrt(max(f2, 0.0))
                    xi = (L @ z) * fac + mu
                elif dist == "twopoint":
                    # 矩-最坏二点(Cantelli 紧, ε=0.05): 沿随机方向, 5% 落在 +4.36σ, 95% 落在 −0.229σ
                    eps0 = 0.05
                    kC = math.sqrt((1.0 - eps0) / eps0)        # 4.359
                    d = math.sqrt(max((1.0 - eps0 * kC * kC) / (1.0 - eps0), 0.0))
                    theta = rng.uniform(0, 2 * math.pi)
                    u = np.array([math.cos(theta), math.sin(theta)])
                    r = kC if rng.random() < eps0 else -d
                    xi = (L @ (r * u)) + mu
                else:
                    xi = L @ z + mu
                recs.append(("ALL", h, c, t0 + k * 60.0, xi[0], xi[1]))
    return pd.DataFrame(recs, columns=["mmsi", "h_min", "c_state", "t0_epoch", "xi_e_m", "xi_n_m"])


def _synthetic_samples(horizons, states, n_per=4000, seed=0, scale=1.0):
    """重尾(多元 t, df=3)合成误差样本。

    ``scale`` 仅用于 standalone 机制自检调节场景难度；正式实验必须使用阶段3矩或真实
    留出样本。生成器标准化为协方差与声明的尺度矩阵一致。
    """
    rng = np.random.default_rng(seed)
    base5 = {"低速": 405.0, "动力定位": 186.0, "直航": 727.0, "转弯": 989.0}
    df_t = 3.0
    recs = []
    t0 = 1.7e9
    for c in states:
        for h in horizons:
            sl = float(scale) * base5.get(c, 400.0) * (h / 5.0) ** 1.38
            # 标准化多元 t，使协方差等于 L L^T。
            L = np.array([[sl, 0.0], [0.18 * sl, 0.6 * sl]])
            for k in range(n_per):
                z = rng.standard_normal(2)
                g = rng.chisquare(df_t) / df_t
                xi = ((L @ z) / math.sqrt(g) * math.sqrt((df_t - 2.0) / df_t)
                      + float(scale) * np.array([20.0, -15.0]))
                recs.append(("ALL", h, c, t0 + k * 60.0, xi[0], xi[1]))
    return pd.DataFrame(recs, columns=["mmsi", "h_min", "c_state", "t0_epoch", "xi_e_m", "xi_n_m"])


# =============================================================================
# 5. 主流程 / 自检
# =============================================================================
def _load_turbines_ship(n_turbines, horizons):
    here = Path(__file__).resolve().parent
    turb_csv = M._first_existing([here / "data" / "turbines_Rodsand_II_clean.csv"])
    if turb_csv:
        turbines = M.load_turbines(turb_csv, farm="Rodsand_II")[:n_turbines]
    else:
        turbines = [M.Turbine(f"DEMO_{i}", np.array([11.55 + 0.006 * i, 54.55]), 68.5, 115.0)
                    for i in range(n_turbines)]
    lat0, lon0 = turbines[0].lonlat[1], turbines[0].lonlat[0]
    for t in turbines:
        t.local = M.latlon_to_local_m(t.lonlat[1], t.lonlat[0], lat0, lon0)
    centroid = np.mean([t.local for t in turbines], axis=0)
    ship = RM.ShipPrediction.from_cv(centroid + np.array([-600.0, -400.0]),
                                     v_ship=np.array([1.2, 0.9]), horizons=horizons,
                                     c_state="动力定位")   # 自洽: 动力定位回收
    return turbines, ship


def main():
    ap = argparse.ArgumentParser(description="逐轨迹 purged train/validation/test 联合回放")
    ap.add_argument("--samples", type=Path, default=None,
                    help="优先使用 step7 的 xi_samples_all_caseB.csv；缺省用重尾合成样本")
    ap.add_argument("--n-turbines", type=int, default=14)
    ap.add_argument("--max-stops", type=int, default=4)
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--validation-frac", type=float, default=0.20)
    ap.add_argument("--purge-min", type=float, default=None,
                    help="切分隔离带分钟数；默认最大预测时长")
    ap.add_argument("--min-cell-n", type=int, default=30)
    ap.add_argument("--research-baseline-route-enumeration", action="store_true",
                    dest="research_baseline_route_enumeration",
                    help="使用历史路线枚举研究基线；不产生车队全局Gap或最优证书")
    ap.add_argument("--exact", action="store_true",
                    dest="research_baseline_route_enumeration", help=argparse.SUPPRESS)
    ap.add_argument("--allow-synth", action="store_true",
                    help="仅用于机制测试；缺少真实回放样本时显式允许重尾合成样本")
    args = ap.parse_args()

    horizons = list(range(5, 61, 5))
    states = ["低速", "动力定位", "直航", "转弯"]
    wx = dict(wind10=2.7, wind_dir_from=230.0, Hs=0.16, Tp=2.1,
              wave_dir=200.0, ship_heading=90.0)

    here = Path(__file__).resolve().parent
    sp = args.samples or M._first_existing([here / "tracks" / "xi_samples_all_caseB.csv",
                                            here / "xi_samples_all_caseB.csv",
                                            here / "tracks" / "xi_samples_caseB.csv"])
    if sp and Path(sp).is_file():
        df = load_samples(Path(sp), mmsi="ALL")
        src = f"真实样本 {Path(sp).name}"
    elif args.allow_synth:
        df = _synthetic_samples(horizons, states, scale=0.30)
        df["source_track_id"] = "synthetic-track"
        src = "重尾合成样本(仅验证回放管线)"
    else:
        raise SystemExit(
            "正式回放必须通过 --samples 显式提供本地真实样本；"
            "仅机制测试可显式添加 --allow-synth。")

    declared = set(df["split"].astype(str).str.lower()) if "split" in df.columns else set()
    if {"train", "validation", "test"}.issubset(declared):
        train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
        validation_df = df[df["split"].astype(str).str.lower() == "validation"].copy()
        test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
        purge_min = float(args.purge_min if args.purge_min is not None else max(horizons))
        split_meta = validate_holdout_independence(train_df, validation_df, test_df,
                                                   purge_min=purge_min)
        split_source = "step7-declared-splits"
    else:
        train_df, validation_df, test_df, split_meta = purged_temporal_split(
            df, args.train_frac, args.validation_frac, args.purge_min)
        validate_holdout_independence(train_df, validation_df, test_df,
                                      purge_min=float(split_meta["purge_min"]))
        split_source = "recomputed-per-track-purged"
    turbines, ship = _load_turbines_ship(args.n_turbines, horizons)

    print("\n================ step15_replay.py 联合回放验证 ================")
    print(f"样本来源: {src}")
    print(f"切分: {split_source} | train={len(train_df)} validation={len(validation_df)} "
          f"test={len(test_df)} purge={split_meta.get('purge_min')}min")
    print(f"风机 {len(turbines)} | 路线生成研究基线 "
          f"{'历史枚举入口' if args.research_baseline_route_enumeration else '历史启发式入口'} "
          "| 不提供车队全局Gap")

    base_params = M.apply_uav_profile(M.Params(), "L")
    # 多预算曲线只在 validation 上作诊断；独立 test 只运行冻结的 5% 任务预算一次。
    df_eps = replay_vs_eps(turbines, ship, wx, train_df, validation_df, horizons, states,
                           max_stops=args.max_stops,
                           use_exact=args.research_baseline_route_enumeration,
                           outdir=RESULTS, min_cell_n=args.min_cell_n,
                           base_params=base_params,
                           evaluation_role="validation-budget-diagnostic")
    df_final = replay_vs_eps(turbines, ship, wx, train_df, test_df, horizons, states,
                             eps_list=(0.05,), max_stops=args.max_stops,
                             use_exact=args.research_baseline_route_enumeration, outdir=None,
                             min_cell_n=args.min_cell_n, base_params=base_params,
                             holdout_independence_verified=True,
                             evaluation_role="frozen-final-test")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(RESULTS / "replay_final_test.csv", index=False, encoding="utf-8-sig")
    print("\n--- validation：任务级风险预算诊断扫描 ---")
    print(df_eps.to_string(index=False))
    print("\n--- 独立 test：冻结 eps_mission=0.05 后只审计一次 ---")
    print(df_final.to_string(index=False))
    print("\n判定规则: validation 仅用于选型；冻结 final test 上 Hoeffding-Azuma 条件风险 upper95≤eps_mission 才可形成正式证据。")
    print("真实最终可靠性声明还要求 evidence_scope=confirmatory-purged-disjoint-real-joint-holdout-with-terminal-sensor-error-out-of-scope；"
          "缺真实天气或近端定位误差时只能作为部分证据/机制压力测试。")
    print(f"\n结果写入 {RESULTS}/replay_vs_eps.csv 与 replay_final_test.csv。")
    if not (sp and Path(sp).is_file()):
        print("注: 当前为合成样本，不能形成真实平台可靠性结论。")

if __name__ == "__main__":
    main()
