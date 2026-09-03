#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step8_gen_recovery.py — 从真实 CTV 轨迹生成无泄漏的起飞/回收场景。

正式默认：
  - horizons=5,10,...,60，与 step7 统计支持一致；
  - 起飞时刻按确定性时间网格采样，不再默认随机抽 200 个点；
  - 名义回收点采用仅使用 t≤t_L 的 CV 预测；
  - 输出记录预测器、起飞状态代理、轨迹时间范围、段号和实际误差分量。

随机抽样仍以 --sampling-mode random 显式保留，仅用于旧实验复现。
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from step7_compute_xi import (PREDICTOR_CONTRACT, TIMESTAMP_EPOCH_CONTRACT,
                              to_epoch_seconds_utc)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("recov")

R_EARTH = 6_371_000.0
KN = 0.514444
DEFAULT_HORIZONS_MIN = list(range(5, 61, 5))
PREDICTOR_NAME = "cv_noleak"


def _norm(c):
    return "".join(ch for ch in str(c).strip().lower() if ch.isalnum())


def find_col(cols, cands):
    nm = {_norm(c): c for c in cols}
    for cand in cands:
        if _norm(cand) in nm:
            return nm[_norm(cand)]
    # Substring fallback is allowed only for descriptive aliases.  A one-character
    # candidate such as ``t`` would otherwise match ``latitude`` before the true time column.
    for cand in cands:
        key = _norm(cand)
        if len(key) < 3:
            continue
        for c in cols:
            if key in _norm(c):
                return c
    return None


def cv_recovery_state(v_ship: np.ndarray, launch_state: str) -> str:
    """Same no-leak CV recovery-state semantics as step10.ShipPrediction.from_cv."""
    speed_kn = float(np.linalg.norm(np.asarray(v_ship, float))) / KN
    if speed_kn < 0.3:
        return "动力定位"
    if speed_kn < 1.0:
        return "低速"
    return "直航"


def parse_horizons(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})
    if not vals or any(h <= 0 for h in vals):
        raise ValueError("--horizons 必须是逗号分隔的正整数分钟。")
    return vals


def load_track(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    tc = find_col(cols, ["t", "timestamp", "datetime"])
    la = find_col(cols, ["lat", "latitude"])
    lo = find_col(cols, ["lon", "longitude", "lng"])
    if not (tc and la and lo):
        raise ValueError(f"{path.name} 缺 t/lat/lon。现有:{cols}")
    out = pd.DataFrame()
    out["t"] = pd.to_datetime(df[tc], errors="coerce", utc=True)
    if out["t"].isna().mean() > 0.5:
        out["t"] = pd.to_datetime(df[tc], dayfirst=True, errors="coerce", utc=True)
    out["lat"] = pd.to_numeric(df[la], errors="coerce")
    out["lon"] = pd.to_numeric(df[lo], errors="coerce")
    return out.dropna(subset=["t", "lat", "lon"]).sort_values("t").drop_duplicates("t").reset_index(drop=True)


def to_seconds(t: pd.Series) -> np.ndarray:
    return to_epoch_seconds_utc(t)


def local_xy(lat, lon, lat0, lon0):
    x = R_EARTH * np.radians(lon - lon0) * math.cos(math.radians(lat0))
    y = R_EARTH * np.radians(lat - lat0)
    return x, y


def xy_to_lonlat(x, y, lat0, lon0):
    lat = lat0 + math.degrees(y / R_EARTH)
    lon = lon0 + math.degrees(x / (R_EARTH * math.cos(math.radians(lat0))))
    return lon, lat


def classify_state(sp_t0_ms, turn_rate_deg_min, disp_win_m,
                   low_kn, dp_kn, dp_disp_m, turn_thr):
    sp_kn = sp_t0_ms / KN
    if sp_kn < dp_kn and disp_win_m < dp_disp_m:
        return "动力定位"
    if sp_kn < low_kn:
        return "低速"
    if abs(turn_rate_deg_min) > turn_thr:
        return "转弯"
    return "直航"


def deterministic_indices(n: int, cap: int = 0) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, cap, dtype=int))


def main() -> int:
    ap = argparse.ArgumentParser(description="从真实 CTV 轨迹生成起飞/回收场景点")
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("./tracks/recovery_scenarios.csv"))
    ap.add_argument("--horizons", type=str,
                    default=",".join(map(str, DEFAULT_HORIZONS_MIN)),
                    help="回收时长(分钟)，默认 5,10,...,60")
    ap.add_argument("--sampling-mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--launch-step-min", type=float, default=5.0,
                    help="grid 模式的确定性起飞时间步长")
    ap.add_argument("--max-launches-per-track", type=int, default=0,
                    help="grid 模式可选确定性等距上限；0 表示不截断")
    ap.add_argument("--n-per-track", type=int, default=200,
                    help="仅 random 旧复现模式使用")
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--seg-gap-min", type=float, default=5.0)
    ap.add_argument("--vel-window-s", type=float, default=60.0)
    ap.add_argument("--low-kn", type=float, default=1.0)
    ap.add_argument("--dp-kn", type=float, default=0.3)
    ap.add_argument("--dp-disp-m", type=float, default=30.0)
    ap.add_argument("--turn-deg-min", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42, help="仅 random 模式使用")
    ap.add_argument("--bbox", type=str, default="11.40,11.80,54.50,54.60",
                    help='合并海域框 "lon_min,lon_max,lat_min,lat_max"；空字符串关闭')
    args = ap.parse_args()

    horizons = parse_horizons(args.horizons)
    if args.dt <= 0 or args.launch_step_min <= 0:
        raise SystemExit("--dt 与 --launch-step-min 必须为正。")
    for h in horizons:
        steps = round(float(h) * 60.0 / float(args.dt))
        if abs(steps * float(args.dt) - float(h) * 60.0) > 1e-7:
            raise SystemExit(
                f"--dt={args.dt:g}s 不能精确表示 h={h}min；正式场景要求 h·60/dt 为整数。")
    launch_steps = float(args.launch_step_min) * 60.0 / float(args.dt)
    if abs(round(launch_steps) - launch_steps) > 1e-7:
        raise SystemExit("--launch-step-min 必须与 --dt 对齐，避免起飞网格发生隐式取整。")
    vel_steps = float(args.vel_window_s) / float(args.dt)
    if abs(round(vel_steps) - vel_steps) > 1e-7:
        raise SystemExit("--vel-window-s 必须是 --dt 的整数倍，避免预测器窗口口径漂移。")
    bbox = None
    if args.bbox.strip():
        lo1, lo2, la1, la2 = (float(v) for v in args.bbox.split(","))
        bbox = (lo1, lo2, la1, la2)
        log.info("bbox: lon[%.2f,%.2f] lat[%.2f,%.2f]", lo1, lo2, la1, la2)
    rng = np.random.default_rng(args.seed)
    dt = float(args.dt)
    seg_gap_s = float(args.seg_gap_min) * 60.0
    vw = max(1, int(round(float(args.vel_window_s) / dt)))
    launch_stride = max(1, int(round(float(args.launch_step_min) * 60.0 / dt)))
    rows = []

    for tp in args.tracks:
        path = Path(tp)
        source_track_id = str(path.expanduser().resolve())
        mmsi = path.stem.replace("track_", "")
        df = load_track(path)
        if df.empty:
            log.warning("%s 空，跳过。", path.name)
            continue
        track_start = pd.to_datetime(df["t"].min(), utc=True).isoformat()
        track_end = pd.to_datetime(df["t"].max(), utc=True).isoformat()
        lat0, lon0 = float(df["lat"].mean()), float(df["lon"].mean())
        t = to_seconds(df["t"])
        x, y = local_xy(df["lat"].to_numpy(), df["lon"].to_numpy(), lat0, lon0)
        gaps = np.where(np.diff(t) > seg_gap_s)[0]
        bounds = np.concatenate([[0], gaps + 1, [len(t)]])
        segs = []
        for segment_id in range(len(bounds) - 1):
            a, b = bounds[segment_id], bounds[segment_id + 1]
            if b - a < 3:
                continue
            raw_t = t[a:b]
            grid = np.arange(raw_t[0], raw_t[-1] + 1e-9, dt)
            if len(grid) < vw + int(round(max(horizons) * 60.0 / dt)) + 2:
                continue
            gx = np.interp(grid, raw_t, x[a:b])
            gy = np.interp(grid, raw_t, y[a:b])
            segs.append((int(segment_id), grid, gx, gy))
        if not segs:
            log.warning("%s 无足够长连续段。", path.name)
            continue

        candidates = []
        max_h_steps = int(round(max(horizons) * 60.0 / dt))
        for segment_id, grid, gx, gy in segs:
            if args.sampling_mode == "grid":
                indices = range(vw, len(grid) - max_h_steps, launch_stride)
            else:
                indices = range(vw, len(grid) - max_h_steps)
            for i in indices:
                launch_lon, launch_lat = xy_to_lonlat(gx[i], gy[i], lat0, lon0)
                if bbox is not None and not (bbox[0] <= launch_lon <= bbox[1] and
                                             bbox[2] <= launch_lat <= bbox[3]):
                    continue
                candidates.append((segment_id, grid, gx, gy, i, launch_lon, launch_lat))
        if not candidates:
            log.warning("%s 在时间/海域约束下无合法起飞点。", path.name)
            continue

        if args.sampling_mode == "random":
            chosen_pos = np.sort(rng.choice(len(candidates),
                                            size=min(args.n_per_track, len(candidates)),
                                            replace=False))
        else:
            chosen_pos = deterministic_indices(len(candidates), int(args.max_launches_per_track))
        chosen = [candidates[int(i)] for i in chosen_pos]
        rows_before = len(rows)

        for segment_id, grid, gx, gy, i, launch_lon, launch_lat in chosen:
            vx = (gx[i] - gx[i - vw]) / (vw * dt)
            vy = (gy[i] - gy[i - vw]) / (vw * dt)
            sp = math.hypot(vx, vy)
            c0 = math.degrees(math.atan2(gy[i] - gy[i - vw], gx[i] - gx[i - vw]))
            cw = max(1, vw // 2)
            if i - vw - cw >= 0:
                c1 = math.degrees(math.atan2(gy[i - cw] - gy[i - vw - cw],
                                             gx[i - cw] - gx[i - vw - cw]))
                dcourse = (c0 - c1 + 180.0) % 360.0 - 180.0
                turn_rate = dcourse / (cw * dt / 60.0)
            else:
                turn_rate = 0.0
            disp_win = math.hypot(gx[i] - gx[i - vw], gy[i] - gy[i - vw])
            state = classify_state(sp, turn_rate, disp_win,
                                   args.low_kn, args.dp_kn, args.dp_disp_m,
                                   args.turn_deg_min)
            t_L_s = float(grid[i])
            t_L_iso = pd.to_datetime(t_L_s, unit="s", utc=True).isoformat()

            for h in horizons:
                hs = int(round(h * 60.0 / dt))
                j = i + hs
                if j >= len(grid):
                    continue
                px = gx[i] + vx * (hs * dt)
                py = gy[i] + vy * (hs * dt)
                pred_lon, pred_lat = xy_to_lonlat(px, py, lat0, lon0)
                true_lon, true_lat = xy_to_lonlat(gx[j], gy[j], lat0, lon0)
                xi_e = float(gx[j] - px); xi_n = float(gy[j] - py)
                rows.append(dict(
                    mmsi=mmsi, source_track=path.name, source_track_id=source_track_id,
                    segment_id=segment_id,
                    predictor=PREDICTOR_NAME, predictor_contract=PREDICTOR_CONTRACT,
                    timestamp_epoch_contract=TIMESTAMP_EPOCH_CONTRACT,
                    sampling_mode=args.sampling_mode,
                    track_start=track_start, track_end=track_end,
                    t_L=t_L_iso, t_R=pd.to_datetime(float(grid[j]), unit="s", utc=True).isoformat(),
                    h_min=int(h), c_state=state,
                    # CV has no turn-rate dynamics.  Derive the recovery state from the
                    # predicted motion instead of mechanically persisting a launch turn state.
                    recovery_state_pred=cv_recovery_state(np.array([vx, vy]), state),
                    recovery_state_source="cv-noleak-state-from-predicted-motion",
                    launch_lon=round(launch_lon, 6), launch_lat=round(launch_lat, 6),
                    pred_recover_lon=round(pred_lon, 6), pred_recover_lat=round(pred_lat, 6),
                    true_recover_lon=round(true_lon, 6), true_recover_lat=round(true_lat, 6),
                    xi_e_m=round(xi_e, 2), xi_n_m=round(xi_n, 2),
                    pred_err_m=round(math.hypot(xi_e, xi_n), 2)))
        log.info("%s: 起飞点=%d × horizons=%d → 场景行=%d",
                 path.name, len(chosen), len(horizons), len(rows) - rows_before)

    if not rows:
        log.error("没有生成任何场景。")
        return 2
    out = pd.DataFrame(rows).sort_values(["mmsi", "t_L", "h_min"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("\n==== 真实回收场景生成汇总 ====")
    print(f"预测器: {PREDICTOR_NAME} | 采样: {args.sampling_mode} | horizons: {horizons}")
    print(f"总场景行: {len(out)} | 船: {sorted(out.mmsi.astype(str).unique())}")
    print(f"状态分布: {out.c_state.value_counts().to_dict()}")
    print(f"预测误差 m: median={out.pred_err_m.median():.0f}, p95={out.pred_err_m.quantile(.95):.0f}, max={out.pred_err_m.max():.0f}")
    print(f"输出: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
