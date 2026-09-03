#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_export_tracks.py — 把 step1_fetch_ais.py 产出的逐点日文件合并成"每船一条连续轨迹"。

用途(P2 前置):
  船位预测器与误差 ξ 统计需要选定 CTV 的【逐点连续轨迹】(timestamp/lat/lon/sog/cog/heading),
  而 step1_fetch_ais.py 产出的是按天切片的 aisdk-YYYY-MM-DD.csv。本脚本把它们:
    1) 跨天拼接;2) 按 MMSI 拆分;3) DD/MM/YYYY dayfirst 正确解析时间戳(项目已知坑);
    4) 按时间排序、去重;5) 每船写 track_<mmsi>.csv,并合并写 ctv_tracks.csv。

输入:一个目录,内含 step1_fetch_ais.py 的逐点日文件 aisdk-YYYY-MM-DD.csv
      (用 --keep-scope mmsi_only 或 region_or_mmsi 产出;summary_only 不含逐点行,不可用)。

典型用法(Case B 两艘 CTV):
  python step6_export_tracks.py --in-dir ./ais_filtered \
      --mmsi 219018788,219028973 --out-dir ./tracks

  # 若你的逐点文件含全部 6 艘,本脚本会在合并时过滤到 --mmsi 指定的船,无需重跑过滤。
  # 若 ./ais_filtered 里只有 _vessel_day_summary.csv(没有 aisdk-*.csv 逐点文件),
  # 说明上次是 summary_only,请先重跑 step1_fetch_ais.py(见 README/本文件末注释)。

输出(--out-dir):
  track_<mmsi>.csv     每船一条按时间排序的连续轨迹
  ctv_tracks.csv       两船合并(含 mmsi 列)
  _tracks_report.csv   每船点数/时间跨度/天数/采样间隔中位数/疑似缺口
"""
from __future__ import annotations
import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 逐点日文件里(step1_fetch_ais.py 写出的)列名是规整后的 role 名:
# timestamp / mmsi / latitude / longitude / sog / cog / heading / nav_status / region / date
ROLE_CANDS = {
    "timestamp": ["timestamp", "# timestamp", "datetime", "t"],
    "mmsi": ["mmsi"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "sog": ["sog", "speedoverground", "speed"],
    "cog": ["cog", "courseoverground", "course"],
    "heading": ["heading", "trueheading"],
    "nav_status": ["nav_status", "navigationalstatus", "navigationstatus", "navstatus"],
}


def _norm(c: str) -> str:
    return "".join(ch for ch in str(c).strip().lower() if ch.isalnum())


def find_col(columns, cands):
    norm = {_norm(c): c for c in columns}
    for cand in cands:
        if _norm(cand) in norm:
            return norm[_norm(cand)]
    return None


def load_one(path: Path, keep_mmsi: set[str]) -> pd.DataFrame | None:
    try:
        head = pd.read_csv(path, nrows=3, dtype=str)
    except Exception as exc:
        logging.warning("跳过不可读文件 %s (%s)", path.name, exc)
        return None
    cols = list(head.columns)
    m = {role: find_col(cols, cands) for role, cands in ROLE_CANDS.items()}
    for req in ("timestamp", "mmsi", "latitude", "longitude"):
        if m[req] is None:
            logging.warning("%s 缺列 %s,跳过。实际列:%s", path.name, req, cols)
            return None
    usecols = [v for v in m.values() if v]
    df = pd.read_csv(path, usecols=usecols, dtype=str, on_bad_lines="skip", low_memory=False)
    df = df.rename(columns={v: k for k, v in m.items() if v})
    df["mmsi"] = df["mmsi"].astype(str).str.strip()
    if keep_mmsi:
        df = df[df["mmsi"].isin(keep_mmsi)]
    return df if not df.empty else None


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def main() -> int:
    ap = argparse.ArgumentParser(description="合并逐点日文件为每船连续轨迹(Case B CTV)")
    ap.add_argument("--in-dir", type=Path, default=Path("./ais_filtered"),
                    help="含 aisdk-YYYY-MM-DD.csv 逐点日文件的目录")
    ap.add_argument("--mmsi", type=str, default="219018788,219028973",
                    help="逗号分隔的目标 CTV(Case B 默认 CARRIER+219028973)")
    ap.add_argument("--out-dir", type=Path, default=Path("./tracks"))
    ap.add_argument("--glob", type=str, default="aisdk-*.csv", help="逐点日文件名模式")
    ap.add_argument("--gap-min", type=float, default=30.0,
                    help="相邻点时间差超过该分钟数计为一次缺口")
    args = ap.parse_args()

    keep = {x.strip() for x in args.mmsi.split(",") if x.strip()}
    files = sorted(glob.glob(str(args.in_dir / args.glob)))
    if not files:
        logging.error("在 %s 下未找到 %s 逐点文件。", args.in_dir, args.glob)
        logging.error("若该目录只有 _vessel_day_summary.csv,说明上次用了 --keep-scope summary_only,")
        logging.error("请先重跑(Case B 例):")
        logging.error('  python step1_fetch_ais.py --start 2025-03-01 --end 2025-06-30 \\')
        logging.error('     --raw-dir ./data --out-dir ./ais_filtered --no-download \\')
        logging.error('     --keep-scope mmsi_only --mmsi %s \\', args.mmsi)
        logging.error("     --bbox 'NysRod:54.902,11.032,54.202,12.232'")
        return 2

    logging.info("发现 %d 个逐点日文件,合并中(目标 MMSI=%s) ...", len(files), sorted(keep))
    parts = []
    for fp in files:
        d = load_one(Path(fp), keep)
        if d is not None:
            parts.append(d)
    if not parts:
        logging.error("目标 MMSI 在这些文件里没有任何逐点记录。检查 --mmsi 是否正确。")
        return 3

    df = pd.concat(parts, ignore_index=True)
    # 关键:aisdk 时间戳是 DD/MM/YYYY,必须 dayfirst=True(否则 3月1日→1月3日)
    df["t"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df["lat"] = to_num(df["latitude"])
    df["lon"] = to_num(df["longitude"])
    for opt in ("sog", "cog", "heading"):
        if opt in df.columns:
            df[opt] = to_num(df[opt])
    before = len(df)
    df = df.dropna(subset=["t", "lat", "lon"])
    # 基本越界过滤(丹麦海域)
    df = df[(df.lat.between(53.0, 58.5)) & (df.lon.between(7.0, 16.0))]
    logging.info("有效点 %d / %d(丢弃时间或坐标缺失/越界 %d)", len(df), before, before - len(df))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [c for c in ["t", "mmsi", "lat", "lon", "sog", "cog", "heading", "nav_status", "region"] if c in df.columns]

    report = []
    combined = []
    for mm, g in df.groupby("mmsi"):
        g = g.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
        out = g[keep_cols].copy()
        out_path = args.out_dir / f"track_{mm}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        combined.append(out.assign(mmsi=mm))

        dt_min = g["t"].diff().dt.total_seconds().div(60)
        gaps = int((dt_min > args.gap_min).sum())
        report.append({
            "mmsi": mm,
            "points": len(g),
            "first": g["t"].min(),
            "last": g["t"].max(),
            "days_span": (g["t"].max() - g["t"].min()).days + 1,
            "distinct_days": g["t"].dt.date.nunique(),
            "median_dt_sec": round(float(dt_min.median() * 60), 1) if len(g) > 1 else None,
            "gaps_gt_%dmin" % int(args.gap_min): gaps,
        })
        logging.info("track_%s.csv: %d 点 | %s → %s | %d 天 | 缺口(>%.0fmin)=%d",
                     mm, len(g), g["t"].min(), g["t"].max(), g["t"].dt.date.nunique(), args.gap_min, gaps)

    pd.concat(combined, ignore_index=True).to_csv(args.out_dir / "ctv_tracks.csv",
                                                  index=False, encoding="utf-8-sig")
    rep = pd.DataFrame(report)
    rep.to_csv(args.out_dir / "_tracks_report.csv", index=False, encoding="utf-8-sig")
    print("\n==== 轨迹导出汇总 ====")
    print(rep.to_string(index=False))
    print(f"\n输出目录: {args.out_dir.resolve()}")
    print("  track_<mmsi>.csv  每船连续轨迹(已 dayfirst 解析+按时间排序+去重)")
    print("  ctv_tracks.csv    两船合并")
    print("  _tracks_report.csv 点数/跨度/采样间隔/缺口")
    return 0


if __name__ == "__main__":
    sys.exit(main())
