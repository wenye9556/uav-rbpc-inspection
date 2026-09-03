#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_fetch_turbines.py — 用 OpenStreetMap Overpass API 抓取并裁剪海上风机单机坐标。

流程(一步完成):
  1. 用"大框"向 Overpass 查询 power=generator + generator:source=wind 的节点/面。
     大框覆盖风场全域(含港口通勤区), 确保海上阵列不漏标。
  2. 查询结果在内存里按各风场的"紧致海上框"裁剪, 排除岸上风机。
  3. 直接输出 turbines_<farm>_clean.csv, 不保留中间原始文件。

用法:
  python step3_fetch_turbines.py                 # 默认: Anholt / Nysted / Rødsand II
  python step3_fetch_turbines.py --out-dir ./data/turbines

输出:
  <out-dir>/turbines_Anholt_clean.csv
  <out-dir>/turbines_Nysted_clean.csv
  <out-dir>/turbines_Rodsand_II_clean.csv
  列: turbine_id, lon, lat, farm, osm_id

校验: 与公开装机台数对照(Anholt 111, Nysted 72, Rødsand II 90)。
     若 OSM 缺标(如 Nysted 目前仅 70), 建议补查丹麦能源署 Energistyrelsen 主数据。

可选: 若要更稳的裁剪, 可改用 EMODnet 多边形做点-在-面(见文件末注释, 需 geopandas)。
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ── Overpass 查询框 (大框, 覆盖海上阵列 + 周边, 确保不漏) ─────────────────
# 格式: "查询组名" -> (N, W, S, E)
# 一个查询组可对应多个风场(下面 CLIP_CONFIG 按查询组拆分)
FETCH_BBOX = {
    "Anholt": (57.0, 10.618, 56.2, 11.818),
    "NysRod": (54.902, 11.032, 54.202, 12.232),
}

# ── 裁剪配置 (紧致海上框, 排除岸上风机) ────────────────────────────────────
# 格式: 输出风场名 -> (查询组名, lon_min, lon_max, lat_min, lat_max, 公开台数)
# BBOX 已核对: 落在海上阵列范围内, 不含陆上风机。
CLIP_CONFIG = {
    "Anholt":     ("Anholt", 11.05, 11.35, 56.45, 56.72, 111),
    "Nysted":     ("NysRod", 11.64, 11.80, 54.52, 54.58,  72),
    "Rodsand_II": ("NysRod", 11.40, 11.63, 54.50, 54.60,  90),
}

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


# ── Overpass 查询 ────────────────────────────────────────────────────────────

def build_query(n, w, s, e):
    bbox = f"{s},{w},{n},{e}"   # Overpass 格式: south,west,north,east
    return f"""
[out:json][timeout:180];
(
  node["power"="generator"]["generator:source"="wind"]({bbox});
  way["power"="generator"]["generator:source"="wind"]({bbox});
  node["power"="generator"]["generator:method"="wind_turbine"]({bbox});
  way["power"="generator"]["generator:method"="wind_turbine"]({bbox});
);
out center tags;
"""


def run_overpass(query, group_name):
    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for ep in ENDPOINTS:
        for attempt in range(3):
            try:
                logging.info("[%s] Overpass %s (尝试 %s)", group_name, ep, attempt + 1)
                req = urllib.request.Request(
                    ep, data=data, headers={"User-Agent": "turbine-fetch/1.0"})
                with urllib.request.urlopen(req, timeout=200) as r:
                    return json.loads(r.read().decode())
            except Exception as exc:
                last = exc
                logging.warning("[%s] 失败(%s), 重试", group_name, exc)
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"[{group_name}] Overpass 全部端点失败: {last}")


def parse_elements(result):
    """把 Overpass JSON 解析为 {osm_id, lon, lat} 列表。"""
    pts = []
    for el in result.get("elements", []):
        if el["type"] == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:
            c = el.get("center", {})
            lon, lat = c.get("lon"), c.get("lat")
        if lon is None or lat is None:
            continue
        pts.append({"osm_id": f"{el['type']}/{el['id']}",
                    "lon": lon, "lat": lat})
    return pts


# ── 裁剪 ─────────────────────────────────────────────────────────────────────

def clip(pts, lon_min, lon_max, lat_min, lat_max):
    return [p for p in pts
            if lon_min <= p["lon"] <= lon_max and lat_min <= p["lat"] <= lat_max]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="OSM 单机风机坐标 抓取 + 裁剪(一步完成)")
    ap.add_argument("--out-dir", type=Path, default=Path("./data"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 按查询组抓取原始点(在内存里)
    raw = {}
    for group, (n, w, s, e) in FETCH_BBOX.items():
        result = run_overpass(build_query(n, w, s, e), group)
        raw[group] = parse_elements(result)
        logging.info("[%s] 原始点 %s 个(含岸上)", group, len(raw[group]))

    # Step 2: 按风场裁剪 → 直接写 _clean.csv
    for farm, (group, lon_min, lon_max, lat_min, lat_max, n_official) in CLIP_CONFIG.items():
        if group not in raw:
            logging.warning("[%s] 查询组 %s 无数据, 跳过", farm, group)
            continue

        clipped = clip(raw[group], lon_min, lon_max, lat_min, lat_max)
        clipped_sorted = sorted(clipped, key=lambda p: (p["lat"], p["lon"]))

        out = args.out_dir / f"turbines_{farm}_clean.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["turbine_id", "lon", "lat", "farm", "osm_id"])
            for i, p in enumerate(clipped_sorted, 1):
                w.writerow([f"{farm}_{i:03d}",
                            round(p["lon"], 6), round(p["lat"], 6),
                            farm, p["osm_id"]])

        n_got = len(clipped)
        if n_got == n_official:
            flag = f"✓ 与公开台数一致({n_official})"
        else:
            flag = (f"⚠ 得到 {n_got} / 公开 {n_official}; "
                    f"OSM 可能缺标, 建议补查丹麦能源署 Energistyrelsen")
        logging.info("[%s] 裁剪后 %s 台 %s -> %s", farm, n_got, flag, out.name)

    logging.info("完成。")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 可选: 用 EMODnet 风场多边形做更稳的点-在-面裁剪 (需 geopandas):
#   import geopandas as gpd
#   farms = gpd.read_file("emodnet_windfarmspoly_all.geojson")
#   poly = farms[farms["farm_id"] == "windfarmspoly.46"].geometry.iloc[0]
#   pts_gdf = gpd.GeoDataFrame(pts_df, geometry=gpd.points_from_xy(pts_df.lon, pts_df.lat),
#                               crs="EPSG:4326")
#   inside = pts_gdf[pts_gdf.within(poly)]
# ---------------------------------------------------------------------------
