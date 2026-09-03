#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_match_windfarms.py  （精简实现）

输入目录应包含:
- aisdk-*.csv                              (丹麦海事局 AIS, step1 产出, 默认在 ./ais_filtered/)
- emodnet_windfarmspoly_all.geojson        (EMODnet 风场边界多边形, 默认在 ./data/ 下)

相对 的修订要点:
  1. 时间戳按 DD/MM/YYYY 解析 (dayfirst=True), 并在汇总里输出原始时间戳样本,
     用来核验 "文件名日期 vs 解析出的日期" 谁对 (aisdk 内部是 DD/MM/YYYY)。
  2. 默认只对 Production(已投运) 风场做空间归属 (--status-keywords 可调)。
  3. 每个 AIS 点只归给【最近的一个 Production 风场】(sjoin_nearest),
     彻底消除 中 30km 缓冲重叠造成的 ~2.49 倍重复计数。
  4. 抽样核验 10/15/20/30 km 缓冲的 "每点平均落入风场数"(重叠倍数), 辅助选半径。
  5. 提取 AIS 静态字段 (船名/船型/Type of mobile/长/宽), 按 MMSI 取众数;
     与 MMSI 编码规则联合, 把基站(00)/航标(99)/SAR(97,111)/真实船舶区分开。
  6. 预筛候选运维船 (真实船舶 + 有 SOG + 航速合理 + 风场内停留多 + 归属风场集中) 并打分。
  7. 精简输出: 4 个 CSV + 1 个 Markdown
       01_file_summary.csv        每个源文件的质量/匹配/原始时间戳样本
       02_farm_production.csv      仅 Production 风场, 多半径唯一点数/船数/航速
       03_vessel_candidates.csv    真实船舶 (含静态信息) + 候选打分
       04_vessel_farm_visits.csv   候选船 × 风场 × 日期 的到访/停留(便于今后做往返识别)
       data_usability_report.md    可读报告(含日期核验结论与重叠倍数对比)
  (可选) --trajectory-mmsi 指定 MMSI 时, 额外导出 05_trajectory_<mmsi>.csv 原始轨迹点。

注意:
  - 大型 AIS CSV 用 chunksize 分块读取, 不会一次性加载整个文件。
  - EMODnet 多边形是风场边界, 不是单机风机坐标。
  - 本程序只做 "数据可用性与空间匹配探索", 不能直接证明某船一定是运维船。
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


WGS84 = "EPSG:4326"
EUROPE_METRIC_CRS = "EPSG:3035"

# 字段自动识别候选 (含 没有的静态字段)
AIS_CANDIDATES = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "basedatetime", "basetime", "utc"],
    "mmsi": ["mmsi"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "lng", "x"],
    "sog": ["sog", "speedoverground", "speed_over_ground", "speed"],
    "cog": ["cog", "courseoverground", "course_over_ground", "course"],
    "heading": ["heading", "trueheading"],
    "rot": ["rot", "rateofturn", "rate_of_turn"],
    "nav_status": ["navigationalstatus", "navigationstatus", "navstatus"],
    "ship_type": ["shiptype", "vesseltype"],          # aisdk "Ship type"
    "type_of_mobile": ["typeofmobile", "mobiletype"],  # aisdk "Type of mobile"
    "name": ["name", "shipname", "vesselname"],
    "length": ["length", "loa", "shiplength"],
    "width": ["width", "beam", "shipwidth"],
    "imo": ["imo"],
    "callsign": ["callsign", "call_sign"],
}

FARM_STATUS_CANDIDATES = ["status", "status_en", "operational_status", "development_status", "phase"]
FARM_NAME_CANDIDATES = ["name", "windfarm_name", "wind_farm_name", "site_name", "project_name", "title"]
FARM_ID_CANDIDATES = ["id", "objectid", "fid", "gid", "windfarm_id", "wind_farm_id", "site_id"]

# 海事 MID(MMSI 前 3 位)→ 船籍国, 仅列北海/波罗的海相关常见者
MID_TO_FLAG = {
    "219": "DK", "220": "DK", "211": "DE", "218": "DE", "265": "SE", "266": "SE",
    "257": "NO", "258": "NO", "259": "NO", "244": "NL", "245": "NL", "246": "NL",
    "232": "GB", "233": "GB", "234": "GB", "235": "GB", "230": "FI", "231": "FO",
    "227": "FR", "226": "FR", "228": "FR", "212": "CY", "209": "CY", "210": "CY",
    "304": "AG", "305": "AG", "311": "BS", "319": "KY", "538": "MH", "636": "LR",
}

LOW_SPEED_KN = 2.0   # "低速/在场" 阈值, 用于估计作业/停留占比
SOG_VALID_MAX = 80.0
SOG_PLAUSIBLE_MAX = 30.0  # 船舶平均航速上限, 超过视为异常 AIS


# ----------------------------------------------------------------------------
# 通用工具 (保留 中稳健的实现)
# ----------------------------------------------------------------------------
def normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(columns, candidates):
    mapping = {normalise_name(c): c for c in columns}
    for cand in candidates:
        cn = normalise_name(cand)
        if cn in mapping:
            return mapping[cn]
    for cand in candidates:
        cn = normalise_name(cand)
        for col_norm, original in mapping.items():
            if len(cn) >= 4 and (cn in col_norm or col_norm in cn):
                return original
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )


def sniff_csv(path: Path):
    sample_bytes = path.open("rb").read(200000)
    for encoding in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            sample = sample_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return encoding, dialect.delimiter
        except csv.Error:
            counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
            return encoding, max(counts, key=counts.get)
    return "utf-8-sig", ","


def detect_ais_schema(csv_path: Path):
    encoding, sep = sniff_csv(csv_path)
    sample = pd.read_csv(csv_path, sep=sep, encoding=encoding, nrows=30,
                         dtype=str, on_bad_lines="skip", low_memory=False)
    mapping = {role: find_column(sample.columns, cands) for role, cands in AIS_CANDIDATES.items()}
    return mapping, sample, encoding, sep


def is_operational(value, keywords) -> bool:
    if pd.isna(value):
        return False
    v = str(value).lower()
    return any(w in v for w in keywords)


def union_geometry(gdf):
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


def classify_mmsi_numeric(mmsi: str) -> str:
    """仅凭 MMSI 数字的快速分类 (chunk 内用)。"""
    s = str(mmsi)
    if not s.isdigit():
        return "invalid"
    if len(s) < 9:                       # int 解析会吃掉前导零
        full = s.zfill(9)
        if full.startswith("00"):
            return "base_station"
        if full.startswith("99"):
            return "aid_to_nav"
        return "short"
    if len(s) == 9:
        if s[0] in "234567":
            return "ship"
        if s.startswith("98"):
            return "craft_assoc"
        if s.startswith("99"):
            return "aid_to_nav"
        if s.startswith("97") or s.startswith("111"):
            return "sar"
    return "other"


def flag_from_mmsi(mmsi: str) -> str:
    s = str(mmsi).zfill(9)
    return MID_TO_FLAG.get(s[:3], f"Other({s[:3]})")


# ----------------------------------------------------------------------------
# 风场准备: 读取多边形, 仅保留目标状态, 生成米制几何
# ----------------------------------------------------------------------------
def prepare_windfarms(geojson_path: Path, status_keywords, output_dir: Path):
    logging.info("读取风场文件: %s", geojson_path.name)
    farms = gpd.read_file(geojson_path)
    if farms.empty:
        raise RuntimeError("风场 GeoJSON 为空。")

    farms = farms.set_crs(WGS84) if farms.crs is None else farms.to_crs(WGS84)

    non_geom = [c for c in farms.columns if c != farms.geometry.name]
    status_col = find_column(non_geom, FARM_STATUS_CANDIDATES)
    name_col = find_column(non_geom, FARM_NAME_CANDIDATES)
    id_col = find_column(non_geom, FARM_ID_CANDIDATES)

    farms = farms.copy()
    farms["farm_id"] = farms[id_col].astype(str) if id_col else [f"farm_{i:04d}" for i in range(len(farms))]
    farms["farm_name"] = farms[name_col].fillna("Unknown").astype(str) if name_col else farms["farm_id"]
    farms["farm_status"] = farms[status_col].fillna("Unknown").astype(str) if status_col else "Unknown"
    farms["is_operational_keyword"] = farms["farm_status"].map(lambda v: is_operational(v, status_keywords))
    farms["geometry"] = farms.geometry.buffer(0)  # 修复自相交

    # 全部风场的状态分布(用于报告)
    status_counts = farms["farm_status"].value_counts(dropna=False)

    target = farms[farms["is_operational_keyword"]].copy()
    if target.empty:
        raise RuntimeError(
            f"按关键词 {status_keywords} 未筛到任何风场。可用状态: {list(status_counts.index)}"
        )

    target_metric = target.to_crs(EUROPE_METRIC_CRS).reset_index(drop=True)
    target_metric["area_km2"] = target_metric.geometry.area / 1_000_000
    cent = target_metric.geometry.centroid.to_crs(WGS84)
    target_metric["centroid_lon"] = cent.x.values
    target_metric["centroid_lat"] = cent.y.values
    target_metric["sea_region"] = [
        coarse_region(lo, la)
        for lo, la in zip(target_metric["centroid_lon"], target_metric["centroid_lat"])
    ]

    logging.info(
        "风场总数 %s; 命中目标状态(默认 Production) %s 个。",
        len(farms), len(target_metric),
    )
    info = {
        "status_col": status_col, "name_col": name_col, "id_col": id_col,
        "farm_count": int(len(farms)),
        "target_count": int(len(target_metric)),
        "status_counts": status_counts.to_dict(),
        "status_keywords": status_keywords,
    }
    return target_metric, info


def coarse_region(lon, lat):
    """粗略海区标签(仅辅助区分丹麦西海岸 vs 波罗的海, 非严格 EEZ)。"""
    if lon < 9.0:
        return "North_Sea"
    if lat >= 56.0:
        return "Kattegat_Skagerrak"
    return "Baltic"


# ----------------------------------------------------------------------------
# 处理单个 AIS 文件
# ----------------------------------------------------------------------------
def new_farm_rec():
    return {
        "farm_name": "", "farm_status": "",
        "pts10": 0, "pts15": 0, "pts20": 0, "pts30": 0,
        "inside": 0, "low_speed": 0,
        "sog_sum": 0.0, "sog_count": 0,
        "mmsi_ship": set(), "mmsi_inside": set(),
        "first_time": None, "last_time": None,
    }


def new_vessel_rec():
    return {
        "records": 0, "inside": 0, "low_speed": 0,
        "sog_sum": 0.0, "sog_count": 0,
        "farms": defaultdict(int), "dates": set(),
        "first_time": None, "last_time": None,
        "name": Counter(), "ship_type": Counter(), "type_of_mobile": Counter(),
        "length": Counter(), "width": Counter(),
    }


def new_visit_rec():
    return {"records": 0, "inside": 0, "sog_sum": 0.0, "sog_count": 0,
            "first_time": None, "last_time": None}


def upd_bounds(rec, ts_series):
    valid = ts_series.dropna()
    if valid.empty:
        return
    lo, hi = valid.min(), valid.max()
    if rec["first_time"] is None or lo < rec["first_time"]:
        rec["first_time"] = lo
    if rec["last_time"] is None or hi > rec["last_time"]:
        rec["last_time"] = hi


def process_ais_file(csv_path, farms_metric, radii_m, chunksize,
                     farm_stats, vessel_stats, visit_stats,
                     mult_sample, mult_sample_cap,
                     trajectory_mmsi, trajectory_rows, dayfirst):
    mapping, schema_sample, encoding, sep = detect_ais_schema(csv_path)
    required = ["timestamp", "mmsi", "latitude", "longitude"]
    missing = [n for n in required if mapping[n] is None]
    if missing:
        raise RuntimeError(f"{csv_path.name} 缺少字段 {missing}; 实际列: {list(schema_sample.columns)}")

    logging.info("处理 AIS 文件: %s", csv_path.name)
    logging.info("字段映射: %s", {k: v for k, v in mapping.items() if v})

    # 原始时间戳样本(核验日期解析用)
    raw_ts_sample = []
    if mapping["timestamp"] in schema_sample.columns:
        raw_ts_sample = (
            schema_sample[mapping["timestamp"]].dropna().astype(str).head(3).tolist()
        )

    max_r = max(max(radii_m), 30000)
    summary = {
        "source_file": csv_path.name, "rows_read": 0,
        "valid_coordinates": 0, "valid_timestamp": 0, "valid_mmsi": 0,
        "sog_count": 0, "sog_sum": 0.0,
        "assigned_points": 0, "inside_points": 0,
        "unique_mmsi": set(),
        "time_start": None, "time_end": None,
        "raw_timestamp_sample": " | ".join(raw_ts_sample),
    }

    usecols = sorted({v for v in mapping.values() if v})
    static_cols = ["name", "ship_type", "type_of_mobile"]
    farms_join = farms_metric[["farm_id", "farm_name", "farm_status", "geometry"]]

    reader = pd.read_csv(csv_path, sep=sep, encoding=encoding, usecols=usecols,
                         dtype=str, chunksize=chunksize, on_bad_lines="skip",
                         low_memory=False)

    for chunk_no, raw in enumerate(reader, start=1):
        summary["rows_read"] += len(raw)
        df = pd.DataFrame(index=raw.index)

        df["timestamp"] = pd.to_datetime(
            raw[mapping["timestamp"]], dayfirst=dayfirst, utc=True, errors="coerce"
        )
        df["mmsi"] = raw[mapping["mmsi"]].astype(str).str.strip()
        df["mmsi"] = df["mmsi"].where(df["mmsi"].str.fullmatch(r"\d{6,10}", na=False))
        df["latitude"] = safe_numeric(raw[mapping["latitude"]])
        df["longitude"] = safe_numeric(raw[mapping["longitude"]])
        df["sog"] = safe_numeric(raw[mapping["sog"]]) if mapping["sog"] else np.nan
        for f in ["cog", "heading"]:
            df[f] = safe_numeric(raw[mapping[f]]) if mapping[f] else np.nan
        df["nav_status"] = raw[mapping["nav_status"]].astype(str) if mapping["nav_status"] else ""
        for f in ["name", "ship_type", "type_of_mobile"]:
            df[f] = raw[mapping[f]].astype(str) if mapping[f] else ""
        df["length"] = safe_numeric(raw[mapping["length"]]) if mapping["length"] else np.nan
        df["width"] = safe_numeric(raw[mapping["width"]]) if mapping["width"] else np.nan

        valid_coord = (df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
                       & ~((df["latitude"] == 0) & (df["longitude"] == 0)))
        valid_time = df["timestamp"].notna()
        valid_mmsi = df["mmsi"].notna()

        summary["valid_coordinates"] += int(valid_coord.sum())
        summary["valid_timestamp"] += int(valid_time.sum())
        summary["valid_mmsi"] += int(valid_mmsi.sum())
        summary["unique_mmsi"].update(df.loc[valid_mmsi, "mmsi"].unique().tolist())

        vts = df.loc[valid_time, "timestamp"]
        if not vts.empty:
            lo, hi = vts.min(), vts.max()
            if summary["time_start"] is None or lo < summary["time_start"]:
                summary["time_start"] = lo
            if summary["time_end"] is None or hi > summary["time_end"]:
                summary["time_end"] = hi

        good_sog = df.loc[df["sog"].between(0, SOG_VALID_MAX), "sog"]
        summary["sog_sum"] += float(good_sog.sum())
        summary["sog_count"] += int(good_sog.count())

        spatial = df.loc[valid_coord & valid_mmsi].copy()
        if spatial.empty:
            continue

        pts = gpd.GeoDataFrame(
            spatial,
            geometry=gpd.points_from_xy(spatial["longitude"], spatial["latitude"]),
            crs=WGS84,
        ).to_crs(EUROPE_METRIC_CRS)

        # 最近 Production 风场单一归属: 每个点只匹配一个最近风场 (距离 <= max_r)
        if not hasattr(gpd, "sjoin_nearest"):
            raise RuntimeError(
                "当前 GeoPandas 不支持 sjoin_nearest, 请升级到 >= 0.10 "
                "(并安装 shapely>=2 / pygeos)。"
            )
        joined = gpd.sjoin_nearest(
            pts, farms_join, how="inner", max_distance=max_r, distance_col="dist_m"
        )
        if joined.empty:
            continue
        # sjoin_nearest 在并列最近时可能给同一点多行, 去重保留首个
        joined = joined[~joined.index.duplicated(keep="first")]

        joined["inside"] = joined["dist_m"] <= 0.0
        joined["date"] = joined["timestamp"].dt.date

        summary["assigned_points"] += len(joined)
        summary["inside_points"] += int(joined["inside"].sum())

        # 重叠倍数抽样(累积近场点几何, 末尾统一在多半径下计数)
        if len(mult_sample) < mult_sample_cap:
            take = min(len(joined), mult_sample_cap - len(mult_sample))
            if take > 0:
                idx = joined.index.to_numpy()
                if take < len(joined):
                    idx = np.random.default_rng(chunk_no).choice(idx, size=take, replace=False)
                mult_sample.extend(joined.loc[idx, "geometry"].tolist())

        # ---- 按风场聚合 (最近归属, 每点一次) ----
        for fid, sub in joined.groupby("farm_id"):
            rec = farm_stats[str(fid)]
            rec["farm_name"] = str(sub["farm_name"].iloc[0])
            rec["farm_status"] = str(sub["farm_status"].iloc[0])
            d = sub["dist_m"]
            rec["pts30"] += int((d <= 30000).sum())
            rec["pts20"] += int((d <= 20000).sum())
            rec["pts15"] += int((d <= 15000).sum())
            rec["pts10"] += int((d <= 10000).sum())
            rec["inside"] += int(sub["inside"].sum())
            is_ship = sub["mmsi"].map(lambda m: classify_mmsi_numeric(m) == "ship")
            rec["mmsi_ship"].update(sub.loc[is_ship, "mmsi"].unique().tolist())
            rec["mmsi_inside"].update(sub.loc[is_ship & sub["inside"], "mmsi"].unique().tolist())
            ssog = sub["sog"].dropna()
            rec["sog_sum"] += float(ssog[ssog.between(0, SOG_VALID_MAX)].sum())
            rec["sog_count"] += int(ssog.between(0, SOG_VALID_MAX).sum())
            rec["low_speed"] += int((ssog.between(0, LOW_SPEED_KN)).sum())
            upd_bounds(rec, sub["timestamp"])

        # ---- 按船舶聚合 ----
        for mmsi, sub in joined.groupby("mmsi"):
            rec = vessel_stats[str(mmsi)]
            rec["records"] += len(sub)
            rec["inside"] += int(sub["inside"].sum())
            for fid2, cnt in sub.groupby("farm_id").size().items():
                rec["farms"][str(fid2)] += int(cnt)
            rec["dates"].update([d for d in sub["date"].dropna().unique()])
            ssog = sub["sog"].dropna()
            ssog = ssog[ssog.between(0, SOG_VALID_MAX)]
            rec["sog_sum"] += float(ssog.sum())
            rec["sog_count"] += int(ssog.count())
            rec["low_speed"] += int(ssog.between(0, LOW_SPEED_KN).sum())
            upd_bounds(rec, sub["timestamp"])
            for col in static_cols:
                vals = sub[col].dropna().astype(str).str.strip()
                vals = vals[(vals != "") & (vals.str.lower() != "nan") & (vals != "Undefined")]
                rec[col].update(vals.tolist())
            for col in ["length", "width"]:
                vals = sub[col].dropna()
                rec[col].update([round(float(x)) for x in vals if x > 0])

            # ---- 候选船 × 风场 × 日期 (仅真实船舶, 控制体积) ----
            if classify_mmsi_numeric(mmsi) == "ship":
                for (fid3, dt), g in sub.dropna(subset=["timestamp"]).groupby(["farm_id", "date"]):
                    vr = visit_stats[(str(mmsi), str(fid3), str(dt))]
                    vr["records"] += len(g)
                    vr["inside"] += int(g["inside"].sum())
                    gs = g["sog"].dropna()
                    gs = gs[gs.between(0, SOG_VALID_MAX)]
                    vr["sog_sum"] += float(gs.sum())
                    vr["sog_count"] += int(gs.count())
                    upd_bounds(vr, g["timestamp"])

            # ---- 可选: 指定 MMSI 的原始轨迹导出 ----
            if trajectory_mmsi and str(mmsi) in trajectory_mmsi:
                keep = sub[["timestamp", "latitude", "longitude", "sog", "cog",
                            "heading", "nav_status", "farm_id", "dist_m", "inside"]].copy()
                keep.insert(0, "mmsi", str(mmsi))
                trajectory_rows[str(mmsi)].append(keep)

        if chunk_no % 20 == 0:
            logging.info("%s: 已处理 %s 块, 累计读取 %s 行, 归属点 %s。",
                         csv_path.name, chunk_no, summary["rows_read"], summary["assigned_points"])

    summary["unique_mmsi"] = len(summary["unique_mmsi"])
    summary["mean_sog_knots"] = (summary["sog_sum"] / summary["sog_count"]
                                 if summary["sog_count"] else np.nan)
    return summary


# ----------------------------------------------------------------------------
# 组装输出表
# ----------------------------------------------------------------------------
def mode_or_blank(counter: Counter):
    return counter.most_common(1)[0][0] if counter else ""


def build_farm_table(farm_stats):
    rows = []
    for fid, r in farm_stats.items():
        rows.append({
            "farm_id": fid, "farm_name": r["farm_name"], "farm_status": r["farm_status"],
            "pts_within_30km": r["pts30"], "pts_within_20km": r["pts20"],
            "pts_within_15km": r["pts15"], "pts_within_10km": r["pts10"],
            "pts_inside_polygon": r["inside"],
            "unique_ship_mmsi": len(r["mmsi_ship"]),
            "unique_ship_mmsi_inside_polygon": len(r["mmsi_inside"]),
            "low_speed_share": (r["low_speed"] / r["sog_count"]) if r["sog_count"] else np.nan,
            "mean_sog_knots": (r["sog_sum"] / r["sog_count"]) if r["sog_count"] else np.nan,
            "first_timestamp_utc": r["first_time"], "last_timestamp_utc": r["last_time"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pts_inside_polygon", ascending=False)
    return df


def build_vessel_table(vessel_stats, farm_centroids):
    rows = []
    for mmsi, r in vessel_stats.items():
        farms = r["farms"]
        n_farms = len(farms)
        dominant = max(farms, key=farms.get) if farms else ""
        mean_sog = (r["sog_sum"] / r["sog_count"]) if r["sog_count"] else np.nan
        cls = classify_mmsi_numeric(mmsi)
        tmob = mode_or_blank(r["type_of_mobile"])
        # 用 Type of mobile 修正分类(更权威)
        cls_final = cls
        tl = tmob.lower()
        if "base" in tl:
            cls_final = "base_station"
        elif "aton" in tl or "aid" in tl:
            cls_final = "aid_to_nav"
        elif "sar" in tl or "search" in tl:
            cls_final = "sar"
        elif "class a" in tl or "class b" in tl:
            cls_final = "ship"
        is_ship = cls_final == "ship"
        rows.append({
            "mmsi": mmsi, "flag": flag_from_mmsi(mmsi),
            "mmsi_class": cls_final, "type_of_mobile": tmob,
            "name": mode_or_blank(r["name"]), "ship_type": mode_or_blank(r["ship_type"]),
            "length_m": mode_or_blank(r["length"]), "width_m": mode_or_blank(r["width"]),
            "assigned_points": r["records"], "inside_polygon_points": r["inside"],
            "n_production_farms": n_farms, "dominant_farm": dominant,
            "mean_sog_knots": round(mean_sog, 3) if pd.notna(mean_sog) else np.nan,
            "low_speed_share": round(r["low_speed"] / r["sog_count"], 3) if r["sog_count"] else np.nan,
            "n_days_seen": len(r["dates"]), "has_sog": r["sog_count"] > 0,
            "is_real_ship": is_ship,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 候选规则: 真实船 + 有SOG + 航速合理 + 风场内停留较多 + 归属风场集中
    df["is_candidate"] = (
        df["is_real_ship"] & df["has_sog"]
        & (df["mean_sog_knots"] > 0) & (df["mean_sog_knots"] <= SOG_PLAUSIBLE_MAX)
        & (df["inside_polygon_points"] >= 500)
        & (df["n_production_farms"] <= 3)
    )
    df["candidate_score"] = np.where(
        df["is_candidate"],
        df["inside_polygon_points"] / df["n_production_farms"].clip(lower=1),
        0.0,
    ).round(1)
    df = df.sort_values(["is_candidate", "candidate_score"], ascending=False)
    return df


def build_visit_table(visit_stats, candidate_mmsi):
    rows = []
    for (mmsi, fid, dt), r in visit_stats.items():
        if candidate_mmsi and mmsi not in candidate_mmsi:
            continue
        dwell_min = np.nan
        if r["first_time"] is not None and r["last_time"] is not None:
            dwell_min = (r["last_time"] - r["first_time"]).total_seconds() / 60.0
        rows.append({
            "mmsi": mmsi, "farm_id": fid, "date": dt,
            "points": r["records"], "inside_polygon_points": r["inside"],
            "dwell_minutes_proxy": round(dwell_min, 1) if pd.notna(dwell_min) else np.nan,
            "mean_sog_knots": round(r["sog_sum"] / r["sog_count"], 3) if r["sog_count"] else np.nan,
            "first_time_utc": r["first_time"], "last_time_utc": r["last_time"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["mmsi", "farm_id", "date"])
    return df


def compute_multiplicity(mult_sample, farms_metric, radii_m):
    """在抽样近场点上, 统计每点落入多少个 Production 风场缓冲(各半径)。"""
    out = {}
    if not mult_sample:
        return {r // 1000: np.nan for r in radii_m}
    sample = gpd.GeoDataFrame(geometry=mult_sample, crs=EUROPE_METRIC_CRS)
    sample = sample.reset_index(drop=True)
    sample["pid"] = sample.index
    for r in sorted(radii_m):
        buf = farms_metric[["farm_id", "geometry"]].copy()
        buf["geometry"] = buf.geometry.buffer(r)
        j = gpd.sjoin(sample, buf, how="inner", predicate="within")
        per_pt = j.groupby("pid").size()
        # 没有落入任何缓冲的点计 0
        mult = per_pt.reindex(sample["pid"], fill_value=0)
        out[r // 1000] = round(float(mult.mean()), 3)
    return out


# ----------------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------------
def write_report(output_dir, file_summary_df, farm_df, vessel_df, visit_df,
                 windfarm_info, radii_m, multiplicity):
    lines = []
    lines.append("# AIS × 海上风场 数据可用性报告\n")

    lines.append("## 1. 日期核验 (文件名 vs 解析时间戳)\n")
    lines.append("aisdk 官方命名为 `aisdk-YYYY-MM-DD`, 内部时间戳为 `DD/MM/YYYY`。下表的"
                 "原始时间戳样本可直接核对解析是否正确:\n")
    for _, row in file_summary_df.iterrows():
        lines.append(f"- `{row['source_file']}` 原始时间戳样本: `{row['raw_timestamp_sample']}` "
                     f"→ 解析范围 {row['time_start_utc']} ~ {row['time_end_utc']}")
    lines.append("")

    lines.append("## 2. AIS 基础质量\n")
    tot = int(file_summary_df["rows_read"].sum())
    vc = int(file_summary_df["valid_coordinates"].sum())
    vm = int(file_summary_df["valid_mmsi"].sum())
    sc = int(file_summary_df["sog_count"].sum())
    lines.append(f"- 合计读取 {tot:,} 行; 有效坐标 {vc/tot:.2%}; 有效 MMSI {vm/tot:.2%}; "
                 f"含有效 SOG 的比例 {sc/tot:.2%}。\n")

    lines.append("## 3. 风场状态分布 (全部) 与本次筛选\n")
    lines.append(f"- 风场总数 {windfarm_info['farm_count']}; 命中目标关键词 "
                 f"{windfarm_info['status_keywords']} 的有 {windfarm_info['target_count']} 个 "
                 f"(本次空间归属仅用这些)。")
    for k, v in windfarm_info["status_counts"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")

    lines.append("## 4. 缓冲重叠倍数 (抽样, 每点平均落入风场数)\n")
    lines.append("用于判断服务区半径是否过宽; 数值越接近 1 越干净:\n")
    for km in sorted(multiplicity):
        lines.append(f"- {km} km: 平均 {multiplicity[km]} 个风场/点")
    lines.append("\n> 本版正式按风场/按船统计已改为【最近风场单一归属】, 不受上面重叠倍数影响; "
                 "该表仅用于选定半径。\n")

    lines.append("## 5. Production 风场 Top 10 (按多边形内点数)\n")
    if not farm_df.empty:
        top = farm_df.head(10)
        lines.append("| farm_id | name | inside_poly | ship_mmsi | mean_sog | low_speed_share |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(f"| {r['farm_id']} | {r['farm_name']} | {r['pts_inside_polygon']:,} | "
                         f"{r['unique_ship_mmsi']} | {r['mean_sog_knots']:.2f} | "
                         f"{r['low_speed_share']:.2%} |")
    lines.append("")

    lines.append("## 6. 候选运维船 Top 15 (预筛 + 打分)\n")
    if not vessel_df.empty:
        cand = vessel_df[vessel_df["is_candidate"]].head(15)
        lines.append(f"- 真实船舶共 {int(vessel_df['is_real_ship'].sum())} 个; "
                     f"通过候选规则 {int(vessel_df['is_candidate'].sum())} 个。\n")
        lines.append("| mmsi | flag | name | ship_type | inside_poly | n_farms | mean_sog | score |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for _, r in cand.iterrows():
            lines.append(f"| {r['mmsi']} | {r['flag']} | {r['name']} | {r['ship_type']} | "
                         f"{r['inside_polygon_points']:,} | {r['n_production_farms']} | "
                         f"{r['mean_sog_knots']} | {r['candidate_score']} |")
    lines.append("")

    lines.append("## 7. 重要限制\n")
    lines.append("- 仅 2 个相隔约 3 个月的单日切片, 只能做原型/流程验证, 不能估计预测误差分布或 DRCC 模糊集。")
    lines.append("- EMODnet 是风场边界, 不是单机风机坐标; 候选船需结合船名/船型/原始轨迹人工核验。")
    lines.append("- 候选打分是启发式, 不构成 '已确认运维船' 的结论。\n")

    (output_dir / "data_usability_report.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def configure_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(output_dir / "run_log.txt", encoding="utf-8")],
    )


def parse_args():
    p = argparse.ArgumentParser(description="AIS × 风场 数据探索 (精简修订版)")
    p.add_argument("--input-dir", type=Path, default=Path("./ais_filtered"))
    p.add_argument("--output-dir", type=Path, default=Path("./out"))
    p.add_argument("--geojson", type=str, default="./data/emodnet_windfarmspoly_all.geojson")
    p.add_argument("--ais-glob", type=str, default="aisdk-*.csv")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--radii", type=str, default="10,15,20,30", help="服务区半径(km), 逗号分隔")
    p.add_argument("--status-keywords", type=str, default="production,operational,operating,commissioned,active,in operation",
                   help="判定为目标风场的状态关键词")
    p.add_argument("--dayfirst", type=lambda s: s.lower() != "false", default=True,
                   help="时间戳按 DD/MM/YYYY 解析(aisdk 应为 True)")
    p.add_argument("--mult-sample-cap", type=int, default=300_000)
    p.add_argument("--trajectory-mmsi", type=str, default="",
                   help="逗号分隔的 MMSI 列表; 提供则额外导出 05_trajectory_<mmsi>.csv")
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(args.output_dir)
    out = args.output_dir

    radii_km = [int(x) for x in args.radii.split(",") if x.strip()]
    radii_m = [k * 1000 for k in radii_km]
    status_keywords = [w.strip().lower() for w in args.status_keywords.split(",") if w.strip()]
    trajectory_mmsi = {x.strip() for x in args.trajectory_mmsi.split(",") if x.strip()}

    # geojson: 若路径含分隔符(./data/...)直接用;否则在 input_dir 下找(向后兼容)
    gj = Path(args.geojson)
    geojson_path = gj if ("/" in args.geojson or "\\" in args.geojson) else args.input_dir / gj
    if not geojson_path.is_file():
        # 兜底:在 data/ 下再找一次
        alt = Path("./data") / gj.name
        if alt.is_file():
            geojson_path = alt
            logging.info("geojson 兜底到 %s", geojson_path)
        else:
            raise SystemExit(f"找不到风场文件: {geojson_path}\n"
                             f"请把 emodnet_windfarmspoly_all.geojson 放到 ./data/ 或用 --geojson 指定路径")
    farms_metric, windfarm_info = prepare_windfarms(geojson_path, status_keywords, out)
    farm_centroids = {r["farm_id"]: (r["centroid_lon"], r["centroid_lat"])
                      for _, r in farms_metric.iterrows()}

    ais_files = sorted(args.input_dir.glob(args.ais_glob))
    if not ais_files:
        raise SystemExit(f"未找到 AIS 文件: {args.input_dir / args.ais_glob}")
    logging.info("待处理 AIS 文件: %s", [f.name for f in ais_files])

    farm_stats = defaultdict(new_farm_rec)
    vessel_stats = defaultdict(new_vessel_rec)
    visit_stats = defaultdict(new_visit_rec)
    mult_sample = []
    trajectory_rows = defaultdict(list)
    file_summaries = []

    for f in ais_files:
        try:
            s = process_ais_file(f, farms_metric, radii_m, args.chunksize,
                                 farm_stats, vessel_stats, visit_stats,
                                 mult_sample, args.mult_sample_cap,
                                 trajectory_mmsi, trajectory_rows, args.dayfirst)
            file_summaries.append(s)
        except Exception:
            logging.exception("处理失败: %s", f.name)

    # ---- 01 file summary ----
    fs = pd.DataFrame(file_summaries)
    if not fs.empty:
        fs = fs.rename(columns={"time_start": "time_start_utc", "time_end": "time_end_utc"})
        for c, r in [("valid_coordinates_rate", "valid_coordinates"),
                     ("valid_mmsi_rate", "valid_mmsi"), ("sog_coverage_rate", "sog_count")]:
            fs[c] = fs[r] / fs["rows_read"]
        cols = ["source_file", "rows_read", "valid_coordinates", "valid_mmsi", "sog_count",
                "valid_coordinates_rate", "valid_mmsi_rate", "sog_coverage_rate",
                "unique_mmsi", "assigned_points", "inside_points", "mean_sog_knots",
                "raw_timestamp_sample", "time_start_utc", "time_end_utc"]
        fs = fs[[c for c in cols if c in fs.columns]]
    fs.to_csv(out / "01_file_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 02 farm production ----
    farm_df = build_farm_table(farm_stats)
    # 合并质心/海区
    meta = farms_metric[["farm_id", "centroid_lon", "centroid_lat", "sea_region", "area_km2"]]
    farm_df = farm_df.merge(meta, on="farm_id", how="left") if not farm_df.empty else farm_df
    farm_df.to_csv(out / "02_farm_production.csv", index=False, encoding="utf-8-sig")

    # ---- 03 vessel candidates ----
    vessel_df = build_vessel_table(vessel_stats, farm_centroids)
    vessel_df.to_csv(out / "03_vessel_candidates.csv", index=False, encoding="utf-8-sig")
    candidate_mmsi = set(vessel_df.loc[vessel_df.get("is_candidate", False) == True, "mmsi"]) \
        if not vessel_df.empty else set()

    # ---- 04 vessel-farm-date visits (仅候选船) ----
    visit_df = build_visit_table(visit_stats, candidate_mmsi)
    visit_df.to_csv(out / "04_vessel_farm_visits.csv", index=False, encoding="utf-8-sig")

    # ---- 重叠倍数 + 报告 ----
    multiplicity = compute_multiplicity(mult_sample, farms_metric, radii_m)
    write_report(out, fs, farm_df, vessel_df, visit_df, windfarm_info, radii_m, multiplicity)

    # ---- 可选轨迹导出 ----
    for mmsi, parts in trajectory_rows.items():
        if parts:
            traj = pd.concat(parts, ignore_index=True).sort_values("timestamp")
            traj.to_csv(out / f"05_trajectory_{mmsi}.csv", index=False, encoding="utf-8-sig")
            logging.info("导出轨迹: 05_trajectory_%s.csv (%s 点)", mmsi, len(traj))

    logging.info("完成。输出目录: %s", out)
    for name in ["01_file_summary.csv", "02_farm_production.csv",
                 "03_vessel_candidates.csv", "04_vessel_farm_visits.csv",
                 "data_usability_report.md"]:
        logging.info(" - %s", out / name)


if __name__ == "__main__":
    main()
