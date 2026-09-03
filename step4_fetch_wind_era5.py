#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_fetch_wind_era5.py — 用 CDS API 下载 ERA5(风 + 浪)并抽取到各候选风场的逐时序列。

需要:
  1) CDS 账户(新版 Climate Data Store): https://cds.climate.copernicus.eu/
     个人资料页拿 API key, 写入 ~/.cdsapirc :
         url: https://cds.climate.copernicus.eu/api
         key: <你的-API-KEY>
     并在 "ERA5 hourly data on single levels" 数据集页面首次点同意许可。
  2) pip install "cdsapi>=0.7" "xarray" "netcdf4"

变量(reanalysis-era5-single-levels, 逐时, ~0.25°):
  风: 10m_u/v_component_of_wind, 100m_u/v_component_of_wind
  浪: significant_height_of_combined_wind_waves_and_swell, mean_wave_period, mean_wave_direction
  ⚠ ERA5 浪在波罗的海/半封闭海(Nysted/Rødsand、Anholt 附近)偏弱; 仅作初筛。
     最终案例的浪建议改用 Copernicus Marine 区域产品(见文件末尾说明)。

用法:
  python step4_fetch_wind_era5.py --start 2025-06 --end 2025-06 \
     --area 57.0,10.6,54.2,12.25 \
     --farms 'Anholt:11.2177,56.6002;Nysted:11.7148,54.5493;Rodsand_II:11.5491,54.5547' \
     --out-dir ./weather --wind-limit 12 --hs-limit 1.5

输出:
  weather/era5_YYYY-MM.nc         每月原始 NetCDF
  weather/weather_<farm>.csv      逐时: wind10/wind100/wind_dir/Hs/Tm/wave_dir + operable 标记
  weather/_operability_summary.csv 各风场可作业小时占比(按阈值)
"""

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# === CDS credentials ===
# No credentials are stored in this repository. cdsapi reads ~/.cdsapirc
# by default; alternatively export CDS_API_KEY before running.
CDS_URL = "https://cds.climate.copernicus.eu/api"


def _cds_client():
    import cdsapi
    key = os.environ.get("CDS_API_KEY", "")
    if key:
        return cdsapi.Client(url=CDS_URL, key=key)
    return cdsapi.Client()  # falls back to ~/.cdsapirc

VARS = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "100m_u_component_of_wind", "100m_v_component_of_wind",
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period", "mean_wave_direction",
]


def months(start, end):
    s = pd.Period(start, "M")
    e = pd.Period(end, "M")
    out = []
    while s <= e:
        out.append((s.year, s.month))
        s += 1
    return out


def download_month(c, year, month, area, out_path: Path):
    if out_path.exists():
        logging.info("已存在, 跳过下载: %s", out_path.name)
        return
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = [f"{h:02d}:00" for h in range(24)]
    req = {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": hours,
        "area": area,           # [N, W, S, E]
        "format": "netcdf",     # 新版 CDS 若报错, 改成 "data_format": "netcdf"
    }
    logging.info("CDS 请求 ERA5 %04d-%02d ...", year, month)
    c.retrieve("reanalysis-era5-single-levels", req, str(out_path))
    logging.info("已下载 %s", out_path.name)


def _pick(ds, names):
    for n in names:
        if n in ds:
            return ds[n]
    raise KeyError(f"变量缺失, 期望其一: {names}; 实际: {list(ds.data_vars)}")


def _resolve_nc_files(out_dir: Path):
    """把 era5_*.nc 里可能是 ZIP 的解开(新版 CDS 常把风/浪拆成两个 nc 再打包),
    返回真实可读的 nc 文件列表。"""
    import zipfile
    out_dir.mkdir(parents=True, exist_ok=True)
    reals = []
    for p in sorted(out_dir.glob("era5_*.nc")):
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                inner = [m for m in z.namelist() if m.lower().endswith(".nc")]
                for m in inner:
                    tgt = out_dir / f"{p.stem}__{Path(m).name}"
                    if not tgt.exists():
                        with z.open(m) as s, open(tgt, "wb") as o:
                            o.write(s.read())
                    reals.append(tgt)
            logging.info("%s 是 ZIP, 解出 %s 个 nc -> %s", p.name, len(inner), out_dir)
            # 把 zip 改名, 避免下次重复解
            bak = p.with_suffix(".nc.zip_bak")
            if not bak.exists():
                p.rename(bak)
        else:
            reals.append(p)
    return reals


def extract(out_dir: Path, farms, wind_limit, hs_limit):
    import xarray as xr
    import tempfile, shutil
    files = _resolve_nc_files(out_dir)
    if not files:
        raise SystemExit("未找到 era5_*.nc")

    # netCDF4 的 C 库在 Windows 上无法打开含非 ASCII(如中文)的路径,
    # 会误报 FileNotFoundError。复制到纯英文系统临时目录再读, 规避之。
    tmpdir = Path(tempfile.mkdtemp(prefix="era5_"))
    logging.info("复制 nc 到临时目录(规避中文路径): %s", tmpdir)

    # 读入所有 nc 到内存。可能是: (a) 多个【月份】文件(各覆盖不同时间段) 需沿 time 拼接;
    # (b) 同一时段的【风】与【浪】两个流(变量不同) 需按坐标合并变量。
    # 旧版用 xr.merge(join="outer") 跨月会错位 -> 4-6 月风变 NaN(只剩首月)。
    # 改为: 先按变量集合分组, 同组(同变量)沿 time 拼接(concat), 不同组再 merge 变量。
    dsets = []
    try:
        for i, f in enumerate(files):
            dst = tmpdir / f"part{i}.nc"
            shutil.copyfile(f, dst)
            with xr.open_dataset(dst) as d0:
                d = d0.load()
            # 归一时间坐标名; 丢掉 expver/number 等单值维(ERA5 vs ERA5T 会引入 expver)
            if "valid_time" in d.coords and "time" not in d.coords:
                d = d.rename({"valid_time": "time"})
            for extra in ("expver", "number"):
                if extra in d.coords:
                    d = d.drop_vars(extra, errors="ignore")
                if extra in d.dims:
                    d = d.isel({extra: 0}, drop=True)
            d = d.squeeze(drop=True)
            dsets.append(d)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 按"变量集合"把数据集分组: 同变量集合的(不同月份)沿 time 拼接; 再把不同变量集合 merge。
    from collections import defaultdict
    groups = defaultdict(list)
    for d in dsets:
        key = tuple(sorted(d.data_vars))
        groups[key].append(d)
    combined = []
    for key, members in groups.items():
        if len(members) > 1:
            cat = xr.concat(members, dim="time")
            cat = cat.sortby("time")
            # 去重(月份边界可能重叠的整点)
            _, idx = np.unique(cat["time"].values, return_index=True)
            cat = cat.isel(time=np.sort(idx))
        else:
            cat = members[0]
        combined.append(cat)
    ds = xr.merge(combined, compat="override", join="outer") if len(combined) > 1 else combined[0]

    # 兼容坐标命名(新旧 CDS): latitude/longitude, time/valid_time
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    op_rows = []
    for name, lon, lat in farms:
        # ★双线性插值(2026-06-27): 用风机/风场周围 4 个网格点插值, 比最近邻平滑、更适合写论文。
        #   weather_i(t)=Σ_{m=1..4} α_m(q_i)·weather_m(t); xarray .interp(method="linear")
        #   在规则网格上即此双线性。对非单调纬度坐标先排序, 失败回退最近邻。
        try:
            dsx = ds.sortby(lat_name).sortby(lon_name)
            sub = dsx.interp({lat_name: lat, lon_name: lon}, method="linear")
        except Exception as exc:
            logging.warning("%s: 双线性插值失败(%s), 回退最近邻。", name, exc)
            sub = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        sub = sub.squeeze(drop=True)
        df = pd.DataFrame()
        # 时间
        tname = "time" if "time" in sub.coords else ("valid_time" if "valid_time" in sub.coords else None)
        df["time"] = pd.to_datetime(sub[tname].values) if tname else range(sub.sizes.get("time", 0))

        u10 = _pick(sub, ["u10", "10u"]).values
        v10 = _pick(sub, ["v10", "10v"]).values
        u100 = _pick(sub, ["u100", "100u"]).values
        v100 = _pick(sub, ["v100", "100v"]).values
        df["wind10_ms"] = np.hypot(u10, v10)
        df["wind100_ms"] = np.hypot(u100, v100)
        # 气象"来向"(0=北,顺时针)
        df["wind_dir_from_deg"] = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0

        # 浪: 优先用 CMEMS waves_<farm>.csv(波罗的海 ERA5 的 swh 多为 NaN, 半封闭海被屏蔽)。
        #     按小时时间戳对齐合并; 无 CMEMS 文件才回退 ERA5 的 swh/mwp/mwd。
        df["time"] = pd.to_datetime(df["time"]).dt.floor("h")  # 对齐到整点, 避免错位
        wave_csv = out_dir / f"waves_{name}.csv"
        used_cmems = False
        if wave_csv.is_file():
            try:
                wv = pd.read_csv(wave_csv)
                wv["time"] = pd.to_datetime(wv["time"]).dt.floor("h")
                wv = wv[["time", "Hs_m", "wave_Tm_s", "wave_dir_deg"]].drop_duplicates("time")
                df = df.merge(wv, on="time", how="left")
                used_cmems = True
                logging.info("%s: 浪用 CMEMS %s(时间戳对齐合并)", name, wave_csv.name)
            except Exception as exc:
                logging.warning("%s: 读 CMEMS 浪失败(%s), 回退 ERA5 swh。", name, exc)
        if not used_cmems:
            df["Hs_m"] = _pick(sub, ["swh"]).values
            try:
                df["wave_Tm_s"] = _pick(sub, ["mwp"]).values
                df["wave_dir_deg"] = _pick(sub, ["mwd"]).values
            except KeyError:
                df["wave_Tm_s"] = np.nan
                df["wave_dir_deg"] = np.nan
            if df["Hs_m"].isna().all():
                logging.warning("%s: ERA5 浪全 NaN(半封闭海); 建议先跑 step5_fetch_wave_cmems.py "
                                "生成 waves_%s.csv 再 --extract-only。", name, name)

        # operable: 风与浪都需有效且在门限内(NaN 视为不可判→False, 但单独记录浪缺测)
        wind_ok = df["wind10_ms"] < wind_limit
        hs_ok = df["Hs_m"] < hs_limit
        df["operable"] = (wind_ok & hs_ok).fillna(False)
        out = out_dir / f"weather_{name}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        n_hs_nan = int(df["Hs_m"].isna().sum())
        share = float(df["operable"].mean()) if len(df) else float("nan")
        op_rows.append({"farm": name, "hours": len(df),
                        "operable_share": round(share, 4),
                        "hs_nan_hours": n_hs_nan,
                        "wave_source": "CMEMS" if used_cmems else "ERA5",
                        "wind_limit_ms": wind_limit, "hs_limit_m": hs_limit})
        logging.info("%s: %s 小时, 浪源 %s, Hs缺测 %d, 可作业 %.1f%% -> %s",
                     name, len(df), "CMEMS" if used_cmems else "ERA5", n_hs_nan, share * 100, out.name)

    pd.DataFrame(op_rows).to_csv(out_dir / "_operability_summary.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="ERA5 风+浪 下载与风场点抽取")
    ap.add_argument("--start", default="2025-06", help="起始月 YYYY-MM")
    ap.add_argument("--end", default="2025-06", help="结束月 YYYY-MM")
    ap.add_argument("--area", default="57.0,10.6,54.2,12.25", help="N,W,S,E (覆盖所有风场的并集框)")
    ap.add_argument("--farms",
                    default="Anholt:11.2177,56.6002;Nysted:11.7148,54.5493;Rodsand_II:11.5491,54.5547",
                    help='"名:lon,lat;名:lon,lat;..."')
    ap.add_argument("--out-dir", type=Path, default=Path("./weather"))
    ap.add_argument("--wind-limit", type=float, default=12.0, help="可作业风速阈值 m/s(10m)")
    ap.add_argument("--hs-limit", type=float, default=1.5, help="可作业有效波高阈值 m")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    area = [float(x) for x in args.area.split(",")]
    farms = []
    for part in args.farms.split(";"):
        if not part.strip():
            continue
        nm, coord = part.split(":")
        lon, lat = [float(x) for x in coord.split(",")]
        farms.append((nm.strip(), lon, lat))

    if not args.extract_only:
        import cdsapi
        c = _cds_client()
        for y, m in months(args.start, args.end):
            download_month(c, y, m, area, args.out_dir / f"era5_{y:04d}-{m:02d}.nc")

    if not args.download_only:
        extract(args.out_dir, farms, args.wind_limit, args.hs_limit)
    logging.info("完成。输出目录: %s", args.out_dir)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 可选(最终案例推荐): Copernicus Marine 区域浪, 比 ERA5 在波罗的海/Kattegat 更准
# ---------------------------------------------------------------------------
# 1) pip install copernicusmarine ; copernicusmarine login (输入 CMEMS 账户)
# 2) 在 https://data.marine.copernicus.eu/products 查准确 dataset_id(下面是常见项, 请核对):
#    - 波罗的海(Nysted/Rødsand): 产品 BALTICSEA_ANALYSISFORECAST_WAV_003_010
#    - 北海/Kattegat(Anholt):    产品 NWSHELF_ANALYSISFORECAST_WAV_004_014
#    变量: VHM0(有效波高 Hs), VTM02 或 VTPK(周期), VMDR(浪向)
# 3) 示例(把 <dataset_id> 换成核对后的 id):
#    copernicusmarine subset \
#       --dataset-id <dataset_id> \
#       --variable VHM0 --variable VTM02 --variable VMDR \
#       --start-datetime 2025-06-01T00:00:00 --end-datetime 2025-06-30T23:00:00 \
#       --minimum-longitude 11.0 --maximum-longitude 12.3 \
#       --minimum-latitude 54.2 --maximum-latitude 57.0 \
#       --output-filename cmems_waves_2025-06.nc
#    然后用本脚本 extract() 同样的 xarray 最近邻方式抽取(变量名改为 VHM0/VTM02/VMDR)。
