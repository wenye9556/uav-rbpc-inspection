#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_fetch_wave_cmems.py — 用 Copernicus Marine 工具箱下载区域海浪并抽取到风场点。

为什么: ERA5 浪在 Anholt/Nysted(半封闭海)为 NaN, 需用 CMEMS 区域浪。

需要:
  pip install copernicusmarine xarray netcdf4
  copernicusmarine login        # 输入 CMEMS 账户(https://marine.copernicus.eu/ 注册)

数据集(★请在 CMEMS 产品目录核对 dataset-id, 下面是常见默认, 可能需改):
  - 波罗的海(Nysted/Rødsand): 产品 BALTICSEA_ANALYSISFORECAST_WAV_003_010
  - 北海/Kattegat(Anholt):    产品 NWSHELF_ANALYSISFORECAST_WAV_004_014
  变量: VHM0(有效波高 Hs), VTM02(平均周期), VMDR(浪向)

用法:
  python step5_fetch_wave_cmems.py --start 2025-03-01 --end 2025-06-30
  (默认: Anholt 用 NWShelf, Nysted/Rødsand 用 Baltic; 各自下载+抽取)

输出:
  data/weather/waves_<farm>.csv   逐时 Hs / 周期 / 浪向

注: Windows 中文路径下 netCDF4 读取会失败, 本脚本已复制到 ASCII 临时目录再读(同 ERA5 脚本)。
"""

from __future__ import annotations
import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# farm -> (dataset_id, lon, lat)。dataset_id 大小写敏感,以本机目录查询结果为准。
# [2026-06-25 作者本机目录实测] BALTICSEA_ANALYSISFORECAST_WAV_003_010 的小时瞬时网格:
#   cmems_mod_bal_wav_anfc_PT1H-i (大写 H, 小写 i)
#   ← copernicusmarine 严格区分大小写；变量名必须保持官方大小写。
#   查询命令: python -c "import copernicusmarine; r=copernicusmarine.describe(
#             contains=['BALTICSEA_ANALYSISFORECAST_WAV_003_010']);
#             [print(ds.dataset_id) for p in r.products for ds in p.datasets]"
#   变量 VHM0/VTM02/VMDR 有效; 如需谱峰周期 Tp 改用变量 VTPK。
DATASET_BALTIC = "cmems_mod_bal_wav_anfc_PT1H-i"   # 大写 H — 本机目录实测 2026-06-25
DATASET_NWSHELF = "cmems_mod_nws_wav_anfc_0.05deg_PT1H-i"  # NWShelf [2025-11 已退役, 见下]
# [2026-06-25] Case B 只需 Baltic(Nysted/Rødsand)。Anholt(NWShelf)浪产品于 2025-11
#   退役并改分辨率(1/20°→1/36°), 旧 id 已 DatasetNotFound。若需 Anholt 对照:
#   用 `copernicusmarine describe --contains NWSHELF_ANALYSISFORECAST_WAV_004_014`
#   查当前 id(含 0.027deg), 填回下面并取消注释。
DEFAULT_FARMS = {
    # "Anholt": (DATASET_NWSHELF, 11.2177, 56.6002),
    "Nysted":     (DATASET_BALTIC, 11.7148, 54.5493),
    "Rodsand_II": (DATASET_BALTIC, 11.5491, 54.5547),
}
WAVE_VARS = ["VHM0", "VTM02", "VMDR"]


def download(dataset_id, lon, lat, start, end, out_nc: Path, pad=0.3):
    import copernicusmarine
    if out_nc.exists():
        logging.info("已存在, 跳过下载: %s", out_nc.name)
        return
    logging.info("CMEMS 下载 %s 附近 (%s) ...", dataset_id, out_nc.name)
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=WAVE_VARS,
        minimum_longitude=lon - pad, maximum_longitude=lon + pad,
        minimum_latitude=lat - pad, maximum_latitude=lat + pad,
        start_datetime=f"{start}T00:00:00", end_datetime=f"{end}T23:00:00",
        output_filename=str(out_nc.name), output_directory=str(out_nc.parent),
    )


def extract_point(nc_path: Path, lon, lat):
    import xarray as xr
    # 复制到 ASCII 临时目录, 规避中文路径
    tmpdir = Path(tempfile.mkdtemp(prefix="cmems_"))
    try:
        dst = tmpdir / "w.nc"
        shutil.copyfile(nc_path, dst)
        with xr.open_dataset(dst) as d0:
            ds = d0.squeeze(drop=True).load()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    # ★双线性插值(2026-06-27): 周围 4 个网格点插值, 比最近邻平滑。.interp(linear) 即双线性; 失败回退最近邻。
    try:
        dsx = ds.sortby(lat_name).sortby(lon_name)
        sub = dsx.interp({lat_name: lat, lon_name: lon}, method="linear").squeeze(drop=True)
    except Exception as exc:
        logging.warning("双线性插值失败(%s), 回退最近邻。", exc)
        sub = ds.sel({lat_name: lat, lon_name: lon}, method="nearest").squeeze(drop=True)
    tname = "time" if "time" in sub.coords else ("valid_time" if "valid_time" in sub.coords else None)

    df = pd.DataFrame()
    df["time"] = pd.to_datetime(sub[tname].values) if tname else range(sub.sizes.get("time", 0))
    df["Hs_m"] = sub["VHM0"].values if "VHM0" in sub else np.nan
    df["wave_Tm_s"] = sub["VTM02"].values if "VTM02" in sub else np.nan
    df["wave_dir_deg"] = sub["VMDR"].values if "VMDR" in sub else np.nan
    return df




# =============================================================================
# 附: 逐风机双线性天气插值(原 util_weather_interp.py, 更新 并入; model.md §15)
#   产出 weather_per_turbine_*.csv, 由 step13.attach_per_turbine_weather 消费。
#   用法: python step5_fetch_wave_cmems.py --per-turbine --nc weather/era5_*.nc \
#         --turbines data/turbines_Rodsand_II_clean.csv --vars u10,v10 \
#         --out weather/weather_per_turbine_Rodsand_II.csv
#   自检: python step5_fetch_wave_cmems.py --per-turbine   (合成网格, 无需 .nc)
# =============================================================================
log = logging.getLogger("wxinterp")

# =============================================================================
# 1. 双线性权重(显式; model.md §15 的 α_m)
# =============================================================================
def _bracket(coord: np.ndarray, q: float) -> tuple[int, int, float]:
    """在【升序】坐标 coord 上, 返回包住 q 的两个下标 (i0,i1) 与归一化位置 t∈[0,1]。
    q 越界则夹到端点(t=0 或 1)。"""
    n = len(coord)
    if q <= coord[0]:
        return 0, min(1, n - 1), 0.0
    if q >= coord[-1]:
        return max(0, n - 2), n - 1, 1.0
    i1 = int(np.searchsorted(coord, q))
    i0 = i1 - 1
    t = (q - coord[i0]) / (coord[i1] - coord[i0] + 1e-12)
    return i0, i1, float(t)


def bilinear_weights(lat: float, lon: float,
                     lats: np.ndarray, lons: np.ndarray) -> tuple[list, np.ndarray]:
    """风机 (lat,lon) 在网格 (lats,lons, 均升序) 上的双线性插值: 返回 4 个角点 (ilat,ilon)
    与权重 α(4,), Σα=1。weather_i = Σ_m α_m · weather[角点_m]。

    角点顺序: (j0,i0),(j0,i1),(j1,i0),(j1,i1); j=纬度向, i=经度向。
    权重: α = [(1-tlat)(1-tlon), (1-tlat)tlon, tlat(1-tlon), tlat·tlon]。
    """
    j0, j1, tlat = _bracket(lats, lat)
    i0, i1, tlon = _bracket(lons, lon)
    corners = [(j0, i0), (j0, i1), (j1, i0), (j1, i1)]
    alpha = np.array([(1 - tlat) * (1 - tlon), (1 - tlat) * tlon,
                      tlat * (1 - tlon), tlat * tlon])
    return corners, alpha


def interp_point(field2d: np.ndarray, lat: float, lon: float,
                 lats: np.ndarray, lons: np.ndarray) -> float:
    """对单个二维场 field2d[j,i](纬×经)在 (lat,lon) 双线性插值。"""
    corners, alpha = bilinear_weights(lat, lon, lats, lons)
    return float(sum(a * field2d[j, i] for a, (j, i) in zip(alpha, corners)))


def interp_series(field3d: np.ndarray, lat: float, lon: float,
                  lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """对时间序列场 field3d[t,j,i] 在 (lat,lon) 双线性插值, 返回长 T 的序列。"""
    corners, alpha = bilinear_weights(lat, lon, lats, lons)
    out = np.zeros(field3d.shape[0])
    for a, (j, i) in zip(alpha, corners):
        out += a * field3d[:, j, i]
    return out


# =============================================================================
# 2. 从网格 .nc 提取逐风机天气(需 xarray; 真实使用)
# =============================================================================
def _open_nc(nc_pattern):
    """稳健打开 .nc: 支持通配符(PowerShell 不自动展开)、多文件、引擎兜底。返回 xarray Dataset。"""
    import glob as _glob
    s = str(nc_pattern)
    files = sorted(_glob.glob(s)) if any(ch in s for ch in "*?[") else ([s] if Path(s).is_file() else [])
    if not files:
        # 帮助定位: 列出该目录下实际有的 .nc
        d = Path(s).parent if Path(s).parent.as_posix() else Path(".")
        avail = sorted(p.name for p in d.glob("*.nc")) if d.is_dir() else []
        raise FileNotFoundError(
            f"未找到匹配 '{s}' 的 .nc 文件。该目录现有 .nc: {avail or '无'}。\n"
            f"  提示: PowerShell 不展开通配符, 请直接给确切文件名(或用引号), 例如 --nc weather/具体文件.nc;\n"
            f"  多文件可逐个空格列出或用确切共同前缀。")
    import xarray as xr
    engines = [None, "netcdf4", "h5netcdf", "scipy", "cfgrib"]
    last = None
    for eng in engines:
        try:
            if len(files) == 1:
                return xr.open_dataset(files[0]) if eng is None else xr.open_dataset(files[0], engine=eng)
            kw = {} if eng is None else {"engine": eng}
            return xr.open_mfdataset(files, combine="by_coords", **kw)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"打开 .nc 失败(试过引擎 {engines[1:]}): {last}\n"
                       f"  可能缺 netCDF 引擎, 装一个: pip install netcdf4  (或 h5netcdf)。")


def _pick_var(ds, name):
    """变量名别名(新旧 CDS): u10↔10u, v10↔10v, u100↔100u, v100↔100v 等。"""
    if name in ds:
        return name
    alias = {"u10": "10u", "v10": "10v", "u100": "100u", "v100": "100v",
             "10u": "u10", "10v": "v10", "100u": "u100", "100v": "v100"}
    a = alias.get(name)
    return a if (a and a in ds) else None


def extract_per_turbine(nc_path: Path, turbines_csv: Path, varnames: list[str],
                        out_csv: Path) -> pd.DataFrame:
    """对每台风机双线性插值各变量, 写长表 (turbine_id, time, var1, var2, ...)。
    稳健处理: 通配符/多文件(_open_nc)、变量名别名(_pick_var)、坐标与时间名兼容。"""
    ds = _open_nc(nc_path)
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise KeyError(f"未识别经纬度坐标, 现有坐标: {list(ds.coords)}")
    ds = ds.sortby(lat_name).sortby(lon_name)
    lats = ds[lat_name].values; lons = ds[lon_name].values
    tname = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else None)
    times = pd.to_datetime(ds[tname].values) if tname else pd.RangeIndex(ds.sizes.get("time", 0))

    # 解析变量名(含别名), 缺失给出提示
    resolved = {}
    for v in varnames:
        rv = _pick_var(ds, v)
        if rv is None:
            log.warning("变量 %s 不在数据中(现有: %s), 跳过", v, list(ds.data_vars))
        else:
            resolved[v] = rv

    turb = pd.read_csv(turbines_csv)
    rows = []
    for _, r in turb.iterrows():
        tid = r.get("turbine_id", r.name)
        lat = float(r["lat"]); lon = float(r["lon"])
        series = {}
        for outname, rv in resolved.items():
            f3 = ds[rv]
            dims = [tname, lat_name, lon_name] if tname else [lat_name, lon_name]
            f3 = f3.transpose(*[d for d in dims if d in f3.dims]).values
            f3 = np.asarray(f3)
            if f3.ndim == 2:    # 无时间维
                f3 = f3[None, ...]
            series[outname] = interp_series(f3, lat, lon, lats, lons)
        nT = len(times) if tname else 1
        for k in range(nT):
            t = times[k] if tname else 0
            rec = dict(turbine_id=tid, time=t)
            for outname, arr in series.items():
                rec[outname] = float(arr[k])
            rows.append(rec)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info("逐风机双线性天气 %d 行(%d 台 × %d 时刻)→ %s",
             len(df), len(turb), len(times) if tname else 1, out_csv)
    return df


# =============================================================================
# 3. 自检(合成网格; 不需 .nc)
# =============================================================================
def _selftest():
    print("\n================ util_weather_interp.py 自检 ================")
    lats = np.array([54.0, 54.5, 55.0])      # 升序
    lons = np.array([11.0, 11.5, 12.0])
    # 线性场 f(lat,lon)=3·lat+2·lon, 双线性插值应【精确】还原线性场
    field = np.array([[3 * la + 2 * lo for lo in lons] for la in lats])

    print("\n--- (a) 权重性质 ---")
    for (la, lo, desc) in [(54.5, 11.5, "正好落在网格点"), (54.25, 11.25, "单元中心偏左下"),
                           (53.0, 10.0, "越界(夹到角)"), (54.6, 11.9, "近上边界")]:
        corners, alpha = bilinear_weights(la, lo, lats, lons)
        print(f"  ({la},{lo}) {desc}: Σα={alpha.sum():.4f} 非负={bool((alpha>=-1e-9).all())} "
              f"α={np.round(alpha,3)}")

    print("\n--- (b) 双线性精度(线性场应精确还原)---")
    ok = True
    for (la, lo) in [(54.2, 11.3), (54.7, 11.8), (54.5, 11.5), (54.9, 11.1)]:
        got = interp_point(field, la, lo, lats, lons)
        exact = 3 * la + 2 * lo
        err = abs(got - exact)
        ok = ok and err < 1e-9
        print(f"  ({la},{lo}): 插值={got:.4f} 解析={exact:.4f} 误差={err:.2e}")
    print(f"\n  线性场双线性还原: {'✓精确' if ok else '✗有误差'}")

    print("\n--- (c) 时间序列插值 ---")
    T = 5
    f3 = np.stack([field + k for k in range(T)])   # 每步整体 +k
    s = interp_series(f3, 54.5, 11.5, lats, lons)  # 网格点 (54.5,11.5)=3*54.5+2*11.5=186.5
    print(f"  点(54.5,11.5)序列={np.round(s,2)} (应为 186.5,187.5,...,190.5)")

    print("\n自检完成。双线性权重(Σα=1、非负、线性场精确)与序列插值已验证。")
    print("公式: weather_i(t)=Σ_{m=1..4} α_m(q_i)·weather_m(t); step4/step5 已用 .interp(linear) 实现。")
    print("真实逐风机天气: python util_weather_interp.py --nc ... --turbines ... --vars u10,v10 --out ...")


def main():
    ap = argparse.ArgumentParser(description="CMEMS 区域浪 下载与风场点抽取(+逐风机插值 --per-turbine)")
    ap.add_argument("--per-turbine", action="store_true",
                    help="逐风机双线性插值模式(原 util_weather_interp): 配 --nc/--turbines/--vars/--out; 缺省跑合成自检")
    ap.add_argument("--nc", type=Path, default=None,
                    help="[per-turbine] 网格天气 .nc(支持通配符/多文件)")
    ap.add_argument("--turbines", type=Path, default=None, help="[per-turbine] turbines_*_clean.csv")
    ap.add_argument("--vars", type=str, default="u10,v10", help="[per-turbine] 逗号分隔变量名")
    ap.add_argument("--out", type=Path, default=Path("weather/weather_per_turbine.csv"),
                    help="[per-turbine] 输出 CSV")
    ap.add_argument("--start", default="2025-03-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default="2025-06-30", help="YYYY-MM-DD")
    ap.add_argument("--out-dir", type=Path, default=Path("./data/weather"))
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.per_turbine:                       # 更新: 逐风机插值分支(原 util_weather_interp.main)
        if args.nc and args.turbines:
            extract_per_turbine(args.nc, args.turbines,
                                [v.strip() for v in args.vars.split(",")], args.out)
        else:
            _selftest()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for farm, (ds_id, lon, lat) in DEFAULT_FARMS.items():
        nc = args.out_dir / f"cmems_{farm}.nc"
        if not args.extract_only:
            try:
                download(ds_id, lon, lat, args.start, args.end, nc)
            except Exception:
                logging.exception("下载失败 %s (检查 dataset-id 与 copernicusmarine 登录)", farm)
                continue
        if not args.download_only:
            try:
                df = extract_point(nc, lon, lat)
                out = args.out_dir / f"waves_{farm}.csv"
                df.to_csv(out, index=False, encoding="utf-8-sig")
                logging.info("%s: %s 小时, Hs 均值 %.2f m -> %s",
                             farm, len(df), float(np.nanmean(df["Hs_m"])), out.name)
            except Exception:
                logging.exception("抽取失败 %s", farm)
    logging.info("完成。输出目录: %s", args.out_dir)


if __name__ == "__main__":
    main()
