#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_fetch_ais.py — 下载（可选）并按 BBOX / MMSI 过滤丹麦 aisdk 日文件。

数据源：丹麦海事局开放 AIS https://web.ais.dk/aisdata/
  - 按天发布，命名 aisdk-YYYY-MM-DD.(zip|csv)；ZIP 内通常含同名 CSV。
  - 单日原始文件可能达到数 GB。本脚本读取 ZIP 内的 CSV 流，不把原始 CSV
    解压到磁盘；每次只在内存中保留一个 pandas chunk。

两种用法：
  A)【推荐】日 ZIP 已下载到 --raw-dir，仅过滤：
     python step1_fetch_ais.py --start 2025-03-01 --end 2025-06-30 \
        --raw-dir ./data --out-dir ./ais_filtered --no-download \
        --keep-scope mmsi_only --remove-legacy-extracted \
        --bbox 'Anholt:57.0,10.618,56.2,11.818' \
        --bbox 'NysRod:54.902,11.032,54.202,12.232'

     对当前 P1 案例筛选，建议使用 --keep-scope mmsi_only：
     只保存目标 CTV 的逐点轨迹和每日汇总，避免把 BBOX 内全部船舶写入磁盘。

  B) 让脚本顺带下载（best-effort）：去掉 --no-download。
     下载到的 ZIP 保留在 --raw-dir；脚本仍不生成解压后的原始 CSV。

输出：
  <out-dir>/aisdk-YYYY-MM-DD.csv        每日精简切片（summary_only 时不写）
  <out-dir>/_vessel_day_summary.csv     本次运行的 (mmsi, date, region) 点数/航速汇总

BBOX 格式："可选名:N,W,S,E"（纬北、经西、纬南、经东；与 ERA5 area 一致）。
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import ssl
import sys
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import BinaryIO, Iterator, Literal, NamedTuple
from urllib.parse import urlparse

import pandas as pd

# 写死的默认值（零参运行即用；命令行可覆盖）
DEFAULT_BBOX = [
    "Anholt:57.0,10.618,56.2,11.818",
    "NysRod:54.902,11.032,54.202,12.232",
]
DEFAULT_MMSI = "219018044,219016873,219018788,219028973,219019936,232046091"

KEEP_ROLES = {  # role -> 候选列名（小写规整后匹配）
    "timestamp": ["timestamp", "time", "datetime", "basedatetime"],
    "mmsi": ["mmsi"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "lng", "x"],
    "sog": ["sog", "speedoverground", "speed"],
    "cog": ["cog", "courseoverground", "course"],
    "heading": ["heading", "trueheading"],
    "nav_status": ["navigationalstatus", "navigationstatus", "navstatus"],
    "name": ["name", "shipname"],
    "ship_type": ["shiptype", "vesseltype"],
    "type_of_mobile": ["typeofmobile"],
    "length": ["length"],
    "width": ["width"],
}


class DaySource(NamedTuple):
    """一个日 CSV 的来源；ZIP 只记录 member，不落盘解压。"""

    kind: Literal["csv", "zip"]
    path: Path
    member: str | None = None


def norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(columns, candidates):
    m = {norm(c): c for c in columns}
    for cand in candidates:
        if norm(cand) in m:
            return m[norm(cand)]
    for cand in candidates:
        cn = norm(cand)
        for k, orig in m.items():
            if len(cn) >= 4 and (cn in k or k in cn):
                return orig
    return None


def daterange(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_bbox(spec: str):
    """'name:N,W,S,E' 或 'N,W,S,E' -> (name, (N,W,S,E))。"""
    name = None
    if ":" in spec:
        name, spec = spec.split(":", 1)
    n, w, s, e = [float(x) for x in spec.split(",")]
    if n < s or e < w:
        raise ValueError(f"BBOX 非法（需 N>=S 且 E>=W）：{spec}")
    return (name or "box", (n, w, s, e))


def in_bbox(lat: pd.Series, lon: pd.Series, box):
    n, w, s, e = box
    return (lat <= n) & (lat >= s) & (lon >= w) & (lon <= e)


def _date_from_name(name: str) -> date | None:
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _csv_members(zp: Path) -> list[str]:
    with zipfile.ZipFile(zp) as z:
        return [m for m in z.namelist() if not m.endswith("/") and m.lower().endswith(".csv")]


def find_zip_member(zp: Path, target_date: date | None = None) -> str | None:
    """找到 ZIP 中与日期匹配的 CSV；日 ZIP 找不到精确名称时允许唯一 CSV 回退。"""
    try:
        members = _csv_members(zp)
    except (OSError, zipfile.BadZipFile) as exc:
        logging.error("无法读取 ZIP %s (%s)", zp.name, exc)
        return None
    if not members:
        logging.error("%s 内未找到 CSV。", zp.name)
        return None

    if target_date is not None:
        expected = f"aisdk-{target_date:%Y-%m-%d}.csv".lower()
        for member in members:
            if Path(member).name.lower() == expected:
                return member
        for member in members:
            if _date_from_name(Path(member).name) == target_date:
                return member
    if len(members) == 1:
        return members[0]

    logging.error("%s 内无法唯一定位 %s 对应的 CSV。候选数=%s",
                  zp.name, target_date, len(members))
    return None


@contextmanager
def open_source(source: DaySource) -> Iterator[BinaryIO]:
    """打开原始 CSV 或 ZIP 内部 CSV。ZIP 文件从不解压到磁盘。"""
    if source.kind == "csv":
        with source.path.open("rb") as f:
            yield f
        return

    if source.member is None:
        raise ValueError(f"ZIP 来源缺少 member：{source.path}")
    with zipfile.ZipFile(source.path) as z:
        with z.open(source.member) as f:
            yield f


def sniff_source(source: DaySource) -> tuple[str, str]:
    """从来源前 100 KB 判断编码与分隔符；随后会重新打开来源正式读取。"""
    with open_source(source) as f:
        head = f.read(100_000)
    for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
        try:
            txt = head.decode(enc)
        except UnicodeDecodeError:
            continue
        try:
            sep = csv.Sniffer().sniff(txt, delimiters=[",", ";", "\t"]).delimiter
        except csv.Error:
            sep = max([",", ";", "\t"], key=txt.count)
        return enc, sep
    return "utf-8-sig", ","


AISDK_DOWNLOAD_HOST = "web.ais.dk"


def _is_aisdk_hostname_mismatch(exc: BaseException) -> bool:
    """识别 urllib 包装的“证书主机名不匹配”错误，避免误提示其他网络问题。"""
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current).casefold()
        if "hostname mismatch" in message or "certificate is not valid for" in message:
            return True
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException):
            pending.append(reason)
        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            pending.append(context)
    return False


def aisdk_ssl_context(url: str, allow_hostname_mismatch: bool) -> ssl.SSLContext | None:
    """严格模式返回 None；兼容模式仅跳过 web.ais.dk 的主机名匹配。

    兼容模式仍使用系统 CA 根证书并保留证书链校验（CERT_REQUIRED）；它并不等于
    ``ssl._create_unverified_context()``。仅在 AISDK 站点证书 SAN 与 URL 主机名不一致、
    且用户显式传入开关时使用。
    """
    if not allow_hostname_mismatch:
        return None
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or parsed.hostname != AISDK_DOWNLOAD_HOST:
        raise ValueError(
            "--allow-aisdk-hostname-mismatch 仅允许用于 https://web.ais.dk/ 的 AISDK 下载地址。"
        )
    context = ssl.create_default_context()
    # 保持 verify_mode=CERT_REQUIRED，仅停止主机名匹配；不接受自签名或过期证书。
    context.check_hostname = False
    return context


def download_to_file(url: str, dst: Path, *, allow_hostname_mismatch: bool) -> None:
    """以原子写入下载一个源文件，避免中断留下被误当作完整 ZIP 的残片。"""
    context = aisdk_ssl_context(url, allow_hostname_mismatch)
    part = dst.with_suffix(dst.suffix + ".part")
    part.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120, context=context) as response, part.open("wb") as f:
            while True:
                buf = response.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
        part.replace(dst)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def ensure_raw_source(d: date, raw_dir: Path, base_url: str, template: str,
                      do_download: bool, allow_hostname_mismatch: bool = False) -> DaySource | None:
    """返回当日 CSV 或 ZIP 内 CSV 的来源，不把 ZIP 解压为磁盘文件。"""
    ds = d.strftime("%Y-%m-%d")
    csv_path = raw_dir / f"aisdk-{ds}.csv"
    if csv_path.exists():
        return DaySource("csv", csv_path)

    zip_path = raw_dir / f"aisdk-{ds}.zip"
    if zip_path.exists():
        member = find_zip_member(zip_path, d)
        return DaySource("zip", zip_path, member) if member else None

    if not do_download:
        logging.warning("%s: --raw-dir 下没有 aisdk-%s.(csv|zip)，且未开启下载，跳过。", ds, ds)
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".zip", ".csv"):
        fname = template.format(date=ds, ext=ext)
        url = base_url.rstrip("/") + "/" + fname
        dst = raw_dir / fname
        try:
            logging.info("下载 %s ...", url)
            download_to_file(url, dst, allow_hostname_mismatch=allow_hostname_mismatch)
            logging.info("已下载 %s (%.1f MB)", dst.name, dst.stat().st_size / 1e6)
            if ext == ".csv":
                return DaySource("csv", dst)
            member = find_zip_member(dst, d)
            if member:
                return DaySource("zip", dst, member)
            dst.unlink(missing_ok=True)
        except Exception as exc:  # 网络端文件命名/可用性不稳定，继续尝试另一扩展名
            if _is_aisdk_hostname_mismatch(exc) and not allow_hostname_mismatch:
                logging.warning(
                    "下载失败 %s (%s)。若浏览器可正常打开 AISDK，可重新运行并附加 "
                    "--allow-aisdk-hostname-mismatch：该模式只跳过 web.ais.dk 的主机名匹配，"
                    "仍校验证书链。",
                    url, exc,
                )
            else:
                logging.warning("下载失败 %s (%s)", url, exc)
            dst.unlink(missing_ok=True)

    logging.error(
        "%s: 下载失败，请确认 web.ais.dk 上的文件名/扩展名；若日志为主机名不匹配，"
        "使用 --allow-aisdk-hostname-mismatch 后重试。",
        ds,
    )
    return None


DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")


def discover_day_files(raw_dir: Path, start: date, end: date) -> dict[date, DaySource]:
    """扫描 raw_dir 中的日 CSV、日 ZIP、月 ZIP；只读 ZIP 目录，不解压。"""
    files = [p for p in raw_dir.rglob("*") if p.is_file() and "aisdk" in p.name.lower()]
    bydate: dict[date, DaySource] = {}

    # 直接 CSV 优先。若是历史遗留的解压文件，后续可由 --remove-legacy-extracted 清理。
    for p in files:
        if p.suffix.lower() != ".csv":
            continue
        d = _date_from_name(p.name)
        if d and start <= d <= end:
            bydate.setdefault(d, DaySource("csv", p))

    # ZIP 仅补尚缺日期；月度 ZIP 也可以包含多个日 CSV。
    for p in files:
        if p.suffix.lower() != ".zip":
            continue
        try:
            with zipfile.ZipFile(p) as z:
                members = [m for m in z.namelist() if not m.endswith("/") and m.lower().endswith(".csv")]
        except (OSError, zipfile.BadZipFile) as exc:
            logging.warning("跳过无法读取的 ZIP %s (%s)", p.name, exc)
            continue
        for member in members:
            d = _date_from_name(Path(member).name)
            if d and start <= d <= end and d not in bydate:
                bydate[d] = DaySource("zip", p, member)

    return dict(sorted(bydate.items()))


def _merge_summary(target: dict, incoming: dict) -> None:
    for key, value in incoming.items():
        dst = target.setdefault(key, {"points": 0, "sog_sum": 0.0, "sog_n": 0})
        dst["points"] += value["points"]
        dst["sog_sum"] += value["sog_sum"]
        dst["sog_n"] += value["sog_n"]


def filter_one_day(d: date, source: DaySource, boxes, mmsi_set: set[str], out_dir: Path,
                   chunksize: int, summary: dict, keep_scope: str) -> int:
    """流式读取一个日来源，筛选后原子替换日输出；失败不污染既有输出或总汇总。"""
    enc, sep = sniff_source(source)
    source_label = f"{source.path.name}::{source.member}" if source.kind == "zip" else source.path.name

    with open_source(source) as f:
        head = pd.read_csv(f, sep=sep, encoding=enc, nrows=5, dtype=str,
                           on_bad_lines="skip", low_memory=False)
    mapping = {role: find_col(head.columns, cands) for role, cands in KEEP_ROLES.items()}
    for req in ("timestamp", "mmsi", "latitude", "longitude"):
        if mapping[req] is None:
            logging.error("%s 缺列 %s；实际列：%s", source_label, req, list(head.columns))
            return 0

    usecols = sorted({value for value in mapping.values() if value})
    inv = {value: role for role, value in mapping.items() if value}
    ds = d.strftime("%Y-%m-%d")
    out_path = out_dir / f"aisdk-{ds}.csv"
    part_path = out_dir / f".{out_path.name}.part"
    part_path.unlink(missing_ok=True)

    if keep_scope == "mmsi_only" and not mmsi_set:
        raise ValueError("--keep-scope mmsi_only 需要至少一个 --mmsi。")

    wrote_header = False
    kept_total = 0
    day_summary: dict = {}
    try:
        with open_source(source) as f:
            reader = pd.read_csv(f, sep=sep, encoding=enc, usecols=usecols, dtype=str,
                                 chunksize=chunksize, on_bad_lines="skip", low_memory=False)
            for raw in reader:
                df = raw.rename(columns=inv)
                lat = pd.to_numeric(df["latitude"].str.replace(",", ".", regex=False), errors="coerce")
                lon = pd.to_numeric(df["longitude"].str.replace(",", ".", regex=False), errors="coerce")
                mmsi = df["mmsi"].astype(str).str.strip()

                region_mask = pd.Series(False, index=df.index)
                region = pd.Series("", index=df.index, dtype="object")
                for name, box in boxes:
                    inside = in_bbox(lat, lon, box).fillna(False)
                    region = region.mask(inside & (region == ""), name)
                    region_mask = region_mask | inside
                mmsi_mask = mmsi.isin(mmsi_set)

                if keep_scope == "mmsi_only":
                    mask = mmsi_mask
                else:  # region_or_mmsi 或 summary_only 都按原有地理/目标船并集选择
                    mask = region_mask | mmsi_mask

                sel = df.loc[mask].copy()
                if sel.empty:
                    continue
                sel["region"] = region.loc[mask].replace("", "MMSI_only")
                sel["date"] = ds
                kept_total += len(sel)

                # 通勤汇总先写入当天局部字典；完整日文件成功后才并入全局 summary。
                sog = pd.to_numeric(sel.get("sog", pd.Series(dtype=str)).astype(str)
                                    .str.replace(",", ".", regex=False), errors="coerce")
                sel["_sog"] = sog
                for (mm, rg), group in sel.groupby(["mmsi", "region"]):
                    key = (str(mm), ds, rg)
                    stats = day_summary.setdefault(key, {"points": 0, "sog_sum": 0.0, "sog_n": 0})
                    stats["points"] += len(group)
                    gs = group["_sog"].dropna()
                    gs = gs[gs.between(0, 80)]
                    stats["sog_sum"] += float(gs.sum())
                    stats["sog_n"] += int(gs.count())

                if keep_scope != "summary_only":
                    sel.drop(columns=["_sog"], inplace=True)
                    sel.to_csv(part_path, mode="a", index=False, header=not wrote_header, encoding="utf-8-sig")
                    wrote_header = True
    except Exception:
        part_path.unlink(missing_ok=True)
        raise

    if keep_scope != "summary_only":
        if wrote_header:
            part_path.replace(out_path)
        else:
            out_path.unlink(missing_ok=True)
    _merge_summary(summary, day_summary)
    logging.info("%s: 保留 %s 行 <- %s%s", ds, kept_total, source_label,
                 "（仅汇总，未写日切片）" if keep_scope == "summary_only" else f" -> {out_path.name if kept_total else '(无)'}")
    return kept_total


def remove_legacy_extracted_files(source: DaySource, out_dir: Path) -> None:
    """清理旧版脚本留下的可再生解压 CSV，绝不删除 ZIP、精简输出或唯一 CSV 源。"""
    # 旧的 --no-download 行为 曾把 ZIP member 解到 <out-dir>/_unzipped/；当前实现不创建该目录。
    if source.kind == "zip" and source.member:
        stale_member = out_dir / "_unzipped" / Path(source.member).name
        if stale_member.is_file():
            try:
                size_mb = stale_member.stat().st_size / 1e6
                stale_member.unlink()
                logging.info("已删除旧临时解压文件 %s (%.1f MB；ZIP 保留)", stale_member, size_mb)
            except OSError as exc:
                logging.warning("无法删除旧临时解压文件 %s (%s)", stale_member, exc)
        return

    # 旧版下载/过滤流程可能把 <raw-dir>/aisdk-YYYY-MM-DD.zip 解为同名 CSV。
    if source.kind != "csv":
        return
    zip_path = source.path.with_suffix(".zip")
    if not zip_path.is_file() or find_zip_member(zip_path, _date_from_name(source.path.name)) is None:
        return
    try:
        size_mb = source.path.stat().st_size / 1e6
        source.path.unlink()
        logging.info("已删除冗余旧解压原始文件 %s (%.1f MB；ZIP 保留)", source.path.name, size_mb)
    except OSError as exc:
        logging.warning("无法删除旧解压原始文件 %s (%s)", source.path, exc)


def write_summary(summary: dict, out_dir: Path) -> None:
    rows = []
    for (mm, ds, rg), stats in summary.items():
        rows.append({
            "mmsi": mm,
            "date": ds,
            "region": rg,
            "points": stats["points"],
            "mean_sog_knots": round(stats["sog_sum"] / stats["sog_n"], 3) if stats["sog_n"] else None,
        })
    columns = ["mmsi", "date", "region", "points", "mean_sog_knots"]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(["mmsi", "date", "region"])
    result.to_csv(out_dir / "_vessel_day_summary.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser(description="aisdk 日文件下载/流式过滤（ZIP 不落盘解压）")
    ap.add_argument("--start", default="2025-03-01", help="起始日 YYYY-MM-DD")
    ap.add_argument("--end", default="2025-06-30", help="结束日 YYYY-MM-DD")
    ap.add_argument("--raw-dir", type=Path, default=Path("./data"), help="原始 AIS ZIP/CSV 所在目录")
    ap.add_argument("--out-dir", type=Path, default=Path("./ais_filtered"), help="精简输出目录")
    ap.add_argument("--bbox", action="append", default=[], help='可重复；"名:N,W,S,E"')
    ap.add_argument("--mmsi", type=str, default=DEFAULT_MMSI, help="逗号分隔，追踪指定船")
    ap.add_argument("--chunksize", type=int, default=500_000, help="每次读取的原始行数")
    ap.add_argument("--keep-scope", choices=["region_or_mmsi", "mmsi_only", "summary_only"],
                    default="mmsi_only",
                    help="默认 mmsi_only（只保留目标船）；region_or_mmsi=保留框内所有船和目标船；summary_only=只写汇总")
    ap.add_argument("--remove-legacy-extracted", action="store_true",
                    help="成功处理后清理旧版留下的解压 CSV / _unzipped 临时 CSV（ZIP 与精简输出保留）")
    ap.add_argument("--no-download", action="store_true", help="只过滤 --raw-dir 已有文件")
    ap.add_argument("--base-url", type=str, default="https://web.ais.dk/aisdata")
    ap.add_argument("--allow-aisdk-hostname-mismatch", action="store_true",
                    help="仅对 https://web.ais.dk/ 跳过 TLS 主机名匹配；仍校验证书链。仅在该站点证书 SAN 异常时使用")
    ap.add_argument("--fname-template", type=str, default="aisdk-{date}{ext}",
                    help="下载文件名模板，含 {date} 和 {ext}")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.chunksize <= 0:
        ap.error("--chunksize 必须为正整数")
    if args.allow_aisdk_hostname_mismatch:
        logging.warning(
            "已启用 AISDK TLS 主机名兼容模式：仅允许 web.ais.dk，仍保留系统 CA 证书链校验。"
        )

    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        if start > end:
            ap.error("--start 不能晚于 --end")
        boxes = [parse_bbox(b) for b in (args.bbox or DEFAULT_BBOX)]
    except ValueError as exc:
        ap.error(str(exc))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mmsi_set = {x.strip() for x in args.mmsi.split(",") if x.strip()}
    summary: dict = {}
    processed_days = 0

    if args.no_download:
        day_files = discover_day_files(args.raw_dir, start, end)
        if not day_files:
            logging.error("在 %s 下未发现 [%s, %s] 的 aisdk 日文件；确认 --raw-dir、文件名和日期。",
                          args.raw_dir, start, end)
        else:
            logging.info("发现 %s 个日文件：%s ... %s；ZIP 将直接流式读取，不会解压到磁盘。",
                         len(day_files), min(day_files), max(day_files))
        work = day_files.items()
    else:
        work = ((d, None) for d in daterange(start, end))

    for d, discovered_source in work:
        source = discovered_source or ensure_raw_source(
            d, args.raw_dir, args.base_url, args.fname_template, do_download=True,
            allow_hostname_mismatch=args.allow_aisdk_hostname_mismatch,
        )
        if source is None:
            continue
        try:
            filter_one_day(d, source, boxes, mmsi_set, args.out_dir, args.chunksize,
                           summary, args.keep_scope)
            processed_days += 1
            if args.remove_legacy_extracted:
                remove_legacy_extracted_files(source, args.out_dir)
        except Exception:
            logging.exception("处理失败：%s", d)

    write_summary(summary, args.out_dir)
    logging.info("完成：成功处理 %s 日；逐日精简在 %s；通勤汇总为 %s。",
                 processed_days, args.out_dir, args.out_dir / "_vessel_day_summary.csv")
    return 0 if processed_days else 1


if __name__ == "__main__":
    sys.exit(main())
