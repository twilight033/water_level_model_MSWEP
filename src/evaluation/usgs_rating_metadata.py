"""拉取 86 个流域的 USGS 官方率定曲线元数据。

动机
----
审稿角度的一个自然问题：如果部分流域的径流不是由水位经率定曲线换算、而是直接
测量的（index-velocity 等），那么"用水位标签替代径流标签"在这些流域上是否失效？
要回答它必须先有两组站可比。本脚本查清 86 个流域各自用的是什么方法。

只读**响应头部的注释行**，不解析下面的率定表——判断方法只需要 ``RATING ... TYPE=``：

- ``TYPE="STGQ"``：stage-discharge，径流由水位换算而来；
- 查询返回 ``NUMBER OF FILES RETURNED BY QUERY = 0``：该站没有当前在用的率定曲线，
  可能是 index-velocity 站，也可能只是已停测——两者要靠 ``remarks`` 与站名区分，
  不能仅凭"无曲线"就断定是直接测流。

顺带记录 ``RATING ID``（如 47.2 表示这条曲线改到第 47 版）。改版次数是官方记录的
"水位—径流关系随时间漂移得多快"，与 ``redundancy.py`` 的 ``r2_h_given_q``（静态耦合
强度）是两个维度，现有分析未覆盖。

**不做**的事，理由记在 plan 里：不下载率定表用于建模，不与自拟合曲线做形状对比
（官网只发当前在用的那一版，而训练段最早到 1987 年），不替换 ``two_stage_baseline``
的区域化曲线（官方曲线本身由大量实测径流拟合而成，用了即违反留出流域零径流观测的设定）。
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import PROJECT_ROOT, RESULTS_ROOT, load_basin_ids  # noqa: E402

RATINGS_URL = "https://waterdata.usgs.gov/nwisweb/get_ratings"
# 无率定曲线的站要区分"已停测"与"仍在运行却无曲线"，前者不能算作直接测流的证据
DAILY_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
# 返回 0 文件的 ratings 响应里没有站名，无率定曲线的站要从监测站接口补
LOCATIONS_URL = ("https://api.waterdata.usgs.gov/ogcapi/v0/collections"
                 "/monitoring-locations/items")
CACHE_DIR = PROJECT_ROOT / "data" / "usgs_ratings_cache"
SUMMARY_DIR = RESULTS_ROOT / "summary"

REQUEST_DELAY_SEC = 0.5   # 站间限速；命中缓存时不触发
REQUEST_TIMEOUT_SEC = 30

# 头部形如：
#   # //STATION NAME="OYSTER RIVER NEAR DURHAM, NH"
#   # //RATING ID="47.2" TYPE="STGQ" NAME="stage-discharge" AGING=Working
#   # //RATING REMARKS="Rating 47.1 extended beyond April 2007 HW."
#   # //RATING_INDEP ROUNDING="????" PARAMETER="Gage height (ft)"
#   # //RATING_DATETIME BEGIN=20161023230000 BZONE=-05:00 END=-------------- ...
_RE_NO_FILES = re.compile(r"NUMBER OF FILES RETURNED BY QUERY\s*=\s*(\d+)")
_RE_STATION = re.compile(r'# //STATION NAME="([^"]*)"')
_RE_RETRIEVED = re.compile(r"# //RETRIEVED:\s*(\S+ \S+)")
_RE_RATING = re.compile(r'# //RATING ID="([^"]*)"\s+TYPE="([^"]*)"\s+NAME="([^"]*)"')
_RE_REMARKS = re.compile(r'# //RATING REMARKS="([^"]*)"')
_RE_EXPANSION = re.compile(r'# //RATING EXPANSION="([^"]*)"')
_RE_INDEP = re.compile(r'# //RATING_INDEP[^\n]*PARAMETER="([^"]*)"')
_RE_DEP = re.compile(r'# //RATING_DEP[^\n]*PARAMETER="([^"]*)"')
_RE_DATETIME_BEGIN = re.compile(r"# //RATING_DATETIME BEGIN=(\d{14})")


def _session() -> requests.Session:
    """带重试的会话。历史上的 qualifiers_fetcher 只有 timeout、没有重试，
    单次网络抖动就会把该站静默记成空结果。"""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET"]))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_rating(site_no: str, session: requests.Session, file_type: str = "base",
                 use_cache: bool = True) -> tuple:
    """取一个站的率定文件原文，返回 (文本, 是否命中缓存)。

    site_no 全程按字符串处理：86 个站里既有 8 位也有 9 位（010735562、010965852），
    前导零一旦丢失 USGS 会返回 400（见 qualifiers_fetcher/BASIN_ID_FIX.md）。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{site_no}.{file_type}.rdb"
    if use_cache and cache_file.is_file():
        return cache_file.read_text(encoding="utf-8", errors="replace"), True

    response = session.get(RATINGS_URL,
                           params={"site_no": site_no, "file_type": file_type},
                           timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    cache_file.write_text(response.text, encoding="utf-8")
    return response.text, False


def parse_rating_header(text: str, site_no: str) -> dict:
    """从率定文件头部解析元数据。纯函数，不发请求。"""
    row = {
        "basin": site_no,
        "has_rating": False,
        "rating_type": None,
        "rating_id": None,
        "rating_major": None,
        "station_name": None,
        "rating_name": None,
        "expansion": None,
        "indep_param": None,
        "dep_param": None,
        "n_rating_periods": None,
        "current_period_begin": None,
        "retrieved": None,
        "remarks": None,
        "note": "",
    }

    no_files = _RE_NO_FILES.search(text)
    if no_files and no_files.group(1) == "0":
        row["note"] = "查询返回 0 个文件：无当前在用的率定曲线"
        return row

    rating = _RE_RATING.search(text)
    if rating is None:
        # 既不是 0 文件、又解析不出 RATING 行——格式意外，留痕而非静默跳过
        row["note"] = "响应中未找到 RATING ID/TYPE 行，格式与预期不符"
        return row

    row["has_rating"] = True
    row["rating_id"], row["rating_type"], row["rating_name"] = rating.groups()
    try:
        # "47.2" → 47、"4.00" → 4：整数部分即这条曲线改过的版次
        row["rating_major"] = int(float(row["rating_id"]))
    except ValueError:
        row["note"] = f"RATING ID 非数值：{row['rating_id']}"

    for key, pattern in (("station_name", _RE_STATION), ("remarks", _RE_REMARKS),
                         ("expansion", _RE_EXPANSION), ("indep_param", _RE_INDEP),
                         ("dep_param", _RE_DEP), ("retrieved", _RE_RETRIEVED)):
        match = pattern.search(text)
        if match:
            row[key] = match.group(1).strip()

    begins = _RE_DATETIME_BEGIN.findall(text)
    if begins:
        row["n_rating_periods"] = len(begins)
        row["current_period_begin"] = pd.to_datetime(begins[-1], format="%Y%m%d%H%M%S",
                                                     errors="coerce")
    return row


def last_discharge_date(site_no: str, session: requests.Session):
    """该站最后一条日均流量的日期；查不到返回 None。

    用于给"无当前率定曲线"的站定性：若流量记录早已终止，缺曲线只是停测后
    率定文件被撤下，**不能**当作该站直接测流的证据。
    """
    try:
        response = session.get(DAILY_URL, timeout=REQUEST_TIMEOUT_SEC, params={
            "monitoring_location_id": f"USGS-{site_no}", "parameter_code": "00060",
            "limit": 1, "sortby": "-time", "f": "json"})
        response.raise_for_status()
        features = response.json().get("features") or []
    except (requests.exceptions.RequestException, ValueError):
        return None
    return features[0]["properties"].get("time") if features else None


def station_name_from_api(site_no: str, session: requests.Session):
    """从监测站接口取站名；查不到返回 None。

    ratings 响应在"0 个文件"时不含 `STATION NAME=`，故无率定曲线的站只能这样补。
    """
    try:
        response = session.get(LOCATIONS_URL, timeout=REQUEST_TIMEOUT_SEC, params={
            "monitoring_location_number": site_no, "f": "json"})
        response.raise_for_status()
        features = response.json().get("features") or []
    except (requests.exceptions.RequestException, ValueError):
        return None
    return features[0]["properties"].get("monitoring_location_name") if features else None


def classify_no_rating(table: pd.DataFrame, session: requests.Session) -> pd.DataFrame:
    """给无率定曲线的站补上流量记录终止日期与定性结论。"""
    missing = table.index[~table["has_rating"].fillna(False)]
    if not len(missing):
        return table

    table["last_discharge"] = pd.NA
    print()
    print(f"为 {len(missing)} 个无率定曲线的站查询流量记录终止日期:")
    for i in missing:
        site_no = table.at[i, "basin"]
        last = last_discharge_date(site_no, session)
        table.at[i, "last_discharge"] = last
        if last is None:
            verdict = "无法判定：查不到日均流量记录"
        elif pd.Timestamp(last[:10]) < pd.Timestamp.now() - pd.Timedelta(days=365):
            verdict = f"已停测（流量记录止于 {last[:10]}），缺曲线非测流方法所致"
        else:
            verdict = f"仍在运行（最后记录 {last[:10]}）却无 base/exsa/corr 曲线，原因无法从公开接口判定"
        table.at[i, "note"] = verdict
        print(f"  {site_no}  {verdict}")
        time.sleep(REQUEST_DELAY_SEC)
    return table


def build_metadata_table(basin_ids, use_cache: bool = True) -> pd.DataFrame:
    """逐站串行抓取并解析。失败的站也写一行，绝不静默丢站。"""
    session = _session()
    rows = []
    for i, site_no in enumerate(basin_ids, 1):
        try:
            text, cached = fetch_rating(site_no, session, use_cache=use_cache)
            row = parse_rating_header(text, site_no)
        except requests.exceptions.RequestException as exc:
            rows.append({"basin": site_no, "has_rating": False,
                         "note": f"请求失败：{type(exc).__name__}: {exc}"})
            print(f"  [{i:>2}/{len(basin_ids)}] {site_no}  请求失败：{exc}")
            continue

        rows.append(row)
        flag = "缓存" if cached else "抓取"
        detail = (f"{row['rating_type']} v{row['rating_id']}" if row["has_rating"]
                  else row["note"])
        print(f"  [{i:>2}/{len(basin_ids)}] {site_no}  {flag}  {detail}")
        if not cached:
            time.sleep(REQUEST_DELAY_SEC)

    return pd.DataFrame(rows)


# 判据先写死，避免拿到结果后挑配置：四个含留出情景的可比配置里至少 3 个
# 同方向且 p<0.05 才算稳健，否则与 redundancy 的 F3 一样记为"不进正文"。
MIN_ROBUST_CONFIGS = 3
ALPHA = 0.05


def revision_gain_correlation(table: pd.DataFrame) -> pd.DataFrame:
    """率定曲线改版次数与留出替代增益的相关。

    改版次数是官方记录的"水位—径流关系随时间漂移得多快"，与 redundancy 的
    ``r2_h_given_q``（静态耦合强度）是两个维度。若替代增益确实来自 h→Q 的物理
    耦合，则关系越不稳（改版越勤）增益应越小，预期为负相关。
    """
    from evaluation.redundancy import correlate_with_gain
    from evaluation.run_config import load_summary

    if not (SUMMARY_DIR / "all_metrics.csv").exists():
        print("尚无 all_metrics.csv，跳过相关性分析")
        return pd.DataFrame()

    metrics, _ = load_summary()
    ratings = table[["basin", "rating_major"]].dropna(subset=["rating_major"])
    configs = sorted(set(metrics[metrics["scenario"].str.startswith("q_holdout")]["config"]))

    rows = []
    for config in configs:
        for ratio in (30, 50, 70):
            rows.append(correlate_with_gain(
                ratings, metrics, task="flow", config=config,
                scenario=f"q_holdout{ratio}", held_out=True,
                columns=("rating_major",)))
    return pd.DataFrame([r for r in rows if r.get("n", 0) >= 5])


def report_correlation(corr: pd.DataFrame):
    """按预先写死的判据判断结果稳不稳健。"""
    if corr.empty or "spearman_rating_major" not in corr:
        print("可配对的运行不足，无相关性结果")
        return

    out = SUMMARY_DIR / "rating_revision_gain_correlation.csv"
    corr.to_csv(out, index=False, encoding="utf-8-sig")

    print()
    print("率定曲线改版次数 vs 留出替代增益（被留出流域，同一可比配置内）:")
    cols = ["config", "scenario", "n", "mean_gain",
            "spearman_rating_major", "p_rating_major"]
    print(corr[[c for c in cols if c in corr]].round(4).to_string(index=False))

    # 主检验只看 70% 留出：替代效应最强、n 最大（85 个被留出流域）
    main = corr[corr["scenario"] == "q_holdout70"].dropna(subset=["spearman_rating_major"])
    if main.empty:
        print("  70% 留出无可用结果")
        return

    sig = main[main["p_rating_major"] < ALPHA]
    same_sign = len(sig) and (sig["spearman_rating_major"] > 0).nunique() == 1
    robust = len(sig) >= MIN_ROBUST_CONFIGS and same_sign

    print()
    print(f"判据：70% 留出的 {len(main)} 个配置中，显著(p<{ALPHA})的有 {len(sig)} 个，"
          f"需 ≥{MIN_ROBUST_CONFIGS} 个且同方向")
    print(f"结论：{'稳健，可考虑写入正文' if robust else '不稳健，不作机制主张，仅留 CSV 备查'}")
    print(f"  已写入 {out}")


# ===== --scan-all：在 CAMELSH 全集中筛出径流非率定曲线推算的流域 =====
#
# 目的是给"替代效应是否依赖 Q=f(H) 这一数据生成过程"提供对照组。判据只能是间接的：
# 公开接口取不到站级径流计算方法编码（`time-series-methods` 查 QVELO/QGATE 均为 0 条），
# 因此用"仍在运行却没有任何率定曲线"来推断该站的径流并非由水位换算。已停测的站必须排除
# ——停测后率定文件会被撤下，无法区分它历史上用的是哪种方法。

# 有效比例阈值与现有 86 站一致（source_of_truth_v2.md §3.2），新流域才能同口径。
MIN_VALID_FRAC = 0.10
MIN_OVERLAP_YEARS = 8
MAX_YEARS_SINCE_LAST_Q = 2
HOURS_PER_YEAR = 8766

CANDIDATES_CSV = "nonrating_basin_candidates.csv"


def _camelsh_paths() -> dict:
    """CAMELSH 各数据源路径；根目录取自 config.CAMELSH_DATA_PATH。"""
    import config

    root = Path(config.CAMELSH_DATA_PATH) / "CAMELSH"
    return {
        "forcing": root / "timeseries" / "Data" / "CAMELSH" / "timeseries",
        "waterlevel": root / "Hourly2" / "Hourly2",
        "basin_id": root / "attributes" / "attributes_gageii_BasinID.csv",
        "shapefile": root / "shapefiles" / "CAMELSH_shapefile",
    }


def scan_basin_ids(paths: dict) -> list:
    """强迫与水位文件都存在的站号——两者齐备才可能进入建模。"""
    forcing = {p.stem for p in paths["forcing"].glob("*.nc")}
    wl = {p.name[:-len("_hourly.nc")] for p in paths["waterlevel"].glob("*_hourly.nc")}
    return sorted(forcing & wl)


def basin_coverage(site_no: str, paths: dict) -> dict:
    """从 Hourly2 读逐时径流与水位，算有效比例与重叠年数。

    该文件同时含两个变量，一次读取即可。**"有文件"不等于"有数据"**——已知
    02469761 虽有 Hourly2 文件，water_level 却全为 NaN，故必须实读而非只看文件是否存在。
    """
    import numpy as np
    import xarray as xr

    out = {"q_valid_frac": None, "h_valid_frac": None,
           "overlap_years": None, "last_waterlevel": None}
    path = paths["waterlevel"] / f"{site_no}_hourly.nc"
    if not path.is_file():
        return out
    with xr.open_dataset(path) as d:
        q = np.isfinite(d["streamflow"].values)
        h = np.isfinite(d["water_level"].values)
        times = d["time"].values
    out["q_valid_frac"] = round(float(q.mean()), 4)
    out["h_valid_frac"] = round(float(h.mean()), 4)
    out["overlap_years"] = round(float((q & h).sum()) / HOURS_PER_YEAR, 2)
    if h.any():
        out["last_waterlevel"] = str(times[h][-1])[:10]
    return out


def shapefile_basins(paths: dict) -> set:
    """流域边界 dbf 中的站号；用户自行提取降水需要几何。

    同时收原始写法与补零到 8 位的写法：dbf 若把站号存成整数，前导零会丢失。
    """
    import shapefile

    ids = set()
    with shapefile.Reader(str(paths["shapefile"])) as reader:
        fields = [f[0] for f in reader.fields[1:]]
        key = next((f for f in fields if f.upper() in ("STAID", "GAGE_ID", "GAGEID")),
                   fields[0])
        idx = fields.index(key)
        for record in reader.records():
            raw = str(record[idx]).strip()
            ids.add(raw)
            ids.add(raw.zfill(8))
    return ids


def basin_areas(paths: dict) -> dict:
    """站号 → 流域面积（km²）。面积在 BasinID.csv，不在 Bas_Morph.csv。"""
    df = pd.read_csv(paths["basin_id"], dtype={"STAID": str})
    df["STAID"] = df["STAID"].astype(str).str.strip().str.zfill(8)
    return dict(zip(df["STAID"], df["DRAIN_SQKM"]))


def _years_since(date_str) -> float:
    if not date_str:
        return float("inf")
    return (pd.Timestamp.now() - pd.Timestamp(date_str[:10])).days / 365.25


def _screen_one(row: dict, session: requests.Session, paths: dict,
                shp_ids: set) -> dict:
    """对一个无 base 曲线的站跑完步 2–5，就地补齐字段与落选原因。"""
    site_no = row["basin"]

    # 步 2：确认 exsa / corr 也没有曲线。任一存在都说明径流仍由水位换算。
    for file_type in ("exsa", "corr"):
        try:
            text, _ = fetch_rating(site_no, session, file_type=file_type)
        except requests.exceptions.RequestException as exc:
            row["reject_reason"] = f"请求 {file_type} 失败：{type(exc).__name__}"
            return row
        no_files = _RE_NO_FILES.search(text)
        if not (no_files and no_files.group(1) == "0"):
            row["reject_reason"] = f"{file_type} 仍有率定曲线"
            return row

    # 步 3：仍在运行。停测站无法判断历史上用的是哪种方法，必须排除。
    row["last_discharge"] = last_discharge_date(site_no, session)
    years = _years_since(row["last_discharge"])
    if years > MAX_YEARS_SINCE_LAST_Q:
        row["reject_reason"] = (f"已停测（径流止于 {row['last_discharge'] or '无记录'}）"
                                if row["last_discharge"] else "查不到径流记录")
        return row

    # 确认在运行后再补站名，省掉停测站的请求
    if not row.get("station_name"):
        row["station_name"] = station_name_from_api(site_no, session)

    # 步 4：水位与径流覆盖足够
    row.update(basin_coverage(site_no, paths))
    if row["h_valid_frac"] is None:
        row["reject_reason"] = "无 Hourly2 文件"
        return row
    if row["h_valid_frac"] == 0:
        row["reject_reason"] = "水位数据全为空"
        return row
    for key, label in (("q_valid_frac", "径流"), ("h_valid_frac", "水位")):
        if row[key] < MIN_VALID_FRAC:
            row["reject_reason"] = f"{label}有效比例 {row[key]:.3f} < {MIN_VALID_FRAC}"
            return row
    if row["overlap_years"] < MIN_OVERLAP_YEARS:
        row["reject_reason"] = f"Q/H 重叠 {row['overlap_years']:.1f} 年 < {MIN_OVERLAP_YEARS}"
        return row

    # 步 5：有流域边界
    if site_no not in shp_ids:
        row["reject_reason"] = "流域边界缺失"
        return row

    row["reject_reason"] = ""
    return row


def scan_all():
    """扫描 CAMELSH 全集，输出候选流域表。"""
    paths = _camelsh_paths()
    if not paths["forcing"].is_dir():
        raise FileNotFoundError(
            f"CAMELSH 强迫目录不存在: {paths['forcing']}\n"
            f"请确认数据盘已挂载，或修改 config.py 的 CAMELSH_DATA_PATH")

    basins = scan_basin_ids(paths)
    areas = basin_areas(paths)
    shp_ids = shapefile_basins(paths)
    session = _session()
    print(f"强迫 ∩ 水位 共 {len(basins)} 站；边界 dbf 收录 {len(shp_ids)} 个站号写法")
    print(f"阈值：有效比例 ≥ {MIN_VALID_FRAC}，Q/H 重叠 ≥ {MIN_OVERLAP_YEARS} 年，"
          f"径流最后记录距今 ≤ {MAX_YEARS_SINCE_LAST_Q} 年")
    print()

    rows, n_norating = [], 0
    for i, site_no in enumerate(basins, 1):
        # 步 1：base 曲线。有曲线即为水位换算站，不是候选。
        try:
            text, cached = fetch_rating(site_no, session)
        except requests.exceptions.RequestException as exc:
            rows.append({"basin": site_no, "has_rating": None,
                         "drain_sqkm": areas.get(site_no),
                         "reject_reason": f"请求 base 失败：{type(exc).__name__}"})
            continue

        row = parse_rating_header(text, site_no)
        row["drain_sqkm"] = areas.get(site_no)
        if row["has_rating"]:
            row["reject_reason"] = "有水位-流量率定曲线"
        else:
            n_norating += 1
            row = _screen_one(row, session, paths, shp_ids)
            mark = "★入选" if row["reject_reason"] == "" else row["reject_reason"]
            print(f"  [{i:>4}/{len(basins)}] {site_no}  无 base 曲线 → {mark}")
        rows.append(row)

        if not cached:
            time.sleep(REQUEST_DELAY_SEC)
        if i % 250 == 0:
            print(f"  … 已扫 {i}/{len(basins)}，其中无 base 曲线 {n_norating} 个")

    table = pd.DataFrame(rows)
    table["selected"] = table["reject_reason"].fillna("x") == ""

    cols = ["basin", "station_name", "drain_sqkm", "has_rating", "rating_type", "rating_id",
            "last_discharge", "last_waterlevel", "q_valid_frac", "h_valid_frac",
            "overlap_years", "selected", "reject_reason"]
    table = table.reindex(columns=cols + [c for c in table.columns if c not in cols])

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARY_DIR / CANDIDATES_CSV
    table.to_csv(out, index=False, encoding="utf-8-sig")

    print()
    print(f"扫描 {len(table)} 站：无 base 曲线 {n_norating} 个，最终入选 "
          f"{int(table['selected'].sum())} 个")
    print()
    print("落选原因分布:")
    print(table.loc[~table["selected"], "reject_reason"].value_counts().to_string())

    picked = table[table["selected"]]
    if len(picked):
        print()
        show = ["basin", "station_name", "drain_sqkm", "last_discharge",
                "q_valid_frac", "h_valid_frac", "overlap_years"]
        print(picked[show].to_string(index=False))
        print()
        print("入选站号（可直接粘进降水提取流程）:")
        print(" ".join(picked["basin"].tolist()))
    else:
        print()
        print("无入选站——需放宽阈值重筛，各站的覆盖率与落选原因已写入 CSV")

    print()
    print(f"  已写入 {out}")


def main():
    basin_ids = load_basin_ids()
    print(f"共 {len(basin_ids)} 个流域，缓存目录 {CACHE_DIR}")
    print()

    table = build_metadata_table(basin_ids)
    table = classify_no_rating(table, _session())

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARY_DIR / "usgs_rating_metadata.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")

    print()
    print(f"率定曲线类型分布（共 {len(table)} 站）:")
    print(table["rating_type"].value_counts(dropna=False).to_string())

    n_stgq = int((table["rating_type"] == "STGQ").sum())
    n_other = int(table["has_rating"].fillna(False).sum()) - n_stgq
    n_none = len(table) - int(table["has_rating"].fillna(False).sum())
    print()
    print(f"  水位换算（STGQ）      : {n_stgq}")
    print(f"  其他率定类型          : {n_other}")
    print(f"  无当前率定曲线        : {n_none}")

    if n_none:
        print()
        print("无率定曲线的站:")
        cols = [c for c in ("basin", "last_discharge", "note") if c in table]
        print(table[~table["has_rating"].fillna(False)][cols].to_string(index=False))

    major = table["rating_major"].dropna()
    if len(major):
        print()
        print(f"率定曲线改版次数（n={len(major)}）: "
              f"中位 {major.median():.0f}  最小 {major.min():.0f}  最大 {major.max():.0f}")

    print()
    print(f"  已写入 {out}")

    # 判据写在 plan 里：非 STGQ 不足 15 个就无法构造直接测流对照组
    if n_other + n_none < 15:
        print()
        print(f"判决：非 STGQ 的站共 {n_other + n_none} 个（<15），"
              f"无法构造\"直接测流\"对照组，分层分析不可行。")

    report_correlation(revision_gain_correlation(table))


if __name__ == "__main__":
    # --scan-all：在 CAMELSH 全集中找对照组；不带参数：只处理现有 86 站
    if "--scan-all" in sys.argv:
        scan_all()
    else:
        main()
