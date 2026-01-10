"""
USGS NWIS Data Qualifiers Fetcher
从USGS NWIS API获取数据质量标签（qualifiers）并与本地CAMELSH数据合并

功能：
1. 使用USGS NWIS Instantaneous Values (iv) API
2. 查询指定gauge_id和时间范围的径流(00060)和水位(00065)数据
3. 提取每条观测的qualifiers（数据质量标记）
4. 与本地CAMELSH数据按[gauge_id, time]对齐合并
5. 生成包含Q, H, Q_flag, H_flag的新文件
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import time
from xml.etree import ElementTree as ET

# Optional: used for reading CAMELSH dataset directly (same approach as multi_task_lstm.py).
# Keep imports optional so the fetcher can still be used for "fetch-only" scenarios.
try:
    from improved_camelsh_reader import ImprovedCAMELSHReader  # type: ignore
    from hydrodataset import StandardVariable  # type: ignore
except Exception:  # pragma: no cover
    ImprovedCAMELSHReader = None  # type: ignore
    StandardVariable = None  # type: ignore


class USGSQualifiersFetcher:
    """USGS数据质量标签获取器"""
    
    # USGS NWIS API配置
    BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"
    
    # 参数代码
    PARAM_DISCHARGE = "00060"  # 径流 (cubic feet per second)
    PARAM_GAGE_HEIGHT = "00065"  # 水位 (feet)
    
    # 常见的USGS qualifiers及其含义
    QUALIFIER_MEANINGS = {
        'A': 'Approved for publication',
        'P': 'Provisional data subject to revision',
        'e': 'Estimated value',
        '<': 'Less than indicated value',
        '>': 'Greater than indicated value',
        'X': 'Pumping or nearby pumping',
        '&': 'Value affected by dam, gate, or control structure',
        'R': 'Rating being developed or revised',
        'Eqp': 'Equipment malfunction',
        'Bkw': 'Backwater',
        'Ice': 'Ice affected',
        'Ssn': 'Seasonal site',
        'Rat': 'Rating being developed or revised',
        'Dis': 'Data-collection discontinued',
        'Mnt': 'Site visit or maintenance',
    }
    
    def __init__(self, output_dir: str = "qualifiers_output", cache_dir: str = "qualifiers_cache"):
        """
        初始化
        
        Parameters
        ----------
        output_dir : str
            输出文件目录
        cache_dir : str
            缓存文件目录（避免重复请求）
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"输出目录: {self.output_dir}")
        print(f"缓存目录: {self.cache_dir}")
    
    def _fetch_qualifiers_for_gauge_single_range(
        self,
        gauge_id: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取单个站点的qualifiers
        
        Parameters
        ----------
        gauge_id : str
            USGS站点ID（例如："01646500"）
        start_date : str
            开始日期 "YYYY-MM-DD"
        end_date : str
            结束日期 "YYYY-MM-DD"
        use_cache : bool
            是否使用缓存
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (discharge_df, gage_height_df)
            每个DataFrame包含列：datetime, value, qualifiers
        """
        # 检查缓存
        cache_file = os.path.join(
            self.cache_dir,
            f"{gauge_id}_{start_date}_{end_date}.json"
        )
        
        if use_cache and os.path.exists(cache_file):
            print(f"  从缓存加载: {gauge_id}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            discharge_df = pd.DataFrame(cached_data['discharge'])
            gage_height_df = pd.DataFrame(cached_data['gage_height'])
            
            # 转换datetime列
            if not discharge_df.empty:
                discharge_df['datetime'] = pd.to_datetime(discharge_df['datetime'])
            if not gage_height_df.empty:
                gage_height_df['datetime'] = pd.to_datetime(gage_height_df['datetime'])
            
            return discharge_df, gage_height_df
        
        # 构建API请求
        params = {
            'format': 'json',
            'sites': gauge_id,
            'startDT': start_date,
            'endDT': end_date,
            'parameterCd': f"{self.PARAM_DISCHARGE},{self.PARAM_GAGE_HEIGHT}",
            'siteStatus': 'all'
        }
        
        try:
            print(f"  请求USGS API: {gauge_id} ({start_date} 到 {end_date})")
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 解析响应
            discharge_df = self._parse_timeseries(data, self.PARAM_DISCHARGE)
            gage_height_df = self._parse_timeseries(data, self.PARAM_GAGE_HEIGHT)
            
            # 保存到缓存
            cache_data = {
                'discharge': discharge_df.to_dict('records') if not discharge_df.empty else [],
                'gage_height': gage_height_df.to_dict('records') if not gage_height_df.empty else []
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            return discharge_df, gage_height_df
            
        except requests.exceptions.RequestException as e:
            print(f"  警告: 请求失败 - {gauge_id}: {e}")
            return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            print(f"  警告: 解析失败 - {gauge_id}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def fetch_qualifiers_for_gauge(
        self,
        gauge_id: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取单个站点的qualifiers（自动按年分批，避免超长时间范围导致超时）。

        Notes
        -----
        - USGS IV API 对超长时间范围（例如 2001-2024）很容易超时。
        - 当跨度超过约 370 天时，此函数会自动按年拆分请求，再把结果合并。
        """
        # Decide whether to chunk by year
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        except Exception:
            # Fallback: just try single-range fetch
            return self._fetch_qualifiers_for_gauge_single_range(
                gauge_id=gauge_id,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )

        if (end_dt - start_dt).days <= 370:
            return self._fetch_qualifiers_for_gauge_single_range(
                gauge_id=gauge_id,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
            )

        # Chunk by years (inclusive)
        start_year = int(start_dt.year)
        end_year = int(end_dt.year)

        discharge_parts = []
        gage_height_parts = []

        # Small delay between year-requests to reduce rate-limit risk
        inter_year_delay_sec = 0.2

        for year in range(start_year, end_year + 1):
            y_start = pd.Timestamp(year=year, month=1, day=1)
            y_end = pd.Timestamp(year=year, month=12, day=31)

            # Clip to requested overall range
            chunk_start = max(start_dt, y_start).strftime("%Y-%m-%d")
            chunk_end = min(end_dt, y_end).strftime("%Y-%m-%d")

            if pd.to_datetime(chunk_start) > pd.to_datetime(chunk_end):
                continue

            q_df, h_df = self._fetch_qualifiers_for_gauge_single_range(
                gauge_id=gauge_id,
                start_date=chunk_start,
                end_date=chunk_end,
                use_cache=use_cache,
            )

            if q_df is not None and not q_df.empty:
                discharge_parts.append(q_df)
            if h_df is not None and not h_df.empty:
                gage_height_parts.append(h_df)

            time.sleep(inter_year_delay_sec)

        # Merge parts
        if discharge_parts:
            discharge_df = pd.concat(discharge_parts, ignore_index=False)
            discharge_df = discharge_df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        else:
            discharge_df = pd.DataFrame(columns=["datetime", "value", "qualifiers"])

        if gage_height_parts:
            gage_height_df = pd.concat(gage_height_parts, ignore_index=False)
            gage_height_df = gage_height_df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        else:
            gage_height_df = pd.DataFrame(columns=["datetime", "value", "qualifiers"])

        return discharge_df, gage_height_df
    
    def _parse_timeseries(self, data: dict, param_code: str) -> pd.DataFrame:
        """
        解析USGS JSON响应中的时间序列数据
        
        Parameters
        ----------
        data : dict
            USGS API返回的JSON数据
        param_code : str
            参数代码（00060或00065）
        
        Returns
        -------
        pd.DataFrame
            包含列：datetime, value, qualifiers
        """
        try:
            time_series = data['value']['timeSeries']
            
            for ts in time_series:
                # 检查参数代码
                variable = ts['variable']
                if variable['variableCode'][0]['value'] == param_code:
                    # 提取时间序列值
                    values = ts['values'][0]['value']
                    
                    if not values:
                        return pd.DataFrame(columns=['datetime', 'value', 'qualifiers'])
                    
                    records = []
                    for val in values:
                        dt_str = val['dateTime']
                        value = val['value']
                        qualifiers = val.get('qualifiers', [''])
                        
                        # 转换datetime（处理时区）
                        dt = pd.to_datetime(dt_str)
                        # 转换为UTC
                        if dt.tz is not None:
                            dt = dt.tz_convert('UTC')
                        else:
                            dt = dt.tz_localize('UTC')
                        
                        # 提取qualifiers列表
                        qual_list = [q for q in qualifiers if q]
                        qual_str = ','.join(qual_list) if qual_list else 'none'
                        
                        records.append({
                            'datetime': dt,
                            'value': float(value) if value not in ['', 'NA', 'NaN'] else np.nan,
                            'qualifiers': qual_str
                        })
                    
                    df = pd.DataFrame(records)
                    return df
            
            # 如果没有找到对应参数
            return pd.DataFrame(columns=['datetime', 'value', 'qualifiers'])
            
        except KeyError as e:
            print(f"    警告: 解析数据时缺少键 - {e}")
            return pd.DataFrame(columns=['datetime', 'value', 'qualifiers'])
        except Exception as e:
            print(f"    警告: 解析数据时出错 - {e}")
            return pd.DataFrame(columns=['datetime', 'value', 'qualifiers'])
    
    def fetch_multiple_gauges(
        self,
        gauge_ids: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        delay: float = 0.5
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        批量获取多个站点的qualifiers
        
        Parameters
        ----------
        gauge_ids : List[str]
            站点ID列表
        start_date : str
            开始日期 "YYYY-MM-DD"
        end_date : str
            结束日期 "YYYY-MM-DD"
        use_cache : bool
            是否使用缓存
        delay : float
            请求之间的延迟（秒），避免被API限流
        
        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            {gauge_id: {'discharge': df, 'gage_height': df}}
        """
        results = {}
        
        print(f"\n开始获取 {len(gauge_ids)} 个站点的qualifiers...")
        print(f"时间范围: {start_date} 到 {end_date}")
        
        for gauge_id in tqdm(gauge_ids, desc="获取qualifiers"):
            discharge_df, gage_height_df = self.fetch_qualifiers_for_gauge(
                gauge_id=gauge_id,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache
            )
            
            results[gauge_id] = {
                'discharge': discharge_df,
                'gage_height': gage_height_df
            }
            
            # 延迟以避免API限流
            # In chunked mode we rely on per-year caching, so the full-range cache file may not exist.
            if not use_cache:
                time.sleep(delay)
        
        # 统计
        n_discharge = sum(1 for r in results.values() if not r['discharge'].empty)
        n_gage_height = sum(1 for r in results.values() if not r['gage_height'].empty)
        
        print(f"\n获取完成:")
        print(f"  有径流数据的站点: {n_discharge}/{len(gauge_ids)}")
        print(f"  有水位数据的站点: {n_gage_height}/{len(gauge_ids)}")
        
        return results

    def _load_camelsh_from_dataset(
        self,
        camelsh_data_path: str,
        gauge_ids: List[str],
        time_range: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load streamflow and water level from the CAMELSH dataset directly.

        Parameters
        ----------
        camelsh_data_path : str
            Path to CAMELSH dataset root.
        gauge_ids : List[str]
            Gauge IDs to load.
        time_range : List[str]
            [start_date, end_date] in YYYY-MM-DD format.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (flow_df, waterlevel_df) with datetime index and gauge_id columns.
        """
        if ImprovedCAMELSHReader is None or StandardVariable is None:
            raise ImportError(
                "Missing dependencies for reading CAMELSH dataset. "
                "Please ensure `hydrodataset` is installed and `improved_camelsh_reader.py` is importable."
            )

        if not isinstance(time_range, list) or len(time_range) != 2:
            raise ValueError("time_range must be a list like ['YYYY-MM-DD', 'YYYY-MM-DD']")

        reader = ImprovedCAMELSHReader(camelsh_data_path, download=False, use_batch=True)

        # Streamflow
        flow_ds = reader.read_ts_xrdataset(
            gage_id_lst=gauge_ids,
            t_range=time_range,
            var_lst=[StandardVariable.STREAMFLOW],
        )
        flow_df = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T

        # Water level (may not exist for all basins)
        try:
            wl_ds = reader.read_ts_xrdataset(
                gage_id_lst=gauge_ids,
                t_range=time_range,
                var_lst=[StandardVariable.WATER_LEVEL],
            )
            waterlevel_df = wl_ds[StandardVariable.WATER_LEVEL].to_pandas().T
        except Exception:
            # Keep same index shape if possible; otherwise return empty DF.
            waterlevel_df = pd.DataFrame(index=flow_df.index)

        # Ensure timezone consistency (UTC)
        if flow_df.index.tz is None:
            flow_df.index = flow_df.index.tz_localize("UTC")
        else:
            flow_df.index = flow_df.index.tz_convert("UTC")

        if waterlevel_df.index.tz is None:
            waterlevel_df.index = waterlevel_df.index.tz_localize("UTC")
        else:
            waterlevel_df.index = waterlevel_df.index.tz_convert("UTC")

        # Normalize column names to strings for consistent matching
        flow_df.columns = flow_df.columns.astype(str)
        waterlevel_df.columns = waterlevel_df.columns.astype(str)

        return flow_df, waterlevel_df

    def _align_qualifiers_to_time_index(
        self,
        q_df: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        *,
        align_mode: str = "nearest",
        tolerance: str = "30min",
        direction: str = "nearest",
    ) -> pd.Series:
        """
        Align qualifier strings onto a target time index.

        Notes
        -----
        - USGS IV data is often 15-min resolution, while CAMELSH target series may be hourly.
          An exact timestamp match can therefore create many 'missing'. This helper supports
          "nearest"/"backward"/"forward" alignment using merge_asof with a tolerance.

        Parameters
        ----------
        q_df : pd.DataFrame
            DataFrame with columns ['datetime', 'qualifiers'] at minimum.
        time_index : pd.DatetimeIndex
            Target time index (timezone-aware preferred; will be coerced to UTC if tz-aware).
        align_mode : str
            'exact' or 'nearest' (asof-based). Any non-'exact' will use merge_asof.
        tolerance : str
            Pandas Timedelta-like string, e.g. '30min', '1H'.
        direction : str
            merge_asof direction: 'nearest', 'backward', or 'forward'.

        Returns
        -------
        pd.Series
            Qualifier strings aligned to time_index, with 'missing' as default.
        """
        if len(time_index) == 0:
            return pd.Series(dtype=object)

        # Ensure sorted unique index
        target_index = pd.DatetimeIndex(time_index).sort_values()
        if target_index.tz is not None:
            target_index = target_index.tz_convert("UTC")

        if q_df is None or q_df.empty:
            return pd.Series(["missing"] * len(target_index), index=target_index, dtype=object)

        if "datetime" not in q_df.columns or "qualifiers" not in q_df.columns:
            return pd.Series(["missing"] * len(target_index), index=target_index, dtype=object)

        right = q_df[["datetime", "qualifiers"]].copy()
        right["datetime"] = pd.to_datetime(right["datetime"])
        # Coerce timezone to UTC for stable alignment
        if right["datetime"].dt.tz is None:
            right["datetime"] = right["datetime"].dt.tz_localize("UTC")
        else:
            right["datetime"] = right["datetime"].dt.tz_convert("UTC")

        right = right.dropna(subset=["datetime"]).sort_values("datetime")

        if right.empty:
            return pd.Series(["missing"] * len(target_index), index=target_index, dtype=object)

        if align_mode == "exact":
            right_indexed = right.set_index("datetime")
            out = right_indexed["qualifiers"].reindex(target_index, fill_value="missing")
            out.index = target_index
            return out

        # Asof-based alignment (nearest/backward/forward within tolerance)
        left = pd.DataFrame({"datetime": target_index})
        tol = pd.Timedelta(tolerance)
        merged = pd.merge_asof(
            left,
            right,
            on="datetime",
            direction=direction,
            tolerance=tol,
        )
        out = merged["qualifiers"].fillna("missing").astype(object)
        out.index = target_index
        return out
    
    def merge_with_camelsh(
        self,
        camelsh_flow_file: str,
        camelsh_waterlevel_file: str,
        qualifiers_data: Dict[str, Dict[str, pd.DataFrame]],
        output_file: str = None,
        add_weights: bool = True
    ) -> pd.DataFrame:
        """
        将qualifiers与本地CAMELSH数据合并
        
        Parameters
        ----------
        camelsh_flow_file : str
            CAMELSH径流数据文件路径（CSV，index=时间，columns=gauge_id）
        camelsh_waterlevel_file : str
            CAMELSH水位数据文件路径
        qualifiers_data : Dict
            获取的qualifiers数据
        output_file : str, optional
            输出文件路径
        add_weights : bool
            是否添加权重列（基于qualifiers计算）
        
        Returns
        -------
        pd.DataFrame
            合并后的数据
        """
        print("\n开始合并qualifiers与CAMELSH数据...")
        
        # 1. 加载CAMELSH数据
        print("  加载CAMELSH径流数据...")
        flow_df = pd.read_csv(camelsh_flow_file, index_col=0, parse_dates=True)
        
        print("  加载CAMELSH水位数据...")
        waterlevel_df = pd.read_csv(camelsh_waterlevel_file, index_col=0, parse_dates=True)
        
        # 确保时区一致（转换为UTC）
        if flow_df.index.tz is None:
            flow_df.index = flow_df.index.tz_localize('UTC')
        else:
            flow_df.index = flow_df.index.tz_convert('UTC')
        
        if waterlevel_df.index.tz is None:
            waterlevel_df.index = waterlevel_df.index.tz_localize('UTC')
        else:
            waterlevel_df.index = waterlevel_df.index.tz_convert('UTC')
        
        print(f"  径流数据形状: {flow_df.shape}")
        print(f"  水位数据形状: {waterlevel_df.shape}")
        
        # 2. 为每个gauge_id创建合并后的数据
        merged_results = []
        
        gauge_ids = list(set(flow_df.columns) | set(waterlevel_df.columns))
        print(f"\n  处理 {len(gauge_ids)} 个站点...")
        
        for gauge_id in tqdm(gauge_ids, desc="合并数据"):
            gauge_id_str = str(gauge_id)
            
            # 获取该站点的CAMELSH数据
            flow_series = flow_df[gauge_id] if gauge_id in flow_df.columns else pd.Series(index=flow_df.index)
            wl_series = waterlevel_df[gauge_id] if gauge_id in waterlevel_df.columns else pd.Series(index=waterlevel_df.index)
            
            # 获取该站点的qualifiers数据
            if gauge_id_str in qualifiers_data:
                q_data = qualifiers_data[gauge_id_str]['discharge']
                h_data = qualifiers_data[gauge_id_str]['gage_height']
            else:
                q_data = pd.DataFrame(columns=['datetime', 'value', 'qualifiers'])
                h_data = pd.DataFrame(columns=['datetime', 'value', 'qualifiers'])
            
            # 创建合并DataFrame
            # 使用CAMELSH的时间索引作为基准
            time_index = pd.DatetimeIndex(sorted(set(flow_series.index) | set(wl_series.index)))
            
            result_df = pd.DataFrame(index=time_index)
            result_df['gauge_id'] = gauge_id_str
            
            # 添加CAMELSH数据
            result_df['Q'] = flow_series.reindex(time_index)
            result_df['H'] = wl_series.reindex(time_index)
            
            # 添加qualifiers（通过时间对齐）
            if not q_data.empty:
                q_data_indexed = q_data.set_index('datetime')
                result_df['Q_flag'] = q_data_indexed['qualifiers'].reindex(time_index, fill_value='missing')
            else:
                result_df['Q_flag'] = 'missing'
            
            if not h_data.empty:
                h_data_indexed = h_data.set_index('datetime')
                result_df['H_flag'] = h_data_indexed['qualifiers'].reindex(time_index, fill_value='missing')
            else:
                result_df['H_flag'] = 'missing'
            
            # 添加权重（可选）
            if add_weights:
                result_df['Q_weight'] = result_df['Q_flag'].apply(self._qualifier_to_weight)
                result_df['H_weight'] = result_df['H_flag'].apply(self._qualifier_to_weight)
            
            merged_results.append(result_df)
        
        # 3. 合并所有站点的数据
        print("\n  合并所有站点数据...")
        final_df = pd.concat(merged_results, ignore_index=False)
        final_df = final_df.reset_index().rename(columns={'index': 'datetime'})
        
        # 4. 保存结果
        if output_file is None:
            output_file = os.path.join(self.output_dir, "camelsh_with_qualifiers.csv")
        
        print(f"\n  保存到: {output_file}")
        final_df.to_csv(output_file, index=False)
        
        # 5. 生成统计报告
        self._generate_report(final_df, qualifiers_data)
        
        return final_df

    def merge_with_camelsh_dataset(
        self,
        camelsh_data_path: str,
        gauge_ids: List[str],
        time_range: List[str],
        qualifiers_data: Dict[str, Dict[str, pd.DataFrame]],
        output_file: str = None,
        add_weights: bool = True,
        basin_time_ranges: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
        align_mode: str = "nearest",
        align_tolerance: str = "1H",
        align_direction: str = "nearest",
    ) -> pd.DataFrame:
        """
        Merge qualifiers with CAMELSH data by reading the CAMELSH dataset directly
        (same data source approach as `multi_task_lstm.py`).

        Parameters
        ----------
        camelsh_data_path : str
            CAMELSH dataset root path.
        gauge_ids : List[str]
            Gauge IDs to load and merge.
        time_range : List[str]
            [start_date, end_date] in YYYY-MM-DD format.
        qualifiers_data : Dict
            {gauge_id: {'discharge': df, 'gage_height': df}}
        output_file : str, optional
            Output CSV path.
        add_weights : bool
            Whether to add Q_weight/H_weight.
        basin_time_ranges : Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]
            If provided, each basin will be sliced to its own [start, end] before merging.

        Returns
        -------
        pd.DataFrame
            Merged dataframe (datetime, gauge_id, Q, H, flags, weights).
        """
        print("\n开始合并qualifiers与CAMELSH数据（直接读取CAMELSH数据集）...")
        print(f"  CAMELSH路径: {camelsh_data_path}")
        print(f"  站点数: {len(gauge_ids)}")
        print(f"  时间范围: {time_range}")
        print(f"  Qualifiers对齐: mode={align_mode}, tolerance={align_tolerance}, direction={align_direction}")

        flow_df, waterlevel_df = self._load_camelsh_from_dataset(
            camelsh_data_path=camelsh_data_path,
            gauge_ids=[str(g) for g in gauge_ids],
            time_range=time_range,
        )

        print(f"  径流数据形状: {flow_df.shape}")
        print(f"  水位数据形状: {waterlevel_df.shape}")

        merged_results = []
        # Use requested gauge_ids order for predictable output
        gauge_ids_str = [str(g) for g in gauge_ids]
        print(f"\n  处理 {len(gauge_ids_str)} 个站点...")

        for gauge_id_str in tqdm(gauge_ids_str, desc="合并数据"):
            # Get CAMELSH series (may be missing column)
            flow_series = (
                flow_df[gauge_id_str] if gauge_id_str in flow_df.columns else pd.Series(index=flow_df.index, dtype=float)
            )
            wl_series = (
                waterlevel_df[gauge_id_str] if gauge_id_str in waterlevel_df.columns else pd.Series(index=waterlevel_df.index, dtype=float)
            )

            # Optionally slice per-basin time range to reduce output size and match per-basin configs
            if basin_time_ranges and gauge_id_str in basin_time_ranges:
                start_ts, end_ts = basin_time_ranges[gauge_id_str]
                flow_series = flow_series.loc[start_ts:end_ts]
                wl_series = wl_series.loc[start_ts:end_ts]

            # Get qualifiers for this gauge
            if gauge_id_str in qualifiers_data:
                q_data = qualifiers_data[gauge_id_str].get("discharge", pd.DataFrame())
                h_data = qualifiers_data[gauge_id_str].get("gage_height", pd.DataFrame())
            else:
                q_data = pd.DataFrame(columns=["datetime", "value", "qualifiers"])
                h_data = pd.DataFrame(columns=["datetime", "value", "qualifiers"])

            time_index = pd.DatetimeIndex(sorted(set(flow_series.index) | set(wl_series.index)))

            result_df = pd.DataFrame(index=time_index)
            result_df["gauge_id"] = gauge_id_str
            result_df["Q"] = flow_series.reindex(time_index)
            result_df["H"] = wl_series.reindex(time_index)

            # Align qualifiers onto CAMELSH time index (exact match can produce lots of 'missing'
            # because USGS IV is often 15-min resolution while CAMELSH targets can be hourly).
            result_df["Q_flag"] = self._align_qualifiers_to_time_index(
                q_data,
                time_index,
                align_mode=align_mode,
                tolerance=align_tolerance,
                direction=align_direction,
            ).reindex(time_index, fill_value="missing")

            result_df["H_flag"] = self._align_qualifiers_to_time_index(
                h_data,
                time_index,
                align_mode=align_mode,
                tolerance=align_tolerance,
                direction=align_direction,
            ).reindex(time_index, fill_value="missing")

            if add_weights:
                result_df["Q_weight"] = result_df["Q_flag"].apply(self._qualifier_to_weight)
                result_df["H_weight"] = result_df["H_flag"].apply(self._qualifier_to_weight)

            merged_results.append(result_df)

        print("\n  合并所有站点数据...")
        final_df = pd.concat(merged_results, ignore_index=False)
        final_df = final_df.reset_index().rename(columns={"index": "datetime"})

        if output_file is None:
            output_file = os.path.join(self.output_dir, "camelsh_with_qualifiers.csv")

        print(f"\n  保存到: {output_file}")
        final_df.to_csv(output_file, index=False)

        self._generate_report(final_df, qualifiers_data)
        return final_df
    
    def _qualifier_to_weight(self, qualifier_str: str) -> float:
        """
        将qualifier转换为数据权重
        
        Parameters
        ----------
        qualifier_str : str
            Qualifier字符串
        
        Returns
        -------
        float
            权重值（0-1）
        """
        if qualifier_str == 'missing' or pd.isna(qualifier_str):
            return 0.0
        
        if qualifier_str == 'none':
            return 1.0
        
        # 根据qualifier类型设置权重
        qualifiers = qualifier_str.split(',')
        
        # 权重规则
        weight = 1.0
        
        for q in qualifiers:
            q = q.strip()
            if q == 'A':  # Approved
                weight *= 1.0
            elif q == 'P':  # Provisional
                weight *= 0.9
            elif q == 'e':  # Estimated
                weight *= 0.7
            elif q in ['<', '>']:  # 不确定范围
                weight *= 0.6
            elif q in ['Ice', 'Bkw', 'Eqp']:  # 受影响的测量
                weight *= 0.5
            else:
                weight *= 0.8  # 其他未知qualifier
        
        return max(weight, 0.0)
    
    def _generate_report(self, merged_df: pd.DataFrame, qualifiers_data: Dict):
        """生成统计报告"""
        report_file = os.path.join(self.output_dir, "qualifiers_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("USGS Qualifiers 数据合并报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本统计
            f.write("1. 数据概览\n")
            f.write("-" * 40 + "\n")
            f.write(f"总记录数: {len(merged_df)}\n")
            f.write(f"站点数: {merged_df['gauge_id'].nunique()}\n")
            f.write(f"时间范围: {merged_df['datetime'].min()} 到 {merged_df['datetime'].max()}\n\n")
            
            # Qualifier统计
            f.write("2. Qualifier统计\n")
            f.write("-" * 40 + "\n")
            
            f.write("径流 (Q) qualifiers分布:\n")
            q_flag_counts = merged_df['Q_flag'].value_counts()
            for flag, count in q_flag_counts.head(10).items():
                pct = count / len(merged_df) * 100
                f.write(f"  {flag}: {count} ({pct:.2f}%)\n")
            
            f.write("\n水位 (H) qualifiers分布:\n")
            h_flag_counts = merged_df['H_flag'].value_counts()
            for flag, count in h_flag_counts.head(10).items():
                pct = count / len(merged_df) * 100
                f.write(f"  {flag}: {count} ({pct:.2f}%)\n")
            
            # 权重统计（如果存在）
            if 'Q_weight' in merged_df.columns:
                f.write("\n3. 权重统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"径流权重平均: {merged_df['Q_weight'].mean():.3f}\n")
                f.write(f"水位权重平均: {merged_df['H_weight'].mean():.3f}\n")
            
            # 数据完整性
            f.write("\n4. 数据完整性\n")
            f.write("-" * 40 + "\n")
            q_missing = (merged_df['Q_flag'] == 'missing').sum()
            h_missing = (merged_df['H_flag'] == 'missing').sum()
            f.write(f"径流缺失qualifiers: {q_missing} ({q_missing/len(merged_df)*100:.2f}%)\n")
            f.write(f"水位缺失qualifiers: {h_missing} ({h_missing/len(merged_df)*100:.2f}%)\n")
            
            # Qualifier含义说明
            f.write("\n5. Qualifier含义说明\n")
            f.write("-" * 40 + "\n")
            for code, meaning in self.QUALIFIER_MEANINGS.items():
                f.write(f"  {code}: {meaning}\n")
        
        print(f"  报告已保存: {report_file}")


def main():
    """主函数 - 示例用法"""
    
    print("\n" + "!" * 80)
    print("⚠️  重要提示")
    print("!" * 80)
    print("这是示例程序，用于演示基本用法。")
    print("实际使用时，请运行 run_quick.py（自动配置）或根据需要修改配置。")
    print()
    print("注意事项：")
    print("1. USGS API单次请求不要超过1年数据，否则会超时")
    print("2. 需要先运行 export_camelsh_data.py 导出CAMELSH数据")
    print("3. 建议先测试少量站点和短时间范围")
    print("!" * 80 + "\n")
    
    # ==================== 配置参数 ====================
    
    # 1. 站点ID列表（根据你的CAMELSH数据调整）
    GAUGE_IDS = [
        "01646500",  # 示例站点
        "01434000",
        # 添加更多站点...
    ]
    
    # 2. 时间范围（⚠️ 建议不超过1年，避免API超时）
    # 如需长时间范围，请分批请求或使用 run_quick.py
    START_DATE = "2023-01-01"
    END_DATE = "2023-12-31"
    
    print(f"⚠️  当前配置:")
    print(f"  站点数: {len(GAUGE_IDS)}")
    print(f"  时间范围: {START_DATE} 到 {END_DATE}")
    print(f"  提示: 如需长时间范围，请分批请求\n")
    
    # 3. CAMELSH数据文件路径（根据实际路径调整）
    # 使用导出的文件路径
    CAMELSH_FLOW_FILE = "camelsh_exported/flow_hourly.csv"
    CAMELSH_WATERLEVEL_FILE = "camelsh_exported/waterlevel_hourly.csv"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(CAMELSH_FLOW_FILE):
        print("!" * 80)
        print("❌ 错误: CAMELSH数据文件不存在")
        print("!" * 80)
        print(f"找不到文件: {CAMELSH_FLOW_FILE}")
        print(f"找不到文件: {CAMELSH_WATERLEVEL_FILE}")
        print()
        print("解决方案：")
        print("1. 运行: uv run python export_camelsh_data.py")
        print("   这会从CAMELSH数据集导出所需的CSV文件")
        print()
        print("2. 或者手动准备CSV文件（格式：index=时间，columns=gauge_id）")
        print("!" * 80)
        return
    
    # 4. 输出配置
    OUTPUT_DIR = "qualifiers_output"
    CACHE_DIR = "qualifiers_cache"
    
    # ==================== 执行流程 ====================
    
    print("=" * 80)
    print("USGS Qualifiers Fetcher")
    print("=" * 80)
    
    # 初始化fetcher
    fetcher = USGSQualifiersFetcher(
        output_dir=OUTPUT_DIR,
        cache_dir=CACHE_DIR
    )
    
    # Step 1: 获取qualifiers
    print("\n" + "=" * 80)
    print("Step 1: 从USGS NWIS获取qualifiers")
    print("=" * 80)
    
    qualifiers_data = fetcher.fetch_multiple_gauges(
        gauge_ids=GAUGE_IDS,
        start_date=START_DATE,
        end_date=END_DATE,
        use_cache=True,  # 使用缓存以避免重复请求
        delay=0.5  # 请求间隔（秒）
    )
    
    # Step 2: 与CAMELSH数据合并
    print("\n" + "=" * 80)
    print("Step 2: 与CAMELSH数据合并")
    print("=" * 80)
    
    merged_df = fetcher.merge_with_camelsh(
        camelsh_flow_file=CAMELSH_FLOW_FILE,
        camelsh_waterlevel_file=CAMELSH_WATERLEVEL_FILE,
        qualifiers_data=qualifiers_data,
        output_file=None,  # 使用默认输出路径
        add_weights=True  # 添加权重列
    )
    
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - 合并数据: {OUTPUT_DIR}/camelsh_with_qualifiers.csv")
    print(f"  - 统计报告: {OUTPUT_DIR}/qualifiers_report.txt")
    print(f"  - 缓存目录: {CACHE_DIR}/")


if __name__ == "__main__":
    main()

