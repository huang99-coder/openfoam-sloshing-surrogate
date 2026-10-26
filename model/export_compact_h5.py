#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一批原始 HDF5 中导出轻量载荷包：
  1) compact_all_loads.npz
  2) h5_manifest.csv
  3) export_report.txt

特点
----
- 只读取时间、六分量载荷和少量元数据，不读取庞大的场快照。
- 自动递归搜索 *.h5 / *.hdf5。
- 自动识别常见的 time / Fx,Fy,Fz / Mx,My,Mz 数据集命名。
- 支持 6 列合并载荷表、3 列力表和 3 列力矩表。
- 支持用命令行明确指定统一的数据集路径。
- 每条时程线性重采样到固定点数，输出通常只有数 MB。

推荐先直接运行：
  python export_compact_h5.py --input-dir all_h5_data --output-dir compact_export --points 200

如果自动识别失败，先导出结构报告：
  python export_compact_h5.py --input-dir all_h5_data --output-dir compact_export --inspect-only --inspect-count 3

再根据结构报告明确指定路径，例如：
  python export_compact_h5.py --input-dir all_h5_data --output-dir compact_export \
      --time-path loads/time \
      --fx-path loads/Fx --fy-path loads/Fy --fz-path loads/Fz \
      --mx-path loads/Mx --my-path loads/My --mz-path loads/Mz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

# 规范化后用于匹配的数据集 basename 别名。
ALIASES: Dict[str, Tuple[str, ...]] = {
    "time": (
        "time", "times", "t", "simulationtime", "physicaltime", "loadtime",
        "forcetime", "momenttime", "timestamps",
    ),
    "Fx": ("fx", "forcex", "xforce", "totalforcex", "wallforcex", "loadfx", "force_x"),
    "Fy": ("fy", "forcey", "yforce", "totalforcey", "wallforcey", "loadfy", "force_y"),
    "Fz": ("fz", "forcez", "zforce", "totalforcez", "wallforcez", "loadfz", "force_z"),
    "Mx": ("mx", "momentx", "xmoment", "totalmomentx", "wallmomentx", "loadmx", "moment_x"),
    "My": ("my", "momenty", "ymoment", "totalmomenty", "wallmomenty", "loadmy", "moment_y"),
    "Mz": ("mz", "momentz", "zmoment", "totalmomentz", "wallmomentz", "loadmz", "moment_z"),
}

FORCE_TABLE_ALIASES = (
    "force", "forces", "totalforce", "totalforces", "wallforce", "wallforces",
    "resultantforce", "resultantforces", "forcevector", "forceseries",
)
MOMENT_TABLE_ALIASES = (
    "moment", "moments", "totalmoment", "totalmoments", "wallmoment", "wallmoments",
    "resultantmoment", "resultantmoments", "momentvector", "momentseries", "torque", "torques",
)
LOAD_TABLE_ALIASES = (
    "load", "loads", "wallloads", "totalloads", "forcemoment", "forcesmoments",
    "loadhistory", "loadseries", "resultantloads",
)

PARAM_KEYS = (
    "case_name", "case_type", "fill_level", "fill_pct", "density", "viscosity",
    "liquid_type", "fluid", "deceleration", "decel_g", "lateral_acceleration",
    "radius", "speed_kmh", "angle", "angle_deg", "direction", "t_change",
    "lane_change_time", "is_double_overtake", "is_double", "total_time", "peak_gx",
)


def norm_name(value: str) -> str:
    """删除路径和名字里的非字母数字字符，统一小写。"""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def to_python_scalar(value: Any) -> Any:
    """把 HDF5/numpy 标量转成 JSON 可序列化值。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_python_scalar(value.item())
        if value.size <= 20:
            return [to_python_scalar(v) for v in value.tolist()]
        return f"<array shape={value.shape}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if np.isfinite(v) else None


def collect_scalar_attrs(h5: h5py.File) -> Dict[str, Any]:
    """收集根、group、dataset 的小型 attrs；保存原键和 attr/规范键。"""
    out: Dict[str, Any] = {}

    def add_attrs(prefix: str, attrs: h5py.AttributeManager) -> None:
        for key, value in attrs.items():
            py = to_python_scalar(value)
            if isinstance(py, str) and py.startswith("<array shape="):
                continue
            full_key = f"attr/{prefix}/{key}" if prefix else f"attr/{key}"
            out[full_key] = py
            # 同名参数的扁平别名，后写不覆盖前写。
            nk = norm_name(key)
            for std in PARAM_KEYS:
                if nk == norm_name(std):
                    out.setdefault(std, py)

    add_attrs("", h5.attrs)

    def visitor(name: str, obj: Any) -> None:
        add_attrs(name, obj.attrs)
        # parameters 下的小型标量 dataset 也作为参数读取。
        if isinstance(obj, h5py.Dataset) and name.lower().startswith("parameters/"):
            if obj.size <= 20:
                try:
                    value = to_python_scalar(obj[()])
                except Exception:
                    return
                key = name.split("/")[-1]
                out[f"attr/{name}"] = value
                out.setdefault(key, value)

    h5.visititems(visitor)
    return out


def detect_case_type(filename: str, params: Mapping[str, Any]) -> str:
    for key in ("case_type", "attr/case_type", "type"):
        if key in params:
            value = str(params[key]).lower()
            if "brak" in value or value in {"0", "0.0"}:
                return "braking"
            if "turn" in value or value in {"1", "1.0"}:
                return "turning"
            if "over" in value or "lane" in value or value in {"2", "2.0"}:
                return "overtaking"
            if "slope" in value or "uphill" in value or "downhill" in value or value in {"3", "3.0"}:
                return "slope"

    base = filename.lower()
    if re.search(r"brak|decel|(^|[_-])br([_-]|$)", base):
        return "braking"
    if re.search(r"turn|(^|[_-])tr([_-]|$)|r\d+.*(?:km|kph)", base):
        return "turning"
    if re.search(r"overtak|lane|double|(^|[_-])ot([_-]|$)", base):
        return "overtaking"
    if re.search(r"slope|uphill|downhill|angle|(^|[_-])sl([_-]|$)", base):
        return "slope"
    return "unknown"


def infer_params(path: Path, attrs: Dict[str, Any], case_type: str) -> Dict[str, Any]:
    """保留 attrs，并从文件名补充绘图脚本需要的标准字段。"""
    p: Dict[str, Any] = dict(attrs)
    stem = path.stem
    low = stem.lower()

    # 找到可能的 case_name / liquid / direction。
    p.setdefault("attr/case_name", p.get("case_name", stem))
    p.setdefault("case_name", p.get("attr/case_name", stem))
    p.setdefault("case_type", case_type)

    liquid = p.get("liquid_type", p.get("fluid", None))
    if liquid is None:
        liquid = "oil" if ("oil" in low or "diesel" in low) else "water"
    liquid = str(liquid).lower()
    if "diesel" in liquid:
        liquid = "oil"
    p.setdefault("attr/liquid_type", liquid)
    p.setdefault("liquid_type", liquid)

    # 充装率。
    fill = safe_float(p.get("fill_level"))
    if fill is None:
        fill_pct = safe_float(p.get("fill_pct"))
        if fill_pct is not None:
            fill = fill_pct / 100.0 if fill_pct > 1 else fill_pct
    if fill is None:
        m = re.search(r"(?:fill)?(\d{2,3})p(?:ct)?", low)
        if not m:
            m = re.search(r"fill[_-]?(\d{2,3})", low)
        if m:
            fill = float(m.group(1)) / 100.0
    if fill is not None:
        if fill > 1.5:
            fill /= 100.0
        p["fill_level"] = fill

    # 密度/黏度兜底。
    if safe_float(p.get("density")) is None:
        p["density"] = 850.0 if liquid == "oil" else 1000.0
    if safe_float(p.get("viscosity")) is None:
        p["viscosity"] = 3.5e-6 if liquid == "oil" else 1.0e-6

    if case_type == "braking":
        decel = safe_float(p.get("deceleration"))
        decel_g = safe_float(p.get("decel_g"))
        if decel is None:
            m = re.search(r"(?:brake|decel)?[_-]?(0?[.]?\d+)g", low)
            if m:
                gval = float(m.group(1))
                if gval > 1.5:
                    gval /= 10.0
                decel_g = gval
                decel = gval * 9.81
        if decel is not None:
            p["deceleration"] = decel
        if decel_g is None and decel is not None:
            decel_g = decel / 9.81
        if decel_g is not None:
            p["decel_g"] = decel_g
            # 与旧绘图脚本中的命名兼容。
            if "case_name" not in attrs and p.get("fill_level") is not None:
                p["attr/case_name"] = f"brake{decel_g:g}g_{int(round(100*p['fill_level']))}p_{liquid}"

    elif case_type == "turning":
        radius = safe_float(p.get("radius"))
        speed = safe_float(p.get("speed_kmh"))
        if radius is None:
            m = re.search(r"(?:^|[_-])r(\d+(?:[.]\d+)?)", low)
            if m:
                radius = float(m.group(1))
        if speed is None:
            m = re.search(r"(\d+(?:[.]\d+)?)(?:kmh|kmph|kph|km)", low)
            if m:
                speed = float(m.group(1))
        if radius is not None:
            p["radius"] = radius
        if speed is not None:
            p["speed_kmh"] = speed
        alat = safe_float(p.get("lateral_acceleration"))
        if alat is None and radius and speed is not None:
            alat = (speed / 3.6) ** 2 / radius
        if alat is not None:
            p["lateral_acceleration"] = alat

    elif case_type == "overtaking":
        t_change = safe_float(p.get("t_change"))
        if t_change is None:
            t_change = safe_float(p.get("lane_change_time"))
        if t_change is None:
            m = re.search(r"(?:t|change)[_-]?(\d+(?:[.]\d+)?)s", low)
            if m:
                t_change = float(m.group(1))
        if t_change is not None:
            p["t_change"] = t_change
            p.setdefault("lane_change_time", t_change)
        is_double = p.get("is_double_overtake", p.get("is_double", None))
        if is_double is None:
            is_double = int("double" in low or "dual" in low)
        try:
            is_double = int(float(is_double) > 0.5)
        except Exception:
            is_double = int(str(is_double).lower() in {"true", "yes", "double"})
        p["is_double_overtake"] = is_double
        total_time = safe_float(p.get("total_time"))
        if total_time is not None:
            p["total_time"] = total_time

    elif case_type == "slope":
        angle = safe_float(p.get("angle"))
        if angle is None:
            angle = safe_float(p.get("angle_deg"))
        if angle is None:
            m = re.search(r"(?:angle)?[_-]?(\d+(?:[.]\d+)?)(?:deg|degree)", low)
            if m:
                angle = float(m.group(1))
        if angle is not None:
            p["angle"] = angle
            p.setdefault("angle_deg", angle)
        direction = p.get("direction", None)
        if direction is None:
            direction = "downhill" if "down" in low else "uphill"
        if isinstance(direction, (int, float, np.number)):
            direction = "downhill" if float(direction) > 0.5 else "uphill"
        direction = str(direction).lower()
        if direction in {"down", "1", "1.0"}:
            direction = "downhill"
        if direction in {"up", "0", "0.0"}:
            direction = "uphill"
        p["attr/direction"] = direction
        p["direction"] = direction

    # 去掉不可序列化或过大的值。
    clean: Dict[str, Any] = {}
    for key, value in p.items():
        py = to_python_scalar(value)
        if isinstance(py, str) and py.startswith("<array shape="):
            continue
        clean[str(key)] = py
    return clean


@dataclass
class DatasetInfo:
    path: str
    shape: Tuple[int, ...]
    dtype: str
    basename_norm: str


def list_datasets(h5: h5py.File) -> List[DatasetInfo]:
    out: List[DatasetInfo] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out.append(DatasetInfo(
                path=name,
                shape=tuple(int(v) for v in obj.shape),
                dtype=str(obj.dtype),
                basename_norm=norm_name(name.split("/")[-1]),
            ))

    h5.visititems(visitor)
    return out


def format_structure(path: Path, max_attrs: int = 30) -> str:
    lines = [f"FILE: {path}"]
    with h5py.File(path, "r") as h5:
        attrs = collect_scalar_attrs(h5)
        for key in sorted(attrs)[:max_attrs]:
            lines.append(f"  ATTR {key} = {attrs[key]}")
        lines.append("  DATASETS:")
        for info in list_datasets(h5):
            lines.append(f"    {info.path}  shape={info.shape}  dtype={info.dtype}")
    return "\n".join(lines)


def read_1d_dataset(ds: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(ds[()])
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"不是一维数组: shape={arr.shape}")
    return arr.astype(float, copy=False)


def candidate_score(info: DatasetInfo, aliases: Sequence[str], expected_length: Optional[int] = None) -> int:
    score = 0
    if info.basename_norm in {norm_name(a) for a in aliases}:
        score += 100
    path_norm = norm_name(info.path)
    for alias in aliases:
        na = norm_name(alias)
        if na and na in path_norm:
            score += 20
    # 只青睐时程类的小维度数组，避开快照场。
    if len(info.shape) == 1:
        score += 20
    elif len(info.shape) == 2 and 1 in info.shape:
        score += 10
    if expected_length is not None and expected_length in info.shape:
        score += 40
    if any(v > 10_000_000 for v in info.shape):
        score -= 100
    return score


def choose_named_dataset(
    h5: h5py.File,
    infos: Sequence[DatasetInfo],
    key: str,
    explicit_path: Optional[str] = None,
    expected_length: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if explicit_path:
        if explicit_path not in h5:
            raise KeyError(f"指定路径不存在: {explicit_path}")
        return read_1d_dataset(h5[explicit_path]), explicit_path

    candidates: List[Tuple[int, DatasetInfo]] = []
    for info in infos:
        if len(info.shape) > 2:
            continue
        if len(info.shape) == 2 and min(info.shape) != 1:
            continue
        score = candidate_score(info, ALIASES[key], expected_length)
        if score >= 100:
            candidates.append((score, info))
    candidates.sort(key=lambda item: (-item[0], len(item[1].path), item[1].path))
    for _, info in candidates:
        try:
            arr = read_1d_dataset(h5[info.path])
            if arr.size >= 2:
                return arr, info.path
        except Exception:
            continue
    return None, None


def parse_column_names(ds: h5py.Dataset, ncols: int) -> Optional[List[str]]:
    for key in ("columns", "column_names", "channels", "channel_names", "components", "labels"):
        if key not in ds.attrs:
            continue
        raw = to_python_scalar(ds.attrs[key])
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    raw = parsed
                else:
                    raw = re.split(r"[,;\s]+", raw.strip())
            except Exception:
                raw = re.split(r"[,;\s]+", raw.strip())
        if isinstance(raw, list) and len(raw) == ncols:
            return [str(v) for v in raw]
    return None


def orient_table(arr: np.ndarray, ncols: int) -> Optional[np.ndarray]:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    if arr.shape[1] == ncols:
        return arr.astype(float, copy=False)
    if arr.shape[0] == ncols:
        return arr.T.astype(float, copy=False)
    return None


def find_table_dataset(
    h5: h5py.File,
    infos: Sequence[DatasetInfo],
    aliases: Sequence[str],
    ncols_options: Sequence[int],
    explicit_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[List[str]]]:
    paths: List[str] = []
    if explicit_path:
        paths = [explicit_path]
    else:
        alias_norm = {norm_name(a) for a in aliases}
        scored: List[Tuple[int, DatasetInfo]] = []
        for info in infos:
            if len(info.shape) not in (2, 3):
                continue
            if not any(v in ncols_options for v in info.shape):
                continue
            score = 100 if info.basename_norm in alias_norm else 0
            path_norm = norm_name(info.path)
            if any(a in path_norm for a in alias_norm):
                score += 30
            if score:
                scored.append((score, info))
        scored.sort(key=lambda item: (-item[0], len(item[1].path), item[1].path))
        paths = [info.path for _, info in scored]

    for ds_path in paths:
        if ds_path not in h5:
            if explicit_path:
                raise KeyError(f"指定路径不存在: {ds_path}")
            continue
        ds = h5[ds_path]
        arr = np.asarray(ds[()])
        arr = np.squeeze(arr)
        # 兼容 OpenFOAM 风格 (N, contribution, xyz) 或 (N, xyz, contribution)，把贡献项求和。
        if arr.ndim == 3:
            dims3 = [i for i, v in enumerate(arr.shape) if v == 3]
            if not dims3:
                continue
            coord_axis = dims3[-1]
            arr = np.moveaxis(arr, coord_axis, -1)
            # 剩下两轴中，较大者视为时间，较小者视为 pressure/viscous/porous 等贡献。
            if arr.shape[0] < arr.shape[1]:
                arr = np.swapaxes(arr, 0, 1)
            arr = arr.sum(axis=1)
        for ncols in ncols_options:
            table = orient_table(arr, ncols)
            if table is not None:
                cols = parse_column_names(ds, ncols)
                return table, ds_path, cols
    return None, None, None


def map_table_columns(table: np.ndarray, columns: Optional[Sequence[str]], wanted: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if columns:
        normalized = {norm_name(name): idx for idx, name in enumerate(columns)}
        for key in wanted:
            aliases = [norm_name(key)] + [norm_name(a) for a in ALIASES.get(key, ())]
            for alias in aliases:
                if alias in normalized:
                    out[key] = table[:, normalized[alias]].astype(float, copy=False)
                    break
    if not out:
        # 没有列名时，仅在列数与预期完全一致时按 x,y,z / Fx..Mz 假设。
        if table.shape[1] == len(wanted):
            out = {key: table[:, i].astype(float, copy=False) for i, key in enumerate(wanted)}
    return out


def extract_loads(
    h5: h5py.File,
    infos: Sequence[DatasetInfo],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, str], List[str]]:
    warnings: List[str] = []
    paths_used: Dict[str, str] = {}

    # 1) 明确指定/自动识别的独立通道。
    series: Dict[str, np.ndarray] = {}
    for key in CHANNELS:
        arr, ds_path = choose_named_dataset(
            h5, infos, key,
            explicit_path=getattr(args, f"{key.lower()}_path"),
        )
        if arr is not None:
            series[key] = arr
            paths_used[key] = ds_path or ""

    # 2) 六列合并表（或 7 列：time + 6 load）。
    if len(series) < 6:
        load_table, ds_path, cols = find_table_dataset(
            h5, infos, LOAD_TABLE_ALIASES, (6, 7), explicit_path=args.loads_path
        )
        if load_table is not None:
            if load_table.shape[1] == 7:
                # 有列名则按列名；无列名默认第一列为 time。
                if cols:
                    mapped = map_table_columns(load_table, cols, ("time",) + CHANNELS)
                    for key in CHANNELS:
                        if key in mapped and key not in series:
                            series[key] = mapped[key]
                            paths_used[key] = f"{ds_path}[{key}]"
                else:
                    for i, key in enumerate(CHANNELS, start=1):
                        if key not in series:
                            series[key] = load_table[:, i]
                            paths_used[key] = f"{ds_path}[col{i}]"
                    warnings.append(f"{ds_path} 无列名，按 [time,Fx,Fy,Fz,Mx,My,Mz] 假设")
            else:
                mapped = map_table_columns(load_table, cols, CHANNELS)
                for key, arr in mapped.items():
                    if key not in series:
                        series[key] = arr
                        paths_used[key] = f"{ds_path}[{key}]"
                if not cols:
                    warnings.append(f"{ds_path} 无列名，按 [Fx,Fy,Fz,Mx,My,Mz] 假设")

    # 3) 三列力/力矩表。
    if not all(k in series for k in ("Fx", "Fy", "Fz")):
        table, ds_path, cols = find_table_dataset(
            h5, infos, FORCE_TABLE_ALIASES, (3, 4), explicit_path=args.force_path
        )
        if table is not None:
            if table.shape[1] == 4 and not cols:
                table = table[:, 1:]
                warnings.append(f"{ds_path} 无列名，按 [time,Fx,Fy,Fz] 假设")
            mapped = map_table_columns(table, cols, ("Fx", "Fy", "Fz"))
            for key, arr in mapped.items():
                if key not in series:
                    series[key] = arr
                    paths_used[key] = f"{ds_path}[{key}]"
            if not cols and table.shape[1] == 3:
                warnings.append(f"{ds_path} 无列名，按 [Fx,Fy,Fz] 假设")

    if not all(k in series for k in ("Mx", "My", "Mz")):
        table, ds_path, cols = find_table_dataset(
            h5, infos, MOMENT_TABLE_ALIASES, (3, 4), explicit_path=args.moment_path
        )
        if table is not None:
            if table.shape[1] == 4 and not cols:
                table = table[:, 1:]
                warnings.append(f"{ds_path} 无列名，按 [time,Mx,My,Mz] 假设")
            mapped = map_table_columns(table, cols, ("Mx", "My", "Mz"))
            for key, arr in mapped.items():
                if key not in series:
                    series[key] = arr
                    paths_used[key] = f"{ds_path}[{key}]"
            if not cols and table.shape[1] == 3:
                warnings.append(f"{ds_path} 无列名，按 [Mx,My,Mz] 假设")

    missing = [key for key in CHANNELS if key not in series]
    if missing:
        raise KeyError("缺少载荷通道: " + ", ".join(missing))

    lengths = [series[key].size for key in CHANNELS]
    n = min(lengths)
    if len(set(lengths)) > 1:
        warnings.append(f"六通道长度不一致 {dict(zip(CHANNELS, lengths))}，统一截取前 {n} 点")
    loads = np.column_stack([series[key][:n] for key in CHANNELS]).astype(np.float64)
    return loads, paths_used, warnings


def extract_time(
    h5: h5py.File,
    infos: Sequence[DatasetInfo],
    n_load: int,
    args: argparse.Namespace,
    params: Mapping[str, Any],
) -> Tuple[np.ndarray, Optional[str], List[str]]:
    warnings: List[str] = []
    arr, ds_path = choose_named_dataset(
        h5, infos, "time", explicit_path=args.time_path, expected_length=n_load
    )
    if arr is not None:
        if arr.size != n_load:
            if arr.size > n_load:
                warnings.append(f"时间长度 {arr.size} > 载荷长度 {n_load}，截取前 {n_load} 点")
                arr = arr[:n_load]
            else:
                warnings.append(f"时间长度 {arr.size} < 载荷长度 {n_load}，按起止时刻重建")
                t0, t1 = float(arr[0]), float(arr[-1])
                arr = np.linspace(t0, t1, n_load)
        return arr.astype(float), ds_path, warnings

    # 没有 time dataset：从元数据总时长兜底，否则用样本序号。
    t_start = safe_float(params.get("t_start"))
    t_end = safe_float(params.get("t_end"))
    total_time = safe_float(params.get("total_time"))
    if t_start is None:
        t_start = 0.0
    if t_end is None and total_time is not None:
        t_end = t_start + total_time
    if t_end is None:
        t_end = float(n_load - 1)
        warnings.append("未找到时间数据集或总时长，使用样本序号作为时间")
    else:
        warnings.append("未找到时间数据集，按元数据起止时刻线性重建")
    return np.linspace(t_start, t_end, n_load), None, warnings


def sanitize_time_and_loads(time: np.ndarray, loads: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    warnings: List[str] = []
    time = np.asarray(time, dtype=float).reshape(-1)
    loads = np.asarray(loads, dtype=float)
    n = min(time.size, loads.shape[0])
    time, loads = time[:n], loads[:n]

    finite_t = np.isfinite(time)
    if finite_t.sum() < 2:
        raise ValueError("有效时间点不足 2 个")
    if not np.all(finite_t):
        warnings.append(f"删除 {np.sum(~finite_t)} 个非有限时间点")
        time, loads = time[finite_t], loads[finite_t]

    order = np.argsort(time, kind="stable")
    time, loads = time[order], loads[order]

    # 重复时间点取最后一条，保证 np.interp 的 x 严格递增。
    _, unique_indices_rev = np.unique(time[::-1], return_index=True)
    keep = np.sort(time.size - 1 - unique_indices_rev)
    if keep.size != time.size:
        warnings.append(f"合并 {time.size - keep.size} 个重复时间点")
        time, loads = time[keep], loads[keep]

    if time.size < 2 or not (time[-1] > time[0]):
        raise ValueError("时间轴无有效跨度")

    # 载荷中的 NaN/Inf 逐通道线性插值；全坏通道报错。
    for j, key in enumerate(CHANNELS):
        y = loads[:, j]
        good = np.isfinite(y)
        if good.sum() < 2:
            raise ValueError(f"{key} 有效点不足 2 个")
        if not np.all(good):
            warnings.append(f"{key} 插值修复 {np.sum(~good)} 个 NaN/Inf")
            y[~good] = np.interp(time[~good], time[good], y[good])
            loads[:, j] = y
    return time, loads, warnings


def resample(time: np.ndarray, loads: np.ndarray, points: int) -> Tuple[np.ndarray, np.ndarray]:
    target = np.linspace(float(time[0]), float(time[-1]), points)
    out = np.column_stack([np.interp(target, time, loads[:, j]) for j in range(loads.shape[1])])
    return target, out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="原始 HDF5 根目录，递归搜索")
    ap.add_argument("--output-dir", default="compact_export", help="输出目录")
    ap.add_argument("--points", type=int, default=200, help="每个工况统一重采样点数")
    ap.add_argument("--inspect-only", action="store_true", help="只导出 HDF5 结构报告，不读取载荷")
    ap.add_argument("--inspect-count", type=int, default=3, help="结构报告包含前几个 HDF5")
    ap.add_argument("--skip-failed", action="store_true", help="遇到识别失败文件时跳过，而不是最终返回错误码")

    # 所有文件路径一致时可显式指定；路径不带开头 /。
    ap.add_argument("--time-path")
    ap.add_argument("--fx-path")
    ap.add_argument("--fy-path")
    ap.add_argument("--fz-path")
    ap.add_argument("--mx-path")
    ap.add_argument("--my-path")
    ap.add_argument("--mz-path")
    ap.add_argument("--loads-path", help="6列或7列合并载荷表路径")
    ap.add_argument("--force-path", help="3列或4列力表路径")
    ap.add_argument("--moment-path", help="3列或4列力矩表路径")
    args = ap.parse_args()

    if args.points < 10:
        ap.error("--points 至少为 10")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(set(input_dir.rglob("*.h5")) | set(input_dir.rglob("*.hdf5")))
    if not files:
        print(f"错误：{input_dir} 下没有找到 .h5/.hdf5", file=sys.stderr)
        return 2
    print(f"找到 {len(files)} 个 HDF5 文件")

    structure_path = output_dir / "h5_structure.txt"
    with structure_path.open("w", encoding="utf-8") as fh:
        for path in files[: max(1, args.inspect_count)]:
            try:
                fh.write(format_structure(path))
            except Exception as exc:
                fh.write(f"FILE: {path}\n  ERROR: {exc}\n")
            fh.write("\n" + "=" * 100 + "\n\n")
    print(f"结构报告：{structure_path}")

    if args.inspect_only:
        print("inspect-only 完成。请把 h5_structure.txt 发来即可，不需要上传原始 HDF5。")
        return 0

    records: List[Dict[str, Any]] = []
    load_blocks: List[np.ndarray] = []
    case_ids: List[str] = []
    case_types: List[str] = []
    params_json: List[str] = []
    t_starts: List[float] = []
    t_ends: List[float] = []
    report_lines: List[str] = []
    failed = 0

    for index, path in enumerate(files, start=1):
        rel = str(path.relative_to(input_dir))
        try:
            with h5py.File(path, "r") as h5:
                attrs = collect_scalar_attrs(h5)
                case_type = detect_case_type(path.name, attrs)
                params = infer_params(path, attrs, case_type)
                infos = list_datasets(h5)
                loads_raw, channel_paths, warn_load = extract_loads(h5, infos, args)
                time_raw, time_path, warn_time = extract_time(h5, infos, loads_raw.shape[0], args, params)

            time_raw, loads_raw, warn_clean = sanitize_time_and_loads(time_raw, loads_raw)
            target_time, loads_small = resample(time_raw, loads_raw, args.points)

            case_id = str(params.get("case_id", params.get("attr/case_id", path.stem)))
            params["source_file"] = rel
            params["case_id"] = case_id
            params["case_type"] = case_type
            params["t_start"] = float(time_raw[0])
            params["t_end"] = float(time_raw[-1])
            params["n_raw"] = int(time_raw.size)
            params["n_export"] = int(args.points)

            warnings_all = warn_load + warn_time + warn_clean
            records.append({
                "source_file": rel,
                "case_id": case_id,
                "case_type": case_type,
                "n_raw": int(time_raw.size),
                "n_export": int(args.points),
                "t_start": float(time_raw[0]),
                "t_end": float(time_raw[-1]),
                "duration": float(time_raw[-1] - time_raw[0]),
                "time_path": time_path or "<reconstructed>",
                "Fx_path": channel_paths.get("Fx", ""),
                "Fy_path": channel_paths.get("Fy", ""),
                "Fz_path": channel_paths.get("Fz", ""),
                "Mx_path": channel_paths.get("Mx", ""),
                "My_path": channel_paths.get("My", ""),
                "Mz_path": channel_paths.get("Mz", ""),
                "fill_level": params.get("fill_level", ""),
                "density": params.get("density", ""),
                "liquid_type": params.get("liquid_type", ""),
                "warnings": " | ".join(warnings_all),
                "params_json": json.dumps(params, ensure_ascii=False, separators=(",", ":")),
            })
            load_blocks.append(loads_small)
            case_ids.append(case_id)
            case_types.append(case_type)
            params_json.append(json.dumps(params, ensure_ascii=False, separators=(",", ":")))
            t_starts.append(float(target_time[0]))
            t_ends.append(float(target_time[-1]))
            print(f"[{index:03d}/{len(files):03d}] OK  {rel}  {case_type}  "
                  f"raw={time_raw.size} -> {args.points}  t={time_raw[0]:.4g}..{time_raw[-1]:.4g}")
            if warnings_all:
                report_lines.append(f"WARN {rel}: " + " | ".join(warnings_all))
        except Exception as exc:
            failed += 1
            msg = f"FAIL {rel}: {type(exc).__name__}: {exc}"
            report_lines.append(msg)
            print(f"[{index:03d}/{len(files):03d}] {msg}", file=sys.stderr)

    if not load_blocks:
        report_path = output_dir / "export_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print("没有成功导出任何工况。请查看 h5_structure.txt 和 export_report.txt。", file=sys.stderr)
        return 3

    loads = np.stack(load_blocks, axis=0).astype(np.float32)
    npz_path = output_dir / "compact_all_loads.npz"
    np.savez_compressed(
        npz_path,
        case_id=np.asarray(case_ids, dtype=str),
        case_type=np.asarray(case_types, dtype=str),
        params_json=np.asarray(params_json, dtype=str),
        t_start=np.asarray(t_starts, dtype=np.float64),
        t_end=np.asarray(t_ends, dtype=np.float64),
        channels=np.asarray(CHANNELS, dtype=str),
        loads=loads,
    )

    manifest_path = output_dir / "h5_manifest.csv"
    fieldnames = list(records[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    report_path = output_dir / "export_report.txt"
    summary = [
        f"input_dir={input_dir}",
        f"found_files={len(files)}",
        f"exported_cases={len(load_blocks)}",
        f"failed_files={failed}",
        f"points_per_case={args.points}",
        f"loads_shape={loads.shape}",
        f"npz={npz_path}",
        f"manifest={manifest_path}",
        "",
        *report_lines,
    ]
    report_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n导出完成：")
    print(f"  {npz_path}  ({npz_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"  {manifest_path}")
    print(f"  {report_path}")
    if failed:
        print(f"注意：有 {failed} 个文件失败。先把 h5_structure.txt 和 export_report.txt 发来定位。")
        return 0 if args.skip_failed else 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
