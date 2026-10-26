#!/usr/bin/env python3
"""
将逐工况 OpenFOAM/POD H5 压缩为轻量载荷数据集。

默认仅导出论文表 9 所需主方向力：
- braking / slope: Fz
- turning / overtaking: Fx

默认把每条时程重采样为 1000 个归一化时间点，并保留原始峰值。
通常 224 个算例导出后仅约 1–3 MB。

也可用 --mode all 导出 Fx/Fy/Fz/Mx/My/Mz 六通道，通常仍小于 10 MB。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


FORCE_NAMES = ("Fx", "Fy", "Fz")
MOMENT_NAMES = ("Mx", "My", "Mz")
ALL_NAMES = FORCE_NAMES + MOMENT_NAMES


def infer_case_type(filename: str) -> str:
    name = filename.lower()
    if "brake" in name or "braking" in name:
        return "braking"
    if "turn" in name:
        return "turning"
    if "overtak" in name or "_ot_" in name or name.startswith("ot_") or "lane" in name:
        return "overtaking"
    if "slope" in name or "uphill" in name or "downhill" in name or "pod_data_3d_up_" in name or "pod_data_3d_down_" in name:
        return "slope"
    return "unknown"


def primary_channel(case_type: str) -> str:
    mapping = {
        "braking": "Fz",
        "slope": "Fz",
        "turning": "Fx",
        "overtaking": "Fx",
    }
    if case_type not in mapping:
        raise ValueError(f"无法根据工况类型 {case_type!r} 确定主载荷方向")
    return mapping[case_type]


def iter_datasets(group: h5py.Group) -> Iterable[tuple[str, h5py.Dataset]]:
    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            items.append((name, obj))

    items: list[tuple[str, h5py.Dataset]] = []
    group.visititems(visitor)
    return items


def find_dataset(root: h5py.File, names: Iterable[str]) -> np.ndarray | None:
    wanted = {name.lower() for name in names}

    # 优先在 monitoring 分组内找，避免误取其他同名数据
    groups: list[h5py.Group | h5py.File] = []
    if "monitoring" in root and isinstance(root["monitoring"], h5py.Group):
        groups.append(root["monitoring"])
    groups.append(root)

    for group in groups:
        for path, dataset in iter_datasets(group):
            basename = path.rsplit("/", 1)[-1].lower()
            if basename in wanted:
                return np.asarray(dataset[()])
    return None


def extract_force_component(root: h5py.File, component: str) -> np.ndarray | None:
    direct = find_dataset(root, [component])
    if direct is not None:
        return np.asarray(direct).squeeze()

    forces = find_dataset(root, ["forces", "force"])
    if forces is None:
        return None

    arr = np.asarray(forces)
    if arr.ndim != 2:
        return None

    idx = FORCE_NAMES.index(component)
    if arr.shape[1] >= 3:
        return arr[:, idx]
    if arr.shape[0] >= 3:
        return arr[idx, :]
    return None


def extract_moment_component(root: h5py.File, component: str) -> np.ndarray | None:
    direct = find_dataset(root, [component])
    if direct is not None:
        return np.asarray(direct).squeeze()

    moments = find_dataset(root, ["moments", "moment"])
    if moments is None:
        return None

    arr = np.asarray(moments)
    if arr.ndim != 2:
        return None

    idx = MOMENT_NAMES.index(component)
    if arr.shape[1] >= 3:
        return arr[:, idx]
    if arr.shape[0] >= 3:
        return arr[idx, :]
    return None


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return json_safe(value.item())
        if value.size <= 50:
            return [json_safe(v) for v in value.tolist()]
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def collect_parameters(root: h5py.File) -> dict[str, Any]:
    result: dict[str, Any] = {}

    # 文件根属性
    for key, value in root.attrs.items():
        result[f"root_attr/{key}"] = json_safe(value)

    if "parameters" not in root or not isinstance(root["parameters"], h5py.Group):
        return result

    params = root["parameters"]
    for key, value in params.attrs.items():
        result[f"attr/{key}"] = json_safe(value)

    for path, dataset in iter_datasets(params):
        try:
            result[path] = json_safe(np.asarray(dataset[()]))
        except Exception as exc:
            result[path] = f"<read error: {exc}>"

    return result


def clean_time_signal(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    y = np.asarray(signal, dtype=np.float64).reshape(-1)

    n = min(t.size, y.size)
    t, y = t[:n], y[:n]

    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if t.size < 2:
        raise ValueError("有效时间点不足 2 个")

    order = np.argsort(t)
    t, y = t[order], y[order]

    unique_t, unique_idx = np.unique(t, return_index=True)
    y = y[unique_idx]
    t = unique_t

    if t.size < 2 or t[-1] <= t[0]:
        raise ValueError("时间轴无有效跨度")

    return t, y


def resample(time: np.ndarray, signal: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    t, y = clean_time_signal(time, signal)
    target_t = t[0] + tau * (t[-1] - t[0])
    y_new = np.interp(target_t, t, y)

    peak_idx = int(np.argmax(np.abs(y)))
    raw_peak_signed = float(y[peak_idx])
    raw_peak_abs = float(abs(y[peak_idx]))

    return y_new.astype(np.float32), float(t[0]), float(t[-1]), raw_peak_signed, raw_peak_abs


def get_time_axis(root: h5py.File, kind: str) -> np.ndarray | None:
    if kind == "force":
        return find_dataset(root, ["force_time", "forces_time", "time_force"])
    return find_dataset(root, ["moment_time", "moments_time", "time_moment"])


def export_dataset(
    input_dir: Path,
    output_path: Path,
    n_points: int,
    mode: str,
) -> None:
    files = sorted(input_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"{input_dir} 中没有找到 .h5 文件")

    tau = np.linspace(0.0, 1.0, n_points, dtype=np.float32)

    case_ids: list[str] = []
    case_types: list[str] = []
    params_json: list[str] = []
    t_start_list: list[float] = []
    t_end_list: list[float] = []
    errors: list[str] = []

    if mode == "primary":
        channel_list: list[str] = []
        signals: list[np.ndarray] = []
        raw_peak_signed_list: list[float] = []
        raw_peak_abs_list: list[float] = []
    else:
        all_signals: list[np.ndarray] = []
        all_peak_signed: list[np.ndarray] = []
        all_peak_abs: list[np.ndarray] = []

    for path in files:
        case_type = infer_case_type(path.name)
        if case_type == "unknown":
            errors.append(f"{path.name}: 无法识别工况类型，已跳过")
            continue

        try:
            with h5py.File(path, "r") as root:
                force_time = get_time_axis(root, "force")
                moment_time = get_time_axis(root, "moment")

                # 某些文件只保存一个通用 time
                if force_time is None:
                    force_time = find_dataset(root, ["time"])
                if moment_time is None:
                    moment_time = force_time

                if force_time is None:
                    raise KeyError("未找到 force_time 或通用 time")

                if mode == "primary":
                    channel = primary_channel(case_type)
                    signal = extract_force_component(root, channel)
                    if signal is None:
                        raise KeyError(f"未找到主载荷 {channel}")

                    y_new, t0, t1, peak_signed, peak_abs = resample(
                        force_time, signal, tau
                    )

                    channel_list.append(channel)
                    signals.append(y_new)
                    raw_peak_signed_list.append(peak_signed)
                    raw_peak_abs_list.append(peak_abs)
                    t_start_list.append(t0)
                    t_end_list.append(t1)

                else:
                    matrix = np.full((n_points, 6), np.nan, dtype=np.float32)
                    peak_signed = np.full(6, np.nan, dtype=np.float32)
                    peak_abs = np.full(6, np.nan, dtype=np.float32)
                    t0_candidates: list[float] = []
                    t1_candidates: list[float] = []

                    for j, name in enumerate(FORCE_NAMES):
                        signal = extract_force_component(root, name)
                        if signal is None:
                            continue
                        y_new, t0, t1, p_signed, p_abs = resample(
                            force_time, signal, tau
                        )
                        matrix[:, j] = y_new
                        peak_signed[j] = p_signed
                        peak_abs[j] = p_abs
                        t0_candidates.append(t0)
                        t1_candidates.append(t1)

                    if moment_time is not None:
                        for j, name in enumerate(MOMENT_NAMES, start=3):
                            signal = extract_moment_component(root, name)
                            if signal is None:
                                continue
                            y_new, t0, t1, p_signed, p_abs = resample(
                                moment_time, signal, tau
                            )
                            matrix[:, j] = y_new
                            peak_signed[j] = p_signed
                            peak_abs[j] = p_abs
                            t0_candidates.append(t0)
                            t1_candidates.append(t1)

                    if np.all(np.isnan(matrix)):
                        raise KeyError("六个力/力矩通道均未找到")

                    all_signals.append(matrix)
                    all_peak_signed.append(peak_signed)
                    all_peak_abs.append(peak_abs)
                    t_start_list.append(min(t0_candidates))
                    t_end_list.append(max(t1_candidates))

                case_ids.append(path.stem)
                case_types.append(case_type)
                params_json.append(
                    json.dumps(
                        collect_parameters(root),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )

        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not case_ids:
        raise RuntimeError("没有成功导出任何算例")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    common = {
        "tau": tau,
        "case_id": np.asarray(case_ids, dtype=str),
        "case_type": np.asarray(case_types, dtype=str),
        "params_json": np.asarray(params_json, dtype=str),
        "t_start": np.asarray(t_start_list, dtype=np.float32),
        "t_end": np.asarray(t_end_list, dtype=np.float32),
        "n_points": np.asarray(n_points, dtype=np.int32),
    }

    if mode == "primary":
        np.savez_compressed(
            output_path,
            **common,
            channel=np.asarray(channel_list, dtype=str),
            signal=np.stack(signals).astype(np.float32),
            raw_peak_signed=np.asarray(raw_peak_signed_list, dtype=np.float32),
            raw_peak_abs=np.asarray(raw_peak_abs_list, dtype=np.float32),
            unit=np.asarray("N"),
        )
    else:
        np.savez_compressed(
            output_path,
            **common,
            channels=np.asarray(ALL_NAMES, dtype=str),
            loads=np.stack(all_signals).astype(np.float32),
            raw_peak_signed=np.stack(all_peak_signed).astype(np.float32),
            raw_peak_abs=np.stack(all_peak_abs).astype(np.float32),
            units=np.asarray(["N", "N", "N", "N*m", "N*m", "N*m"], dtype=str),
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"成功导出 {len(case_ids)} 个算例")
    print(f"输出文件：{output_path}")
    print(f"文件大小：{size_mb:.2f} MB")
    print(f"模式：{mode}，每条时程 {n_points} 点")

    if errors:
        error_path = output_path.with_suffix(".errors.txt")
        error_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"有 {len(errors)} 个文件未导出，详情：{error_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", default=Path("compact_primary_loads.npz"), type=Path)
    parser.add_argument("--n-points", default=1000, type=int)
    parser.add_argument("--mode", choices=("primary", "all"), default="primary")
    args = parser.parse_args()

    if args.n_points < 100:
        raise ValueError("--n-points 不建议低于 100")

    export_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
        n_points=args.n_points,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()