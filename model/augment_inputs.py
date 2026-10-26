#!/usr/bin/env python3
"""
augment_inputs.py  —  给已建好的 shared_*.h5 换上【真实物理输入】

为什么
------
现在的 shared_*.h5 里的输入 X 是从【文件名正则】解析出来的 12 维 FEATURE_NAMES,
而每个原始 H5 的 parameters/ + features/inputs 里存着 CFD 真实记录的物理量
(peak_gx, total_time, t_change, is_double_overtake, radius, speed_kmh ...)。
inspect_params.py 已证明: overtaking/turning 有真实驱动量没被喂进模型,
其中 overtaking 连 a_1 都废了(0.36)——因为它拿到的 lateral_acceleration 是
按 lane_time 套公式现编的代理值, 真正的 peak_gx / 单双超车标志根本没进去。

本脚本【不重算 POD】(那要 6~10GB 内存)。共享基与时间系数 y 完全不变,
只把输入 X 换成真实物理量, 另存 shared_<type>_rich.h5。

依赖: 原始 all_h5_data 还在 (要回去读 parameters/)。

用法
----
  python augment_inputs.py --shared-dir data --input-dir all_h5_data
  python augment_inputs.py --shared-dir data --input-dir all_h5_data --types turning overtaking

产出: data/shared_turning_rich.h5  data/shared_overtaking_rich.h5  ...
然后用打了补丁的 kfold_eval.py 跑 (它会自动识别 rich 文件的 input_cols)。
"""

import os
import glob
import json
import argparse
import numpy as np
import h5py

from merge_pod_h5 import detect_case_type


# ──────────────────────────────────────────────────────────
# 每类要注入的【真实物理输入列】
#   - 已去掉彼此的确定性函数 (sin/cos/angle_rad 之于 angle; fill_y_level 之于 fill;
#     n_phases 之于 is_double_overtake)
#   - braking / slope 本就把全部独立自由度用上了, 这里列出只为统一/做对照,
#     跑出来应与原版基本一致。
# ──────────────────────────────────────────────────────────
RICH_COLS = {
    'turning':    ['fill_level', 'lateral_acceleration', 'radius', 'speed_kmh'],
    'overtaking': ['fill_level', 'peak_gx', 'total_time', 't_change',
                   'is_double_overtake'],
    'braking':    ['fill_level', 'deceleration', 'density', 'viscosity'],
    'slope':      ['fill_level', 'angle', 'direction'],
}


# ──────────────────────────────────────────────────────────
def read_scalars(fp):
    """读单个原始 H5 的 parameters/ + features/inputs 下所有标量, 键用叶名。"""
    out = {}
    with h5py.File(fp, 'r') as f:
        for group in ('parameters', 'features/inputs'):
            if group not in f:
                continue
            grp = f[group]

            def _v(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.shape == ():
                    try:
                        out[name.split('/')[-1]] = float(obj[()])
                    except (TypeError, ValueError):
                        pass
            grp.visititems(_v)
            # 有些量可能存成 attrs
            for k, v in grp.attrs.items():
                try:
                    out[k] = float(np.asarray(v).reshape(-1)[0])
                except (TypeError, ValueError, IndexError):
                    pass
    return out


def build_file_index(input_dir):
    """basename(小写) -> 完整路径"""
    files = sorted(glob.glob(os.path.join(input_dir, '**/*.h5'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, '*.h5')))
    idx = {}
    for fp in files:
        idx[os.path.basename(fp).lower()] = fp
    return idx


# ──────────────────────────────────────────────────────────
def augment_one(ct, shared_path, file_index, out_dir):
    cols = RICH_COLS.get(ct)
    if cols is None:
        print(f"  [{ct}] 无 RICH_COLS 定义, 跳过"); return

    with h5py.File(shared_path, 'r') as f:
        y_all = f['y'][()]
        bd = f['case_boundaries'][()]
        meta = dict(f['metadata'].attrs)
        n_modes = int(meta['n_modes'])
        # 取每个 case 的原始文件名
        gi = f['case_info']
        case_files = []
        for i in range(len(bd)):
            key = f'case_{i:03d}'
            fname = gi[key].attrs['file'] if key in gi else None
            case_files.append(fname)
        # 透传字段
        mean_field = f['mean_field'][()] if 'mean_field' in f else None
        singular_values = f['singular_values'][()] if 'singular_values' in f else None

    # --- 逐 case 读真实参数 ---
    rows, missing = [], []
    for fname in case_files:
        fp = file_index.get(str(fname).lower()) if fname else None
        if fp is None:
            missing.append(fname); rows.append({}); continue
        rows.append(read_scalars(fp))

    if missing:
        print(f"  ⚠ [{ct}] {len(missing)} 个 case 在 {os.path.basename(os.path.dirname(shared_path))} "
              f"找不到原始文件(示例 {missing[:2]}), 这些行将填 NaN→0")

    # --- 哪些列真的能取到且在变 ---
    avail = []
    for c in cols:
        vals = np.array([r.get(c, np.nan) for r in rows], float)
        n_ok = np.sum(~np.isnan(vals))
        nu = len(np.unique(np.round(vals[~np.isnan(vals)], 8))) if n_ok else 0
        if n_ok == 0:
            print(f"      - {c:22s} 全缺失, 丢弃")
        elif nu <= 1:
            print(f"      - {c:22s} 常量(n_unique={nu}), 丢弃")
        else:
            avail.append(c)
    if not avail:
        print(f"  ⚠ [{ct}] 没有可用的在变真实输入, 跳过"); return
    print(f"  [{ct}] 注入真实输入列({len(avail)}): {avail}")

    # --- 组装新的 X (与原 y/boundaries 完全对齐) ---
    feat_names = avail + ['time']
    all_X = []
    for (s, e), r in zip(bd, rows):
        T = int(e - s)
        vals = [r.get(c, np.nan) for c in avail]
        vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
        t_norm = np.linspace(0, 1, T).reshape(-1, 1)
        X_c = np.column_stack([np.tile(vals, (T, 1)), t_norm])
        all_X.append(X_c)
    X_m = np.vstack(all_X)
    assert X_m.shape[0] == y_all.shape[0], "X/y 行数不一致, 对齐出错"

    # --- 写 rich 文件 ---
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'shared_{ct}_rich.h5')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('X', data=X_m, compression='gzip')
        f.create_dataset('y', data=y_all, compression='gzip')
        f.create_dataset('case_boundaries', data=bd)
        if mean_field is not None:
            f.create_dataset('mean_field', data=mean_field, compression='gzip')
        if singular_values is not None:
            f.create_dataset('singular_values', data=singular_values)
        m = f.create_group('metadata')
        m.attrs['n_cases'] = len(bd)
        m.attrs['n_samples'] = X_m.shape[0]
        m.attrs['n_features'] = X_m.shape[1]
        m.attrs['n_modes'] = n_modes
        m.attrs['case_type'] = f'shared_{ct}_rich'
        m.attrs['feature_names'] = json.dumps(feat_names)
        m.attrs['input_cols'] = json.dumps(avail)      # ← kfold_eval 补丁靠这个识别
        if 'energy_captured' in meta:
            m.attrs['energy_captured'] = meta['energy_captured']
        if 'field_source' in meta:
            m.attrs['field_source'] = meta['field_source']
    print(f"  ✅ {out_path}  (工况 {len(bd)}, 模态 {n_modes}, "
          f"输入 {len(avail)}, {os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shared-dir', default='data')
    ap.add_argument('--input-dir', default='all_h5_data')
    ap.add_argument('--types', nargs='+',
                    default=['turning', 'overtaking', 'braking', 'slope'])
    args = ap.parse_args()

    file_index = build_file_index(args.input_dir)
    print(f"原始文件索引: {len(file_index)} 个 H5")

    for ct in args.types:
        sp = os.path.join(args.shared_dir, f'shared_{ct}.h5')
        if not os.path.exists(sp):
            print(f"\n[{ct}] 找不到 {sp}, 跳过"); continue
        print(f"\n{'='*64}\n处理 {sp}\n{'='*64}")
        augment_one(ct, sp, file_index, args.shared_dir)

    print(f"\n完成。rich 文件在 {args.shared_dir}/ 下, 用补丁版 kfold_eval.py 跑。")


if __name__ == '__main__':
    main()