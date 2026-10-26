#!/usr/bin/env python3
"""
build_shared_pod.py  —  按工况类型建立【共享 POD 基】并投影 (v2)

v2 更新
------
  - braking / turning 没存原始快照场(snapshots/.../data 只有 times),
    自动改用 场 = mean_field + modes @ time_coefficients 重构后再做共享 POD。
  - 预分配 S 矩阵 + 逐块填入, 避免 np.hstack 双份内存峰值。
  - 打印每类的能量占比, 便于判断是否需要增大 --n-modes。

用法
----
  python build_shared_pod.py --input-dir all_h5_data --explore
  python build_shared_pod.py --input-dir all_h5_data --n-modes 20
  python build_shared_pod.py --input-dir all_h5_data --n-modes 20 --stride 4   # 省内存
"""

import os
import glob
import json
import argparse
import numpy as np
import h5py

from merge_pod_h5 import (
    detect_case_type, parse_filename, FEATURE_NAMES, CASE_TYPE_MAP,
)


# ──────────────────────────────────────────────────────────
# 判定单文件的取场方式: 'data'(原始快照) 或 'recon'(从自身POD重构)
# ──────────────────────────────────────────────────────────
def get_field_info(f):
    """返回 dict: {source, M, n_ts}"""
    # 1) 优先原始快照
    if 'snapshots/alpha_water/data' in f:
        shp = f['snapshots/alpha_water/data'].shape   # (M, n_ts) 或 (n_ts, M)
        times = f['snapshots/alpha_water/times']
        n_ts = int(times.shape[0])
        if shp[0] == n_ts:          # (n_ts, M)
            M, tax = shp[1], 0
        else:                       # (M, n_ts)
            M, tax = shp[0], 1
        return {'source': 'data', 'M': int(M), 'n_ts': n_ts, 'time_axis': tax}

    # 2) 回退: 用自身 POD 重构
    if 'pod/alpha_water/modes' in f and 'pod/alpha_water/time_coefficients' in f:
        M = int(f['pod/alpha_water/modes'].shape[0])
        n_ts = int(f['pod/alpha_water/time_coefficients'].shape[1])
        return {'source': 'recon', 'M': M, 'n_ts': n_ts, 'time_axis': None}

    raise KeyError("既无 snapshots/.../data 也无 pod/.../modes, 无法取场")


def load_field_matrix(f, stride=1):
    """读取/重构单文件场, 统一成 (M, n_ts) float32; stride 抽样网格点。"""
    info = get_field_info(f)
    if info['source'] == 'data':
        arr = f['snapshots/alpha_water/data'][()].astype(np.float32)
        arr = np.moveaxis(arr, info['time_axis'], -1)        # (..., n_ts)
        arr = arr.reshape(-1, info['n_ts'])                  # (M, n_ts)
    else:
        modes = f['pod/alpha_water/modes'][()].astype(np.float32)        # (M, r)
        coeff = f['pod/alpha_water/time_coefficients'][()].astype(np.float32)  # (r, n_ts)
        mean = f['pod/alpha_water/mean_field'][()].astype(np.float32)    # (M,)
        if stride > 1:
            modes = modes[::stride]
            mean = mean[::stride]
        arr = modes @ coeff + mean[:, None]                  # (M, n_ts)
        return arr
    if stride > 1:
        arr = arr[::stride, :]
    return arr


# ──────────────────────────────────────────────────────────
# 探明模式
# ──────────────────────────────────────────────────────────
def explore(files):
    print(f"\n{'='*70}\n探明 H5 结构(取前 2 个文件)\n{'='*70}")
    for fp in files[:2]:
        print(f"\n■ {os.path.basename(fp)}   [{detect_case_type(fp)}]")
        with h5py.File(fp, 'r') as f:
            try:
                info = get_field_info(f)
                M, n_ts = info['M'], info['n_ts']
                tag = '原始快照' if info['source'] == 'data' else 'POD重构(无原始快照)'
                print(f"  取场方式: {info['source']} ({tag})  M≈{M:,}  n_ts={n_ts}")
            except Exception as e:
                print(f"  ⚠ 无法取场: {e}")

    print(f"\n{'='*70}\n按类型统计 + 内存估算\n{'='*70}")
    by_type = {}
    for fp in files:
        by_type.setdefault(detect_case_type(fp), []).append(fp)
    for ct, fl in sorted(by_type.items()):
        try:
            with h5py.File(fl[0], 'r') as f:
                info = get_field_info(f)
            M, n_ts = info['M'], info['n_ts']
            N = len(fl) * n_ts
            gb = M * N * 4 / 1e9
            print(f"  {ct:11s}: {len(fl):2d} 组 × {n_ts} = {N} 快照, M≈{M:,}, "
                  f"取场={info['source']:5s} → S 约 {gb:.1f} GB")
        except Exception as e:
            print(f"  {ct:11s}: {len(fl):2d} 组 (估算失败: {e})")
    print(f"\n  内存不够就加 --stride N (每 N 个网格点取 1 个), GB 数随之 /N。")


# ──────────────────────────────────────────────────────────
# 单类型: 拼场 → 共享 POD(快照法) → 投影
# ──────────────────────────────────────────────────────────
def build_one_type(ct, files, n_modes, stride, out_dir):
    print(f"\n{'='*70}\n[{ct}]  {len(files)} 个工况\n{'='*70}")

    # --- pass 1: 读元信息, 确定 M / 各 case 时间步 / 总列数 ---
    specs, total, M_ref, src_ref = [], 0, None, None
    for fp in files:
        with h5py.File(fp, 'r') as f:
            info = get_field_info(f)
        Mc = (info['M'] + stride - 1) // stride if stride > 1 else info['M']
        if M_ref is None:
            M_ref, src_ref = Mc, info['source']
        elif Mc != M_ref:
            print(f"  ⚠ {os.path.basename(fp)}: 网格点数 {Mc}≠{M_ref}, 跳过")
            continue
        specs.append((fp, parse_filename(fp, ct), info['n_ts']))
        total += info['n_ts']

    if len(specs) < 3:
        print(f"  ⚠ 有效工况 < 3, 跳过"); return

    gb = M_ref * total * 4 / 1e9
    print(f"  预分配 S = ({M_ref:,}, {total})  ≈ {gb:.1f} GB  (取场={src_ref}, stride={stride})")
    S = np.empty((M_ref, total), dtype=np.float32)

    # --- pass 2: 填入场 ---
    bounds, info_out, cur = [], [], 0
    for fp, params, T in specs:
        with h5py.File(fp, 'r') as f:
            arr = load_field_matrix(f, stride=stride)        # (M, T)
        S[:, cur:cur + T] = arr
        bounds.append((cur, cur + T))
        info_out.append((os.path.basename(fp), params))
        cur += T
        del arr
        print(f"  + {os.path.basename(fp):50s} [{cur}/{total}]")

    # --- 去均值场 + 快照法 POD ---
    print(f"  快照法 POD ...")
    mean_field = S.mean(axis=1, keepdims=True)
    S -= mean_field
    C = S.T @ S                                              # (N, N)
    eigval, eigvec = np.linalg.eigh(C)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.clip(eigval[order], 0, None), eigvec[:, order]

    r = int(min(n_modes, total, np.sum(eigval > 1e-10)))
    energy = float(eigval[:r].sum() / eigval.sum())
    print(f"  共享模态数 r={r}   能量占比={energy:.4f}")
    if energy < 0.9:
        print(f"  ⚠ 能量占比偏低, 建议 --n-modes 调到 30~40")

    sigma = np.sqrt(eigval[:r])
    A = sigma[:, None] * eigvec[:, :r].T                      # (r, N) 共享基下的时间系数

    # --- 组装成 train_podnn.py 的格式 ---
    full_feat = FEATURE_NAMES + ['time']
    all_X, all_y, new_b, cur = [], [], [], 0
    for (fname, params), (s, e) in zip(info_out, bounds):
        y_c = A[:, s:e].T                                     # (T, r)
        T = e - s
        t_norm = np.linspace(0, 1, T).reshape(-1, 1)
        row = [params[k] for k in FEATURE_NAMES]
        X_c = np.column_stack([np.tile(row, (T, 1)), t_norm]) # (T, 13)
        all_X.append(X_c); all_y.append(y_c)
        new_b.append((cur, cur + T)); cur += T

    X_m, y_m = np.vstack(all_X), np.vstack(all_y)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'shared_{ct}.h5')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('X', data=X_m, compression='gzip')
        f.create_dataset('y', data=y_m, compression='gzip')
        f.create_dataset('case_boundaries', data=np.array(new_b))
        f.create_dataset('mean_field', data=mean_field.squeeze().astype(np.float32),
                         compression='gzip')
        f.create_dataset('singular_values', data=sigma)
        meta = f.create_group('metadata')
        meta.attrs['n_cases'] = len(all_X)
        meta.attrs['n_samples'] = X_m.shape[0]
        meta.attrs['n_features'] = X_m.shape[1]
        meta.attrs['n_modes'] = r
        meta.attrs['case_type'] = f'shared_{ct}'
        meta.attrs['energy_captured'] = energy
        meta.attrs['field_source'] = src_ref
        meta.attrs['feature_names'] = json.dumps(full_feat)
        gi = f.create_group('case_info')
        for i, (fname, params) in enumerate(info_out):
            cg = gi.create_group(f'case_{i:03d}')
            cg.attrs['file'] = fname
            for k, v in params.items():
                try:
                    cg.attrs[k] = float(v)
                except (TypeError, ValueError):
                    pass

    del S
    print(f"  ✅ 保存: {out_path}  (工况 {len(all_X)}, 模态 {r}, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='all_h5_data')
    ap.add_argument('--output-dir', default='data')
    ap.add_argument('--n-modes', type=int, default=20)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--explore', action='store_true')
    ap.add_argument('--types', nargs='+', default=None,
                    help='只处理这些工况类型, 例如 --types overtaking')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '**/*.h5'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
    if not files:
        print(f"❌ {args.input_dir} 下没有 H5"); return
    print(f"找到 {len(files)} 个 H5 文件")

    if args.explore:
        explore(files); return

    by_type = {}
    for fp in files:
        by_type.setdefault(detect_case_type(fp), []).append(fp)
    print("工况分布:", {k: len(v) for k, v in sorted(by_type.items())})
    for ct, fl in sorted(by_type.items()):
        if args.types and ct not in args.types:
            print(f"[{ct}] 不在 --types 过滤列表中, 跳过")
            continue
        build_one_type(ct, fl, args.n_modes, args.stride, args.output_dir)
    print(f"\n{'='*70}\n完成。data/ 下 4 个 shared_*.h5, 接着分别训练。\n{'='*70}")


if __name__ == '__main__':
    main()