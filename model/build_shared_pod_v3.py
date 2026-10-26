#!/usr/bin/env python3
"""
build_shared_pod_v3.py — 共享 POD 基、完整特征值谱与空间模态导出。

数值处理
--------
- 保存空间模态 U，用于由系数重构物理场。
- 保存完整特征值谱，以全谱总和计算累计能量及截断误差。
- 使用 float64 分块累加 Gram 矩阵，降低大规模内积的舍入误差。
- 可通过 --cell-volumes 设置体积加权内积。非均匀网格上的未加权
  欧氏内积对应点采样权重；体积加权模态满足 <phi_i, phi_j>_V = delta_ij。

用法
----
  python build_shared_pod_v3.py --input-dir all_h5_data --types slope --n-modes 20
  python build_shared_pod_v3.py --input-dir all_h5_data --types overtaking --n-modes 40
  python build_shared_pod_v3.py --input-dir all_h5_data --types slope --n-modes 20 --cell-volumes cellV.npy

输出字段
--------
  spatial_modes      (M, r) float32   物理坐标下的共享空间模态 U
  eigenvalues_full   (N,)   float64   完整特征值谱
  cell_volumes_used  标量 bool        体积加权标记

重构：alpha(:, t) ≈ mean_field + spatial_modes @ y[t, :]
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
from build_shared_pod import get_field_info, load_field_matrix, explore


# ──────────────────────────────────────────────────────────
# float64 分块 Gram: C = S^T S, 避免 float32 精度损失
# ──────────────────────────────────────────────────────────
def gram_f64(S, chunk=20000):
    """S: (M, N) float32 (已去均值/已加权)。返回 (N, N) float64。"""
    M, N = S.shape
    C = np.zeros((N, N), dtype=np.float64)
    for a in range(0, M, chunk):
        B = S[a:a + chunk, :].astype(np.float64)
        C += B.T @ B
        del B
    return C


# ──────────────────────────────────────────────────────────
def build_one_type(ct, files, n_modes, stride, out_dir, cellV=None, chunk=20000):
    print(f"\n{'='*70}\n[{ct}]  {len(files)} 个工况\n{'='*70}")

    # --- pass 1: 元信息 ---
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
        print("  ⚠ 有效工况 < 3, 跳过")
        return

    # --- 体积权重对齐 ---
    w = None
    if cellV is not None:
        V = np.asarray(cellV, dtype=np.float64).ravel()
        if stride > 1:
            V = V[::stride]
        if V.shape[0] != M_ref:
            raise ValueError(f"单元体积长度 {V.shape[0]} 与网格点数 {M_ref} 不符（注意 stride）")
        w = np.sqrt(V).astype(np.float32)          # 内积权重的平方根
        print(f"  ✓ 体积加权已启用  V: min={V.min():.3e} max={V.max():.3e} "
              f"max/min={V.max()/V.min():.1f}")

    gb = M_ref * total * 4 / 1e9
    print(f"  预分配 S = ({M_ref:,}, {total})  ≈ {gb:.1f} GB  (取场={src_ref}, stride={stride})")
    S = np.empty((M_ref, total), dtype=np.float32)

    # --- pass 2: 填场 ---
    bounds, info_out, cur = [], [], 0
    for fp, params, T in specs:
        with h5py.File(fp, 'r') as f:
            arr = load_field_matrix(f, stride=stride)
        S[:, cur:cur + T] = arr
        bounds.append((cur, cur + T))
        info_out.append((os.path.basename(fp), params))
        cur += T
        del arr
        print(f"  + {os.path.basename(fp):50s} [{cur}/{total}]")

    # --- 去均值（均值在【未加权】的物理场上算） ---
    mean_field = S.mean(axis=1, keepdims=True)
    S -= mean_field

    # --- 加权（原地，省内存） ---
    if w is not None:
        S *= w[:, None]

    # --- 快照法 POD ---
    print("  快照法 POD (float64 分块 Gram) ...")
    C = gram_f64(S, chunk=chunk)
    eigval, eigvec = np.linalg.eigh(C)
    del C
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0, None)
    eigvec = eigvec[:, order]

    r = int(min(n_modes, total, np.sum(eigval > 1e-10 * max(eigval[0], 1e-30))))
    energy = float(eigval[:r].sum() / eigval.sum())
    print(f"  共享模态数 r={r}   前 r 阶占【全谱】={energy:.4f}")

    sigma = np.sqrt(eigval[:r])
    A = sigma[:, None] * eigvec[:, :r].T                       # (r, N) 时间系数

    # --- P0 核心: 空间模态 U = S_w V Sigma^-1, 再除回 sqrt(V) 得物理模态 ---
    print("  计算空间模态 U ...")
    U = (S @ eigvec[:, :r].astype(np.float32)) / sigma.astype(np.float32)[None, :]
    if w is not None:
        U /= w[:, None]                                        # 物理坐标下的模态
    del S

    # --- 自检: 用 6 阶重构训练场的相对误差(仅截断) ---
    for rr in (4, 6, 8, 12, 20, r):
        if rr <= r:
            cum = eigval[:rr].sum() / eigval.sum()
            print(f"    前{rr:>3}阶: 累计能量={cum:.4f}   场相对L2误差={np.sqrt(max(1-cum,0)):.4f}")

    # --- 组装 ---
    full_feat = FEATURE_NAMES + ['time']
    all_X, all_y, new_b, cur = [], [], [], 0
    for (fname, params), (s, e) in zip(info_out, bounds):
        y_c = A[:, s:e].T
        T = e - s
        t_norm = np.linspace(0, 1, T).reshape(-1, 1)
        row = [params[k] for k in FEATURE_NAMES]
        X_c = np.column_stack([np.tile(row, (T, 1)), t_norm])
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
                         compression='lzf')
        f.create_dataset('singular_values', data=sigma)
        # ★ P0 新增
        f.create_dataset('spatial_modes', data=U.astype(np.float32), compression='lzf')
        f.create_dataset('eigenvalues_full', data=eigval)
        meta = f.create_group('metadata')
        meta.attrs['n_cases'] = len(all_X)
        meta.attrs['n_samples'] = X_m.shape[0]
        meta.attrs['n_features'] = X_m.shape[1]
        meta.attrs['n_modes'] = r
        meta.attrs['case_type'] = f'shared_{ct}'
        meta.attrs['energy_captured'] = energy
        meta.attrs['field_source'] = src_ref
        meta.attrs['volume_weighted'] = bool(w is not None)
        meta.attrs['stride'] = int(stride)
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

    print(f"  ✅ 保存: {out_path}  (工况 {len(all_X)}, 模态 {r}, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB)")
    if src_ref == 'recon':
        print("  ⚠ 该类取场方式=recon: 场是各算例自身 POD 重构出来的。"
              "场误差的参考对象为该低秩重构场，不是全阶 CFD 场。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='all_h5_data')
    ap.add_argument('--output-dir', default='data')
    ap.add_argument('--n-modes', type=int, default=20)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--explore', action='store_true')
    ap.add_argument('--types', nargs='+', default=None)
    ap.add_argument('--cell-volumes', default=None,
                    help='单元体积 .npy（长度=未 stride 的网格点数），启用体积加权内积')
    ap.add_argument('--gram-chunk', type=int, default=20000,
                    help='float64 Gram 的行分块大小；内存紧就调小')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '**/*.h5'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
    if not files:
        print(f"❌ {args.input_dir} 下没有 H5"); return
    print(f"找到 {len(files)} 个 H5 文件")

    if args.explore:
        explore(files); return

    cellV = np.load(args.cell_volumes) if args.cell_volumes else None

    by_type = {}
    for fp in files:
        by_type.setdefault(detect_case_type(fp), []).append(fp)
    print("工况分布:", {k: len(v) for k, v in sorted(by_type.items())})
    for ct, fl in sorted(by_type.items()):
        if args.types and ct not in args.types:
            print(f"[{ct}] 不在 --types 过滤列表中, 跳过")
            continue
        build_one_type(ct, fl, args.n_modes, args.stride, args.output_dir,
                       cellV=cellV, chunk=args.gram_chunk)


if __name__ == '__main__':
    main()
