#!/usr/bin/env python3
"""
check_split_energy.py — 比较单次、双次超车分组建基后的累计能量。

使用快照法计算特征值谱，不修改 shared_*.h5。
--stride 控制空间采样步长，降低内存需求；全网格计算使用默认步长。

用法：
  python check_split_energy.py --input-dir all_h5_data --stride 4
"""
import os
import glob
import argparse
import numpy as np
import h5py

from merge_pod_h5 import detect_case_type
from build_shared_pod import get_field_info, load_field_matrix


def group_spectrum(files, stride, label):
    # pass 1: 网格一致性检查
    specs, M_ref, total = [], None, 0
    for fp in files:
        with h5py.File(fp, 'r') as f:
            info = get_field_info(f)
        Mc = (info['M'] + stride - 1) // stride if stride > 1 else info['M']
        if M_ref is None:
            M_ref = Mc
        elif Mc != M_ref:
            print(f"  跳过 {os.path.basename(fp)} (网格 {Mc:,}≠{M_ref:,})")
            continue
        specs.append((fp, info['n_ts']))
        total += info['n_ts']

    print(f"\n[{label}] {len(specs)} 工况, {total} 快照, "
          f"S ≈ {M_ref * total * 4 / 1e9:.1f} GB (stride={stride})")

    # pass 2: 填 S
    S = np.empty((M_ref, total), dtype=np.float32)
    cur = 0
    for fp, T in specs:
        with h5py.File(fp, 'r') as f:
            S[:, cur:cur + T] = load_field_matrix(f, stride=stride)
        cur += T

    # 快照法特征谱
    S -= S.mean(axis=1, keepdims=True)
    C = (S.T @ S).astype(np.float64)
    del S
    ev = np.linalg.eigvalsh(C)[::-1]
    del C
    ev = np.clip(ev, 0, None)
    e = ev / ev.sum()
    cum = np.cumsum(e)

    print(f"  前 8 模态各自占比: {np.round(e[:8], 4)}")
    for k in (4, 6, 8, 10, 20, 40):
        if k <= len(cum):
            print(f"  前{k:>2}模态累计: {cum[k - 1]:.4f}")
    return cum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='all_h5_data')
    ap.add_argument('--stride', type=int, default=4,
                    help='网格点抽样间隔, 只看能量占比时 4 足够 (默认 4)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '**/*.h5'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
    ot = [fp for fp in files if detect_case_type(fp) == 'overtaking']

    double = [fp for fp in ot if '_double_' in os.path.basename(fp).lower()]
    single = [fp for fp in ot if '_double_' not in os.path.basename(fp).lower()]
    print(f"超车共 {len(ot)} 个: 单超车 {len(single)}, 双超车 {len(double)}")

    cum_s = group_spectrum(single, args.stride, '单超车 single')
    cum_d = group_spectrum(double, args.stride, '双超车 double')

    print(f"\n{'=' * 56}")
    print(f"{'':12s} {'前4':>8} {'前6':>8} {'前8':>8} {'前10':>8}")
    print(f"{'单超车':12s} {cum_s[3]:>8.4f} {cum_s[5]:>8.4f} "
          f"{cum_s[7]:>8.4f} {cum_s[9]:>8.4f}")
    print(f"{'双超车':12s} {cum_d[3]:>8.4f} {cum_d[5]:>8.4f} "
          f"{cum_d[7]:>8.4f} {cum_d[9]:>8.4f}")
    print(f"{'混合(参考)':12s} {'0.6732':>8} {'0.7439':>8}   (来自 check_energy)")
    print(f"\n说明：分组后前6模态累计能量达到 0.85 以上，可表明能量集中度提高；")
    print(f"      若仍在 0.75~0.80 附近，分组对能量集中度的改善有限；")
    print(f"      可对比混合基与分组基在 n-modes-use=6~8 时的下游误差。")


if __name__ == '__main__':
    main()
