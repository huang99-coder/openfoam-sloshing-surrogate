#!/usr/bin/env python3
"""
check_mode_ceiling.py — 逐模态 Fourier 时间表示的拟合上限。

对比时间表示上限与回归精度，区分级数截断和系数回归的影响。

用法：
  python check_mode_ceiling.py data/shared_overtaking_rich.h5 --n-modes 6
"""
import argparse
import numpy as np
import h5py

from train_podnn import fourier_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--n-modes', type=int, default=6)
    args = ap.parse_args()

    with h5py.File(args.data, 'r') as f:
        y = f['y'][()]
        bd = f['case_boundaries'][()]
    Tm = int(min(e - s for s, e in bd))
    yc = np.array([y[s:s + Tm, :args.n_modes] for s, e in bd])
    n, T, K = yc.shape
    t = np.linspace(0.0, 1.0, T)

    print(f"\n{args.data}  ({n} 工况, {K} 模态, T={T})")
    header = '  nh |' + ''.join(f"   a_{k+1:<2} " for k in range(K)) + '|  池化'
    print(header)
    print('-' * len(header))
    for nh in (6, 8, 10, 12, 16):
        Phi = fourier_basis(t, nh)
        cells, ss_res_tot, ss_tot_tot = [], 0.0, 0.0
        for k in range(K):
            Y = yc[:, :, k].T                      # (T, n) 该模态所有工况
            coef, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
            rec = Phi @ coef
            ss_res = float(((Y - rec) ** 2).sum())
            ss_tot = float(((Y - Y.mean()) ** 2).sum())
            cells.append(1.0 - ss_res / ss_tot)
            ss_res_tot += ss_res
            ss_tot_tot += ss_tot
        pooled = 1.0 - ss_res_tot / ss_tot_tot
        print(f"  {nh:>2} |" + ''.join(f" {r:6.3f} " for r in cells)
              + f"| {pooled:6.4f}")

    print("\n说明：拟合上限低于 0.8 时，可比较增加 nh 后的时间表示精度；")
    print("      拟合上限较高而交叉验证 R² 较低时，可进一步检查输入特征与回归模型。")


if __name__ == '__main__':
    main()