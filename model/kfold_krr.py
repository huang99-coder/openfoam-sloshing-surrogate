#!/usr/bin/env python3
"""
kfold_krr.py — 用核岭回归(KRR, RBF核)替换 MLP 的 K 折评估

为什么试它
----------
49 样本 × 5~6 维输入是核方法的典型主场; KRR 对小样本插值往往比小 MLP 稳。
折划分与 kfold_eval.py 完全一致 (同 seed 同 KFold), 可直接对比池化 R²。
超参数 (alpha, gamma) 通过每个训练折内部的 5 折 GridSearchCV 选择。

用法:
  python kfold_krr.py --data data/shared_overtaking_rich.h5 --n-modes-use 6 --n-harmonics 12 --kfold 5
"""
import argparse
import numpy as np
import h5py
import torch  # noqa: F401  (kfold_eval 导入链需要)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.kernel_ridge import KernelRidge

from kfold_eval import load_cases, drop_constant, r2
from train_podnn import fit_fourier_coeffs, reconstruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n-modes-use', type=int, default=6)
    ap.add_argument('--n-harmonics', type=int, default=12)
    ap.add_argument('--kfold', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    X, y, names, nmu = load_cases(args.data, args.n_modes_use)
    n_ts = y.shape[1]
    nh = args.n_harmonics
    X, names, _ = drop_constant(X, names)

    theta = np.array([fit_fourier_coeffs(seq, nh) for seq in y], dtype=float)
    y_rec = np.array([reconstruct(t, n_ts, nmu, nh) for t in theta])
    ceil = r2(y, y_rec)

    n = len(X)
    print(f"{'=' * 64}")
    print(f"K折CV (KernelRidge/RBF): {args.data}")
    print(f"  工况数={n}  有效输入={names}")
    print(f"  n_modes_use={nmu}  n_harmonics={nh}  kfold={args.kfold}")
    print(f"  Fourier天花板 R²={ceil:.4f}")
    print(f"{'=' * 64}")

    grid = {
        'alpha': np.logspace(-4, 1, 11),
        'gamma': np.logspace(-2.5, 1, 8),
    }

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    y_pred_all = np.zeros_like(y)
    fold_r2 = []

    for fi, (tr, te) in enumerate(kf.split(np.arange(n))):
        sx, st = StandardScaler(), StandardScaler()
        Xtr = sx.fit_transform(X[tr]); Xte = sx.transform(X[te])
        Ttr = st.fit_transform(theta[tr])

        gs = GridSearchCV(KernelRidge(kernel='rbf'), grid,
                          cv=min(5, len(tr)), n_jobs=-1)
        gs.fit(Xtr, Ttr)
        th_pred = st.inverse_transform(gs.predict(Xte))
        yp = np.array([reconstruct(row, n_ts, nmu, nh) for row in th_pred])
        y_pred_all[te] = yp
        fr = r2(y[te], yp)
        fold_r2.append(fr)
        bp = gs.best_params_
        print(f"  Fold {fi + 1}/{args.kfold}: R²={fr:.4f}  "
              f"(alpha={bp['alpha']:.3g}, gamma={bp['gamma']:.3g})")

    pooled = r2(y, y_pred_all)
    per_mode = [r2(y[:, :, k], y_pred_all[:, :, k]) for k in range(nmu)]

    print(f"{'=' * 64}")
    print(f"池化 CV R² (全部 {n} 工况): {pooled:.4f}")
    print(f"各折 R²: mean={np.mean(fold_r2):.4f}  std={np.std(fold_r2):.4f}")
    print(f"\n逐模态池化 R²:")
    for k, rk in enumerate(per_mode):
        flag = 'OK' if rk > 0.9 else ('~' if rk > 0.7 else 'X')
        print(f"  a_{k + 1:<2} {rk:>8.4f}  {flag}")
    print(f"{'=' * 64}")
    print(f"距天花板: {pooled:.4f} / {ceil:.4f} = {pooled / ceil * 100:.1f}%")
    print(f"参照: NN(64 32)+ensemble5 的池化 CV R² = 0.8778")


if __name__ == '__main__':
    main()
