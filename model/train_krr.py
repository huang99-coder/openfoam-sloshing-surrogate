#!/usr/bin/env python3
"""
train_krr.py — 超车工况 KRR 正式模型 (全量拟合)

kfold_krr.py 已验证泛化 (池化 CV R² = 0.90 ± 0.001, 3种折划分);
本脚本在全部工况上重新拟合, 超参用内部 5 折 CV 选, 存成 pkl 供 predict 端用。

用法:
  python train_krr.py --data data/shared_overtaking_rich.h5 --n-modes-use 6 --n-harmonics 12 --output-dir results_overtaking_krr
"""
import os
import json
import argparse
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.kernel_ridge import KernelRidge

from kfold_eval import load_cases, drop_constant, r2
from train_podnn import fit_fourier_coeffs, reconstruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n-modes-use', type=int, default=6)
    ap.add_argument('--n-harmonics', type=int, default=12)
    ap.add_argument('--output-dir', default='results_overtaking_krr')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    X, y, names, nmu = load_cases(args.data, args.n_modes_use)
    n_ts = y.shape[1]
    nh = args.n_harmonics
    X, names, _ = drop_constant(X, names)

    theta = np.array([fit_fourier_coeffs(seq, nh) for seq in y], dtype=float)
    ceil = r2(y, np.array([reconstruct(t, n_ts, nmu, nh) for t in theta]))

    print(f"{'=' * 64}")
    print(f"KRR 正式训练: {args.data}")
    print(f"  工况数={len(X)}  输入={names}")
    print(f"  n_modes_use={nmu}  n_harmonics={nh}  n_ts={n_ts}")
    print(f"  Fourier天花板 R²={ceil:.4f}")
    print(f"{'=' * 64}")

    sx, st = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    Ts = st.fit_transform(theta)

    grid = {
        'alpha': np.logspace(-4, 1, 11),
        'gamma': np.logspace(-2.5, 1, 8),
    }
    gs = GridSearchCV(
        KernelRidge(kernel='rbf'), grid,
        cv=KFold(5, shuffle=True, random_state=args.seed), n_jobs=-1)
    gs.fit(Xs, Ts)
    print(f"内部CV选定超参: {gs.best_params_}  (CV score={gs.best_score_:.4f})")

    # 全量 in-sample 自检 (仅确认拟合正常, 泛化以 kfold_krr 为准)
    th_fit = st.inverse_transform(gs.predict(Xs))
    y_fit = np.array([reconstruct(row, n_ts, nmu, nh) for row in th_fit])
    print(f"全量 in-sample R² = {r2(y, y_fit):.4f}  (参考值, 非泛化指标)")

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, 'krr_model.pkl')
    joblib.dump({
        'backend': 'krr',
        'model': gs.best_estimator_,
        'scaler_X': sx,
        'scaler_T': st,
        'input_cols': names,
        'n_timesteps': int(n_ts),
        'n_modes': int(nmu),
        'n_harmonics': int(nh),
        'best_params': gs.best_params_,
        'data': args.data,
        'note': '泛化: kfold_krr 池化CV R²=0.90±0.001 (seeds 42/1/7)',
    }, path)
    print(f"OK Model saved: {path}")
    print(f"预测示例:")
    print(f"  python predict_podnn.py --model {path} overtaking "
          f"--fill 50 --peak-gx <值> --total-time 25 --t-change <值> --double")


if __name__ == '__main__':
    main()
