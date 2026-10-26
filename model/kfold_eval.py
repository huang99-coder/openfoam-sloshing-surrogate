#!/usr/bin/env python3
"""
kfold_eval.py  —  POD-NN 的 K 折交叉验证评估

为什么需要它
------------
每类只有 18~30 个工况, 单次 train/val/test 划分里测试集只有 3~4 个,
R² 的随机波动比超参影响还大。K 折让【每个工况都轮流当一次测试】,
给出可信的总指标 (池化 CV R²) 和波动范围 (各折 mean±std)。

用法
----
  python kfold_eval.py --data data/shared_slope.h5      --n-modes-use 6 --n-harmonics 6 --hidden 48 32 --kfold 5
  python kfold_eval.py --data data/shared_turning.h5    --n-modes-use 6 --n-harmonics 8 --hidden 48 32 --kfold 5 --weight-decay 0.1
  python kfold_eval.py --data data/shared_overtaking.h5 --n-modes-use 5 --n-harmonics 6 --hidden 32 16 --kfold 6
"""

import argparse
import json
import numpy as np
import h5py
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from train_podnn import fourier_basis, fit_fourier_coeffs, reconstruct, CASE_PARAM_COLS
from pod_fourier_model import create_model


# ──────────────────────────────────────────────────────────
def load_cases(h5_path, n_modes_use):
    with h5py.File(h5_path, 'r') as f:
        X_all = f['X'][()]
        y_all = f['y'][()]
        bd = f['case_boundaries'][()]
        feat = json.loads(f['metadata'].attrs['feature_names'])
        n_modes = int(f['metadata'].attrs['n_modes'])
        # rich 文件: 用它自带的 input_cols (真实物理输入), 不走 CASE_PARAM_COLS 白名单
        input_cols = None
        if 'input_cols' in f['metadata'].attrs:
            input_cols = json.loads(f['metadata'].attrs['input_cols'])

    sel = input_cols if input_cols else CASE_PARAM_COLS
    idx = [feat.index(c) for c in sel if c in feat]
    names = [c for c in sel if c in feat]
    nmu = min(n_modes_use or n_modes, n_modes)

    Ts = [int(e - s) for s, e in bd]
    Tm = min(Ts)
    if len(set(Ts)) > 1:
        print(f"  ⚠ 各工况时间步不一致 {sorted(set(Ts))}, 统一截断到 T={Tm}")
    Xc, yc = [], []
    for s, e in bd:
        Xc.append(X_all[s, idx])
        yc.append(y_all[s:s + Tm, :nmu])
    return np.asarray(Xc, float), np.asarray(yc, float), names, nmu


def drop_constant(X, names):
    keep = [j for j in range(X.shape[1]) if X[:, j].std() > 1e-10]
    return X[:, keep], [names[j] for j in keep], keep


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


# ──────────────────────────────────────────────────────────
def train_one_fold(X_tr, T_tr, X_va, T_va, cfg, w=None):
    """训练一个 fold, 返回 (model, scaler_X, scaler_T); w 为逐输出维loss权重"""
    sx, st = StandardScaler(), StandardScaler()
    Xtr = sx.fit_transform(X_tr); Xva = sx.transform(X_va)
    Ttr = st.fit_transform(T_tr); Tva = st.transform(T_va)

    bs = min(cfg['batch_size'], len(X_tr))
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ttr)),
                    batch_size=bs, shuffle=True)

    model = create_model(X_tr.shape[1], T_tr.shape[1],
                         hidden_dims=tuple(cfg['hidden']), dropout=cfg['dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(50, min(cfg['epochs'] // 3, 200)), T_mult=2, eta_min=1e-6)
    if w is None:
        loss_fn = nn.MSELoss()
    else:
        def loss_fn(pred, target):
            return (w * (pred - target) ** 2).mean()

    Xva_t = torch.FloatTensor(Xva); Tva_t = torch.FloatTensor(Tva)
    best, best_state, no_imp = float('inf'), None, 0
    for ep in range(cfg['epochs']):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), Tva_t).item()
        if vl < best:
            best, best_state, no_imp = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    return model, sx, st


def predict_seq(model, sx, st, X_cases, n_ts, nmu, nh):
    Xsc = sx.transform(X_cases)
    with torch.no_grad():
        theta_sc = model(torch.FloatTensor(Xsc)).numpy()
    theta = st.inverse_transform(theta_sc)
    return np.array([reconstruct(row, n_ts, nmu, nh) for row in theta])


# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n-modes-use', type=int, default=6)
    ap.add_argument('--n-harmonics', type=int, default=6)
    ap.add_argument('--hidden', type=int, nargs='+', default=[48, 32])
    ap.add_argument('--dropout', type=float, default=0.20)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--patience', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--kfold', type=int, default=5)
    ap.add_argument('--n-val', type=int, default=3, help='每折从训练池里carve出多少做早停')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ensemble', type=int, default=1,
                    help='每折训练N个不同初始化的模型, 预测取平均')
    ap.add_argument('--mode-weight', choices=['none', 'energy'], default='none',
                    help='按模态能量(奇异值平方)给loss加权')
    args = ap.parse_args()
    cfg = vars(args)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, y, names, nmu = load_cases(args.data, args.n_modes_use)
    n_ts = y.shape[1]; nh = args.n_harmonics
    X, names, _ = drop_constant(X, names)

    # 模态能量 loss 权重: 标准化θ空间下, 大模态的误差罚得更重,
    # 让 NN 容量向 a_1~a_4 倾斜, 而不是去追高阶模态谐波里的噪声
    w = None
    if args.mode_weight == 'energy':
        with h5py.File(args.data, 'r') as f:
            sv = f['singular_values'][()] if 'singular_values' in f else None
        if sv is None:
            print("⚠ 文件无 singular_values, --mode-weight energy 退化为等权")
        else:
            e_mode = np.asarray(sv[:nmu], float) ** 2
            w_mode = e_mode / e_mode.mean()
            w = torch.FloatTensor(np.repeat(w_mode, 1 + 2 * nh))
            print(f"模态loss权重: {np.round(w_mode, 3)}")

    # 目标: 每个工况的 Fourier 系数
    theta = np.array([fit_fourier_coeffs(seq, nh) for seq in y], dtype=float)

    # Fourier 天花板
    y_rec = np.array([reconstruct(t, n_ts, nmu, nh) for t in theta])
    ceil = r2(y, y_rec)

    n = len(X)
    print(f"{'='*64}")
    print(f"K折CV: {args.data}")
    print(f"  工况数={n}  有效输入={names}")
    print(f"  n_modes_use={nmu}  n_harmonics={nh}  hidden={args.hidden}")
    print(f"  dropout={args.dropout}  wd={args.weight_decay}  kfold={args.kfold}  "
          f"ensemble={args.ensemble}  mode_weight={args.mode_weight}")
    print(f"  Fourier天花板 R²={ceil:.4f}")
    print(f"{'='*64}")

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    y_pred_all = np.zeros_like(y)           # 每个工况都会被预测恰好一次
    fold_r2 = []

    for fi, (tr_pool, te) in enumerate(kf.split(np.arange(n))):
        rng = np.random.default_rng(args.seed + fi)
        pool = rng.permutation(tr_pool)
        nv = min(args.n_val, max(1, len(pool) // 5))
        va, tr = pool[:nv], pool[nv:]

        yp_members = []
        for j in range(max(1, args.ensemble)):
            torch.manual_seed(args.seed + 1000 * fi + j)   # 成员间仅初始化/shuffle不同
            model, sx, st = train_one_fold(X[tr], theta[tr], X[va], theta[va], cfg, w)
            yp_members.append(predict_seq(model, sx, st, X[te], n_ts, nmu, nh))
        yp = np.mean(yp_members, axis=0)
        y_pred_all[te] = yp
        fr = r2(y[te], yp)
        fold_r2.append(fr)
        print(f"  Fold {fi+1}/{args.kfold}: test={list(te)}  R²={fr:.4f}")

    # 池化 CV R²（所有工况合起来算一次，最稳）
    pooled = r2(y, y_pred_all)
    per_mode = [r2(y[:, :, k], y_pred_all[:, :, k]) for k in range(nmu)]

    print(f"{'='*64}")
    print(f"池化 CV R² (全部 {n} 工况): {pooled:.4f}")
    print(f"各折 R²: mean={np.mean(fold_r2):.4f}  std={np.std(fold_r2):.4f}  "
          f"min={np.min(fold_r2):.4f}  max={np.max(fold_r2):.4f}")
    print(f"\n逐模态池化 R²:")
    for k, rk in enumerate(per_mode):
        flag = 'OK' if rk > 0.9 else ('~' if rk > 0.7 else 'X')
        print(f"  a_{k+1:<2} {rk:>8.4f}  {flag}")
    print(f"{'='*64}")
    print(f"距天花板: {pooled:.4f} / {ceil:.4f} = {pooled/ceil*100:.1f}%")


if __name__ == '__main__':
    main()