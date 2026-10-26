#!/usr/bin/env python3
"""
kfold_infold.py — 折内 POD 建基、交叉验证与场级误差分解。

评价方式
--------
infold：仅使用当前训练折计算均值与 POD 基，再投影测试工况。
global：使用全体工况建立的均值与 POD 基作为固定表示对照；测试工况
参与了表示构建，因此该设置的结果描述给定表示下的条件拟合。

折内建基时，各折系数处于不同坐标系。跨折汇总使用与基无关的场空间
误差，逐折系数 R² 单独报告。通过 Gram 矩阵计算正交误差分解：

    ||重构场_j - 真场_j||² = (K_jj - ||a_j||²) + ||â_j - a_j||²

其中 K_jj = ||真场_j - 训练均值||²，a_j 为投影系数，â_j 为预测系数。
右侧分别为 POD 截断误差与回归误差，无需显式生成全部空间模态。

用法
----
  # 折内建基
  python kfold_infold.py --gram gram/gram_slope.npz --desc data/shared_slope.h5 \
      --model fcnn --n-modes-use 6 --n-harmonics 6 --hidden 64 48 --dropout 0.15

  python kfold_infold.py --gram gram/gram_overtaking.npz --desc data/shared_overtaking_rich.h5 \
      --model krr --n-modes-use 6 --n-harmonics 12

  # 全局基对照
  python kfold_infold.py --gram gram/gram_slope.npz --desc data/shared_slope.h5 \
      --model fcnn --n-harmonics 6 --hidden 64 48 --dropout 0.15 --basis global

  # 整档留出
  python kfold_infold.py --gram gram/gram_slope.npz --desc data/shared_slope.h5 \
      --model fcnn --n-harmonics 6 --holdout-col fill_level --holdout-value 70
"""
import os
import json
import argparse
import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.kernel_ridge import KernelRidge

from train_podnn import fit_fourier_coeffs, reconstruct, CASE_PARAM_COLS


# ══════════════════════════════════════════════════════════
# 1. 从 Gram 做折内 POD
# ══════════════════════════════════════════════════════════
def pod_from_gram(G, tau, eps, r):
    """
    tau: 训练列下标 (1D int array)   eps: 测试列下标
    返回 A_tr (r, |tau|), A_te (r, |eps|), Kdiag_te (|eps|,), lam (全部特征值)

    K[i,j] = G[i,j] - g_i - g_j + gbar,  g_i = 行 i 对训练列的均值, gbar = 训练块总均值
    """
    n = len(tau)
    g = G[:, tau].sum(axis=1) / n                       # (N,)  g_i = x_i^T m
    gbar = float(G[np.ix_(tau, tau)].sum()) / (n * n)   # m^T m

    K_tt = G[np.ix_(tau, tau)] - g[tau][:, None] - g[tau][None, :] + gbar
    K_tt = 0.5 * (K_tt + K_tt.T)                        # 对称化, 压掉浮点毛刺

    lam, V = np.linalg.eigh(K_tt)
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0.0, None)
    V = V[:, order]

    rr = int(min(r, np.sum(lam > 1e-12 * max(lam[0], 1e-30))))
    sigma = np.sqrt(lam[:rr])
    A_tr = sigma[:, None] * V[:, :rr].T                 # (rr, n)

    K_te = G[np.ix_(tau, eps)] - g[tau][:, None] - g[eps][None, :] + gbar
    A_te = (V[:, :rr] / sigma).T @ K_te                 # (rr, |eps|)

    Kdiag_te = np.diag(G)[eps] - 2.0 * g[eps] + gbar    # ||x_j - m||²
    return A_tr, A_te, np.clip(Kdiag_te, 0, None), lam


# ══════════════════════════════════════════════════════════
# 2. 回归器
# ══════════════════════════════════════════════════════════
def fit_predict_krr(X_tr, th_tr, X_te, seed=42):
    sx, st = StandardScaler(), StandardScaler()
    Xtr, Xte = sx.fit_transform(X_tr), sx.transform(X_te)
    Ttr = st.fit_transform(th_tr)
    grid = {'alpha': np.logspace(-4, 1, 11), 'gamma': np.logspace(-2.5, 1, 8)}
    gs = GridSearchCV(KernelRidge(kernel='rbf'), grid,
                      cv=min(5, max(2, len(X_tr))), n_jobs=-1)
    gs.fit(Xtr, Ttr)
    return st.inverse_transform(gs.predict(Xte)), gs.best_params_


def fit_predict_fcnn(X_tr, th_tr, X_te, cfg, seed=42):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from pod_fourier_model import create_model

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    nv = min(cfg['n_val'], max(1, len(perm) // 5))
    va, tr = perm[:nv], perm[nv:]

    sx, st = StandardScaler(), StandardScaler()
    Xtr = sx.fit_transform(X_tr[tr]); Xva = sx.transform(X_tr[va]); Xte = sx.transform(X_te)
    Ttr = st.fit_transform(th_tr[tr]); Tva = st.transform(th_tr[va])

    torch.manual_seed(seed)
    model = create_model(Xtr.shape[1], Ttr.shape[1],
                         hidden_dims=tuple(cfg['hidden']), dropout=cfg['dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(50, min(cfg['epochs'] // 3, 200)), T_mult=2, eta_min=1e-6)
    loss_fn = nn.MSELoss()
    tl = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ttr)),
                    batch_size=min(cfg['batch_size'], len(Xtr)), shuffle=True)
    Xva_t, Tva_t = torch.FloatTensor(Xva), torch.FloatTensor(Tva)

    best, best_state, no_imp = float('inf'), None, 0
    for _ in range(cfg['epochs']):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), Tva_t).item()
        if vl < best:
            best, no_imp = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        th = model(torch.FloatTensor(Xte)).numpy()
    return st.inverse_transform(th), {'best_val_loss': round(best, 5)}


# ══════════════════════════════════════════════════════════
# 3. 描述符
# ══════════════════════════════════════════════════════════
def load_descriptors(desc_h5):
    with h5py.File(desc_h5, 'r') as f:
        Xall = f['X'][()]
        bd = f['case_boundaries'][()]
        feat = json.loads(f['metadata'].attrs['feature_names'])
        cols = (json.loads(f['metadata'].attrs['input_cols'])
                if 'input_cols' in f['metadata'].attrs else CASE_PARAM_COLS)
        names_in_file = None
        if 'case_info' in f:
            names_in_file = [f['case_info'][f'case_{i:03d}'].attrs.get('file')
                             for i in range(len(bd)) if f'case_{i:03d}' in f['case_info']]
    idx = [feat.index(c) for c in cols if c in feat]
    names = [c for c in cols if c in feat]
    X = np.array([Xall[s, idx] for s, _ in bd], dtype=float)
    return X, names, names_in_file


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gram', required=True)
    ap.add_argument('--desc', required=True, help='描述符来源 h5 (shared_*.h5 或 *_rich.h5)')
    ap.add_argument('--model', choices=['fcnn', 'krr'], default='fcnn')
    ap.add_argument('--basis', choices=['infold', 'global'], default='infold',
                    help='infold=训练折内建基; global=全体工况建基的固定表示对照')
    ap.add_argument('--n-modes-use', type=int, default=6)
    ap.add_argument('--n-harmonics', type=int, default=6)
    ap.add_argument('--kfold', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    # 整档留出 (P4)
    ap.add_argument('--holdout-col', default=None)
    ap.add_argument('--holdout-value', type=float, default=None)
    # FCNN
    ap.add_argument('--hidden', type=int, nargs='+', default=[64, 48])
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--patience', type=int, default=80)   # 与原 kfold_eval.py 一致
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--n-val', type=int, default=3)      # 与 kfold_eval.py 的验证集规模一致
    ap.add_argument('--csv', default=None, help='逐工况结果写到这个 csv')
    args = ap.parse_args()
    cfg = vars(args)

    z = np.load(args.gram, allow_pickle=True)
    G = z['G']; T = int(z['T']); n_cases = int(z['n_cases'])
    gram_files = [str(x) for x in z['case_files']]
    src = str(z['field_source']); vw = bool(z['volume_weighted'])

    Xc, names, desc_files = load_descriptors(args.desc)
    if len(Xc) != n_cases:
        raise SystemExit(f"❌ 工况数不一致: gram={n_cases}, desc={len(Xc)}。"
                         f"两边要用同一批算例、同一 --types。")
    if desc_files and [str(a) for a in desc_files] != gram_files:
        print("  ⚠ gram 与 desc 的工况文件名顺序不完全一致, 请核对后再信结果")

    # 训练折内再剔常量列（原脚本是在全量上剔的, 属轻微泄漏）
    r_use, nh = args.n_modes_use, args.n_harmonics
    N = n_cases * T
    cols_of = lambda cs: np.concatenate([np.arange(c*T, (c+1)*T) for c in cs])

    print(f"{'='*72}")
    print(f"折内建基 CV   basis={args.basis}   model={args.model}")
    print(f"  {args.gram}   工况={n_cases}  T={T}  N={N}  取场={src}  体积加权={'是' if vw else '否'}")
    print(f"  描述符={names}")
    print(f"  r={r_use}  H={nh}" + (f"  hidden={args.hidden}" if args.model == 'fcnn' else ""))
    if src == 'recon':
        print("  ⚠ 取场=recon: 场本身是各算例自身 POD 的低秩重构, 下面的场误差是相对低秩场的")
    print(f"{'='*72}")

    # ---------- 划分 ----------
    if args.holdout_col:
        if args.holdout_col not in names:
            raise SystemExit(f"❌ {args.holdout_col} 不在描述符 {names} 里")
        v = Xc[:, names.index(args.holdout_col)]
        te = np.where(np.isclose(v, args.holdout_value))[0]
        tr = np.where(~np.isclose(v, args.holdout_value))[0]
        if len(te) == 0:
            uniq = np.unique(np.round(v, 6))
            raise SystemExit(
                f"❌ 没有 {args.holdout_col}={args.holdout_value} 的工况\n"
                f"   该列实际取值: {list(uniq)}\n"
                f"   注意 fill_level 在 H5 里是 0~1 的小数（fill70 → 0.7），不是百分数")
        splits = [(tr, te)]
        print(f"整档留出: {args.holdout_col}={args.holdout_value} 共 {len(te)} 组做测试, "
              f"{len(tr)} 组训练")
    else:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        splits = list(kf.split(np.arange(n_cases)))

    # ---------- 逐折 ----------
    rows = []
    tot = dict(den=0.0, trunc=0.0, four=0.0, total=0.0)
    fold_coef_r2 = []
    pool_true, pool_pred = [], []      # 仅 basis=global 时可跨折池化

    for fi, (tr_c, te_c) in enumerate(splits):
        tau = cols_of(sorted(tr_c)) if args.basis == 'infold' else np.arange(N)
        eps = cols_of(sorted(te_c))
        A_tau, A_te, Kdiag, lam = pod_from_gram(G, tau, eps, r_use)
        rr = A_tau.shape[0]

        # 训练工况的系数序列（basis=global 时 tau 是全量, 要取出训练工况那部分）
        if args.basis == 'infold':
            y_tr = A_tau.T.reshape(len(tr_c), T, rr)
        else:
            y_tr = A_tau.T.reshape(n_cases, T, rr)[sorted(tr_c)]
        y_te = A_te.T.reshape(len(te_c), T, rr)

        # 训练折内剔常量列
        Xtr_raw, Xte_raw = Xc[sorted(tr_c)], Xc[sorted(te_c)]
        keep = [j for j in range(Xtr_raw.shape[1]) if Xtr_raw[:, j].std() > 1e-10]
        Xtr, Xte = Xtr_raw[:, keep], Xte_raw[:, keep]
        if fi == 0:
            print(f"  折内实际使用的描述符({len(keep)}): {[names[j] for j in keep]}")

        th_tr = np.array([fit_fourier_coeffs(s, nh) for s in y_tr])
        if args.model == 'krr':
            th_hat, extra = fit_predict_krr(Xtr, th_tr, Xte, seed=args.seed + fi)
        else:
            th_hat, extra = fit_predict_fcnn(Xtr, th_tr, Xte, cfg,
                                             seed=args.seed + 1000 * fi)  # 同原 kfold_eval.py
        y_hat = np.array([reconstruct(t, T, rr, nh) for t in th_hat])

        # 傅里叶天花板: 用真实测试系数重拟合
        th_ceil = np.array([fit_fourier_coeffs(s, nh) for s in y_te])
        y_ceil = np.array([reconstruct(t, T, rr, nh) for t in th_ceil])

        # ---- 场空间误差分解（分母 = 测试快照到训练均值的距离平方之和）----
        den = float(Kdiag.sum())
        a_true = y_te.reshape(-1, rr)                       # (|eps|, rr)
        e_trunc = float(Kdiag.sum() - np.sum(a_true ** 2))   # POD 截断
        e_four = float(np.sum((y_ceil.reshape(-1, rr) - a_true) ** 2))
        e_reg = float(np.sum((y_hat.reshape(-1, rr) - a_true) ** 2))
        e_total = e_trunc + e_reg

        coef_r2 = r2(y_te, y_hat)
        fold_coef_r2.append(coef_r2)
        if args.basis == 'global':
            pool_true.append(y_te); pool_pred.append(y_hat)
        for k in ('den', 'trunc', 'four', 'total'):
            tot[k] += {'den': den, 'trunc': e_trunc, 'four': e_four, 'total': e_total}[k]

        cum6 = lam[:rr].sum() / lam.sum()
        print(f"  Fold {fi+1}/{len(splits)}  n_te={len(te_c):2d}  "
              f"系数R²(折内)={coef_r2:.4f}   "
              f"场相对L2: 截断={np.sqrt(e_trunc/den):.1%} "
              f"回归={np.sqrt(e_reg/den):.1%} "
              f"合计={np.sqrt(e_total/den):.1%}   "
              f"训练折前{rr}阶能量={cum6:.3f}  {extra}")

        for i, c in enumerate(sorted(te_c)):
            kj = Kdiag[i*T:(i+1)*T].sum()
            tr_i = kj - np.sum(a_true[i*T:(i+1)*T] ** 2)
            rg_i = np.sum((y_hat[i] - y_te[i]) ** 2)
            rows.append(dict(fold=fi+1, case=c, file=gram_files[c],
                             rel_trunc=np.sqrt(max(tr_i, 0)/kj),
                             rel_reg=np.sqrt(rg_i/kj),
                             rel_total=np.sqrt(max(tr_i+rg_i, 0)/kj)))

    # ---------- 汇总 ----------
    d = tot['den']
    print(f"{'='*72}")
    print(f"逐折系数 R²(各自基内): mean={np.mean(fold_coef_r2):.4f}  "
          f"std={np.std(fold_coef_r2):.4f}")
    if args.basis == 'global' and pool_true:
        pr2 = r2(np.concatenate(pool_true), np.concatenate(pool_pred))
        print(f"池化系数 R²(全局固定基): {pr2:.4f}")
    elif args.basis == 'infold':
        print("  (折内建基时各折坐标系不同, 系数 R² 不可跨折池化; "
              "可池化的是下面的场级口径)")
    print(f"\n场空间误差分解（基无关，支持跨折汇总）")
    print(f"   分母 Σ||α−α_train_mean||²  = {d:.6g}")
    print(f"   ① POD 截断 (r={r_use})        相对L2 = {np.sqrt(tot['trunc']/d):>7.2%}"
          f"   （占误差平方 {tot['trunc']/tot['total']:.1%}）")
    print(f"   ② 傅里叶时间参数化 (H={nh})  相对L2 = {np.sqrt(tot['four']/d):>7.2%}"
          f"   （含在③里, 是③的下界）")
    print(f"   ③ 回归误差                  相对L2 = "
          f"{np.sqrt((tot['total']-tot['trunc'])/d):>7.2%}")
    print(f"   ④ 合计 ①⊕③                 相对L2 = {np.sqrt(tot['total']/d):>7.2%}")
    print(f"\n   场级 R²_field = 1 − ④² = {1 - tot['total']/d:.4f}")
    print(f"   完美回归的上限 (只剩截断)  = {1 - tot['trunc']/d:.4f}")
    print(f"{'='*72}")
    print("读法: 如果 ① 远大于 ③, 说明瓶颈在 POD 阶数不够, 加模态比换模型有用;")
    print("      如果 ③ 远大于 ①, 说明瓶颈在回归/描述符, 该加样本或换输入;")
    print("      如果 ② 接近 ③, 说明傅里叶参数化本身就是天花板, 该提 H 或换时间表示。")

    if args.csv and rows:
        import csv as _csv
        with open(args.csv, 'w', newline='', encoding='utf-8-sig') as fo:
            wtr = _csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            wtr.writeheader(); wtr.writerows(rows)
        print(f"逐工况结果 → {args.csv}")


if __name__ == '__main__':
    main()
