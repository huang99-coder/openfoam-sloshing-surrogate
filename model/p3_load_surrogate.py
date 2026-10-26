#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p3_load_surrogate.py (v3)
==========================
工况参数 -> 载荷时程 + 峰值代理。

v3 相对 v2 的统计口径修正
--------------------------
1. 【严格训练折归一化】载荷各通道 RMS 尺度只用当前训练折计算，测试折不参与
   载荷 POD 的尺度、均值和基构造。
2. 【标准测试集 R²】R² 分母改为测试数据自身均值，而不是训练均值/训练均值时程。
   k 折时按各测试折自己的均值累计 SSE/SST；整档留出时就是标准 holdout R²。
3. 【严格联合空间误差分解】POD 截断/回归/总误差只在“训练折归一化后的联合载荷
   向量空间”中分解，此时正交恒等式严格成立。逐物理通道不再强行用
   total²-trunc² 反推回归误差。
4. 【holdout 不偷看测试集选模型】整档留出时，峰值头的模型族默认只在训练集内部
   做内层 CV 选择，再锁定后读取 holdout 指标；不会根据 holdout 哪个模型分数最高
   再挑模型。四个候选模型的 holdout 表仍全部打印，便于诊断。
5. 【汇总表标明口径】summary CSV 增加模型选择口径和 R² 口径，避免把探索性五折
   最优与确认性 holdout 混为一谈。

说明
----
* 顶部“载荷 POD 谱”是全数据描述性谱，只用于说明低秩程度，不参与任何 CV/holdout
  训练流程，因此不会进入预测评价。
* 正式通道集合已按工况物理对称性预先锁定：坡道/制动 Fy,Fz,Mx；转弯 Fx,Fy,Mz；
  变道六通道全保留。--keep-channels 仅用于噪声诊断时额外强制保留。

典型用法
--------
  # 五折（探索性模型比较）
  python p3_load_surrogate_v3.py --loads loads_all.npz --type slope \
      --summary-append results/p3_summary_v3.csv

  # 整档留出（默认训练集内 inner-CV 锁定峰值模型族）
  python p3_load_surrogate_v3.py --loads loads_all.npz --type slope \
      --holdout-col fill_level --holdout-value 0.7 \
      --summary-append results/p3_summary_v3.csv
"""

import argparse
import csv as _csv
import json
import os as _os
import warnings

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')

META_KEYS = ('n_cell', 'n_mode', 'n_snapshot', 'n_point', 'n_ts', 'index')

# 按几何/激励对称性预先定义进入正式代理评价的物理通道。
# 这样 holdout 的通道选择不依赖任何测试目标数据。
PHYSICAL_KEEP = {
    'slope': {'Fy', 'Fz', 'Mx'},
    'braking': {'Fy', 'Fz', 'Mx'},
    'turning': {'Fx', 'Fy', 'Mz'},
    'overtaking': {'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'},
}


def load_npz(path, ctype, min_ratio, force_keep=()):
    z = np.load(path, allow_pickle=True)
    ct = np.array([str(x) for x in z['case_type']])
    sel = np.where(ct == ctype)[0]
    if len(sel) == 0:
        raise SystemExit(f"❌ {path} 里没有 case_type={ctype}；实际有 {sorted(set(ct))}")

    Y = np.asarray(z['loads'], dtype=np.float64)[sel]
    peak = np.asarray(z['raw_peak_signed'], dtype=np.float64)[sel]
    chan = [str(c) for c in z['channels']]
    unit = [str(u) for u in z['units']]
    ids = [str(c) for c in np.array(z['case_id'])[sel]]
    par = [json.loads(str(p)) for p in np.array(z['params_json'])[sel]]

    for j in range(Y.shape[2]):
        m = np.isnan(Y[:, :, j])
        if m.any():
            Y[:, :, j][m] = np.nanmean(Y[:, :, j]) if not np.all(m) else 0.0
    peak = np.nan_to_num(peak)

    # RMS 这里只用于诊断量级；正式通道集合由 PHYSICAL_KEEP 先验定义，
    # 因而不会因为看了 holdout 目标数据才决定保留/剔除。
    rms = np.array([np.sqrt(np.mean(Y[:, :, j] ** 2)) for j in range(Y.shape[2])])
    keep = []
    print("\n  通道量级（全数据诊断；正式通道按物理先验锁定，POD 尺度按训练折重算）")
    for j in range(Y.shape[2]):
        grp = [k for k in range(Y.shape[2]) if unit[k] == unit[j]]
        ratio = rms[j] / max(rms[grp].max(), 1e-30)
        u = 'kN' if unit[j] == 'N' else 'kN·m'
        forced = chan[j] in force_keep
        prior_keep = chan[j] in PHYSICAL_KEEP.get(ctype, set(chan))
        ok = prior_keep or forced
        tag = ('保留（--keep-channels 强制）' if forced and not prior_keep
               else '保留（物理先验）' if prior_keep
               else '★剔除（物理对称性先验：该分量不进入正式代理）')
        print(f"     {chan[j]:<4} RMS={rms[j]/1e3:>10.3f} {u:<5} 占比={ratio:>7.2%}   {tag}")
        if ok:
            keep.append(j)

    return (Y[:, :, keep], peak[:, keep], [chan[j] for j in keep],
            [unit[j] for j in keep], ids, par)


def build_descriptors(params, extra_h5=None, n_expect=None):
    if extra_h5:
        import h5py
        with h5py.File(extra_h5, 'r') as f:
            Xall = f['X'][()]
            bd = f['case_boundaries'][()]
            feat = json.loads(f['metadata'].attrs['feature_names'])
            cols = (json.loads(f['metadata'].attrs['input_cols'])
                    if 'input_cols' in f['metadata'].attrs else
                    [c for c in feat if c != 'time'])
        idx = [feat.index(c) for c in cols if c in feat]
        names = [c for c in cols if c in feat]
        X = np.array([Xall[s, idx] for s, _ in bd], dtype=float)
        if n_expect is not None and len(X) != n_expect:
            raise SystemExit(f"❌ --extra-desc 工况数 {len(X)} ≠ 载荷里的 {n_expect}")
        src = extra_h5
    else:
        keys = sorted({k for p in params for k, v in p.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)})
        keys = [k for k in keys if not any(t in k.lower() for t in META_KEYS)]
        X = np.array([[float(p.get(k, np.nan)) for k in keys] for p in params])
        ok = [j for j in range(X.shape[1]) if not np.isnan(X[:, j]).any()]
        X, names = X[:, ok], [keys[j] for j in ok]
        src = 'params_json（已剔除网格/POD 元数据）'

    keep = [j for j in range(X.shape[1]) if X[:, j].std() > 1e-10]
    dropped = [names[j] for j in range(X.shape[1]) if j not in keep]
    X, names = X[:, keep], [names[j] for j in keep]
    print(f"\n  描述符来源: {src}")
    print(f"     实际使用({len(names)}): {names}")
    if dropped:
        print(f"     常量已剔除({len(dropped)}): {dropped}")
    return X, names


def make_models(n_train):
    cv = min(5, max(2, n_train // 4))
    return {
        '线性(Ridge)': GridSearchCV(
            Ridge(), {'alpha': np.logspace(-3, 3, 13)}, cv=cv
        ),
        '最近邻(kNN)': GridSearchCV(
            KNeighborsRegressor(),
            {'n_neighbors': [1, 2, 3, 5], 'weights': ['uniform', 'distance']},
            cv=cv
        ),
        '核岭(KRR-rbf)': GridSearchCV(
            KernelRidge(kernel='rbf'),
            {'alpha': np.logspace(-4, 1, 11),
             'gamma': np.logspace(-2.5, 1, 8)},
            cv=cv
        ),
        'MLP(64-48)': MLPRegressor(
            hidden_layer_sizes=(64, 48), max_iter=4000,
            early_stopping=True, n_iter_no_change=50,
            validation_fraction=0.15, alpha=1e-3,
            random_state=0
        ),
    }


def fit_predict(model, X_tr, Y_tr, X_te):
    """所有 scaler 只在传入的训练数据上拟合。"""
    sx, sy = StandardScaler(), StandardScaler()
    Xtr = sx.fit_transform(X_tr)
    Xte = sx.transform(X_te)
    Ytr = sy.fit_transform(Y_tr)
    model.fit(Xtr, Ytr)
    pred = np.atleast_2d(model.predict(Xte))
    return sy.inverse_transform(pred)


def safe_r2(sse, sst):
    return 1.0 - sse / max(sst, 1e-30)


def inner_select_peak_models(Xtr, Ptr, mnames, seed=42, n_splits=4):
    """
    只在外层训练集内部选择峰值头模型族；外层 holdout 完全不参与。

    每个物理通道可选择不同模型族。每个候选模型内部仍按 make_models() 的规则
    在 inner-train 上调超参数，因此所有选择步骤都封闭在外层训练集内。
    """
    n = len(Xtr)
    k = min(n_splits, max(2, n // 8), n)
    if k < 2:
        return {j: mnames[0] for j in range(Ptr.shape[1])}, None

    kf = KFold(k, shuffle=True, random_state=seed + 1701)
    sse = {m: np.zeros(Ptr.shape[1], dtype=float) for m in mnames}
    sst = np.zeros(Ptr.shape[1], dtype=float)

    print(f"\n  holdout 模型锁定：只用训练集做 {k} 折 inner-CV（测试档位完全未参与）")
    for ii, (itr, iva) in enumerate(kf.split(np.arange(n)), 1):
        for j in range(Ptr.shape[1]):
            yv = Ptr[iva, j]
            sst[j] += np.sum((yv - yv.mean()) ** 2)
        for m in mnames:
            ph = fit_predict(make_models(len(itr))[m], Xtr[itr], Ptr[itr], Xtr[iva])
            sse[m] += np.sum((ph - Ptr[iva]) ** 2, axis=0)

    pick = {}
    for j in range(Ptr.shape[1]):
        bm = min(mnames, key=lambda m: sse[m][j])
        pick[j] = bm
    return pick, (sse, sst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loads', default='loads_all.npz')
    ap.add_argument('--type', required=True,
                    choices=['slope', 'braking', 'turning', 'overtaking'])
    ap.add_argument('--extra-desc', default=None)
    ap.add_argument('--r-loads', type=int, default=20)
    ap.add_argument('--min-channel-ratio', type=float, default=0.02)
    ap.add_argument('--keep-channels', nargs='*', default=[])
    ap.add_argument('--mref', type=float, default=273.6)
    ap.add_argument('--kfold', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--holdout-col', default=None)
    ap.add_argument('--holdout-value', type=float, default=None)
    ap.add_argument('--inner-select-folds', type=int, default=4,
                    help='整档留出时，只在训练集内选择峰值模型族的 inner-CV 折数')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--summary-append', default=None)
    args = ap.parse_args()

    if (args.holdout_col is None) ^ (args.holdout_value is None):
        raise SystemExit('❌ --holdout-col 与 --holdout-value 必须同时给出')

    print('=' * 80)
    print(f"P3 载荷代理 v3   {args.loads}   工况类型={args.type}")
    print('=' * 80)

    Y, peak, chan, unit, ids, par = load_npz(
        args.loads, args.type, args.min_channel_ratio, tuple(args.keep_channels)
    )
    n, T, C = Y.shape
    X, names = build_descriptors(par, args.extra_desc, n_expect=n)
    print(f"\n  工况={n}  时间点={T}  保留通道={chan}")

    # ------------------------------------------------------------------
    # 全数据描述性 POD 谱：仅说明低秩程度，不进入任何预测折。
    # ------------------------------------------------------------------
    scale_diag = np.array([
        max(np.sqrt(np.mean(Y[:, :, j] ** 2)), 1e-30) for j in range(C)
    ])
    Fdiag = (Y / scale_diag[None, None, :]).reshape(n, T * C)
    ev = np.linalg.svd(Fdiag - Fdiag.mean(axis=0, keepdims=True),
                       compute_uv=False) ** 2
    cum = np.cumsum(ev) / max(ev.sum(), 1e-30)
    print("\n  ★ 载荷 POD 谱（全数据描述性诊断，不进入 CV/holdout 训练）")
    print(f"     {'r':>4} {'累计能量':>10} {'时程截断相对L2':>16}")
    for r in [1, 2, 4, 6, 8, 12, 16, 20, 25, 30, 40]:
        if r <= len(ev):
            print(f"     {r:>4} {cum[r-1]:>10.4f} "
                  f"{np.sqrt(max(1-cum[r-1], 0)):>15.2%}")
    for f in (0.95, 0.99):
        k = int(np.searchsorted(cum, f) + 1)
        print(f"     → 达 {f:.0%} 能量需 r = {k}")

    # ------------------------------------------------------------------
    # 外层划分
    # ------------------------------------------------------------------
    is_holdout = args.holdout_col is not None
    if is_holdout:
        if args.holdout_col not in names:
            raise SystemExit(f"❌ {args.holdout_col} 不在描述符 {names} 里")
        v = X[:, names.index(args.holdout_col)]
        te0 = np.where(np.isclose(v, args.holdout_value))[0]
        if len(te0) == 0:
            raise SystemExit(
                f"❌ 没有 {args.holdout_col}={args.holdout_value}；"
                f"该列取值 {list(np.unique(np.round(v, 6)))}"
            )
        tr0 = np.where(~np.isclose(v, args.holdout_value))[0]
        splits = [(tr0, te0)]
        print(f"\n  整档留出: {args.holdout_col}={args.holdout_value}，"
              f"{len(te0)} 组测试 / {len(tr0)} 组训练")
        print("  评价范围：结构化档位留出；训练参数范围内的留出值对应插值评价，"
              "训练参数范围外的留出值对应外推评价。")
    else:
        splits = list(KFold(args.kfold, shuffle=True,
                            random_state=args.seed).split(np.arange(n)))
        print(f"\n  {args.kfold} 折交叉验证（工况级；载荷尺度、POD 基、X/Y 标准化均在训练折内构建）")

    rl = min(args.r_loads, max(2, min(len(tr) for tr, _ in splits) - 1))
    print(f"  载荷 POD 保留 r_L = {rl}"
          + ("" if rl == args.r_loads else f"（受训练工况数限制，请求值为 {args.r_loads}）"))

    mnames = list(make_models(10).keys())

    # holdout 的峰值模型族在训练集内部预先锁定。
    locked_peak = None
    inner_diag = None
    if is_holdout:
        tr0, _ = splits[0]
        locked_peak, inner_diag = inner_select_peak_models(
            X[tr0], peak[tr0], mnames, seed=args.seed,
            n_splits=args.inner_select_folds
        )
        print("  训练集内锁定结果：")
        for j, c in enumerate(chan):
            print(f"     {c:<4} -> {locked_peak[j]}")

    # 评价累计器。R² 的 SST 均由当前测试折自身均值构造。
    acc = {
        m: dict(
            sse=np.zeros(C), sst=np.zeros(C),
            psse=np.zeros(C), psst=np.zeros(C), pabs=np.zeros(C),
            pn=0, tn=0,
            joint_reg=0.0, joint_total=0.0
        ) for m in mnames
    }
    joint_trunc = 0.0
    joint_signal = 0.0
    joint_identity_abs = 0.0
    rows = []
    eval_indices = []

    for fi, (tr, te) in enumerate(splits):
        eval_indices.extend(te.tolist())

        # --------------------------------------------------------------
        # v3 核心：载荷尺度只看训练折。
        # --------------------------------------------------------------
        scale_tr = np.array([
            max(np.sqrt(np.mean(Y[tr, :, j] ** 2)), 1e-30) for j in range(C)
        ])
        Ftr = (Y[tr] / scale_tr[None, None, :]).reshape(len(tr), T * C)
        Fte = (Y[te] / scale_tr[None, None, :]).reshape(len(te), T * C)

        mu = Ftr.mean(axis=0)
        _, _, Vt = np.linalg.svd(Ftr - mu, full_matrices=False)
        B = Vt[:rl]
        A_tr = (Ftr - mu) @ B.T
        A_te = (Fte - mu) @ B.T
        F_proj = A_te @ B + mu

        Y_proj = F_proj.reshape(len(te), T, C) * scale_tr[None, None, :]
        Yt = Y[te]

        # 标准测试 R²：每个测试折用自己的真实目标均值作基线。
        for j in range(C):
            ytj = Yt[:, :, j]
            sstj = np.sum((ytj - ytj.mean()) ** 2)
            for m in mnames:
                acc[m]['sst'][j] += sstj

        # 联合归一化空间中的严格正交分解公共项。
        centered = Fte - mu
        joint_signal += np.sum(centered ** 2)
        this_trunc = np.sum((Fte - F_proj) ** 2)
        joint_trunc += this_trunc

        for mname in mnames:
            A_hat = fit_predict(make_models(len(tr))[mname], X[tr], A_tr, X[te])
            F_hat = A_hat @ B + mu
            Yh = F_hat.reshape(len(te), T, C) * scale_tr[None, None, :]

            for j in range(C):
                acc[mname]['sse'][j] += np.sum((Yh[:, :, j] - Yt[:, :, j]) ** 2)

            # 严格联合空间误差分解：F_proj 与 F_hat 都在同一训练 POD 仿射子空间内。
            reg = np.sum((F_hat - F_proj) ** 2)
            tot = np.sum((F_hat - Fte) ** 2)
            acc[mname]['joint_reg'] += reg
            acc[mname]['joint_total'] += tot
            joint_identity_abs += abs(tot - (this_trunc + reg))

            P_hat = fit_predict(make_models(len(tr))[mname], X[tr], peak[tr], X[te])
            for j in range(C):
                ptrue = peak[te, j]
                acc[mname]['psse'][j] += np.sum((P_hat[:, j] - ptrue) ** 2)
                acc[mname]['psst'][j] += np.sum((ptrue - ptrue.mean()) ** 2)
                acc[mname]['pabs'][j] += np.sum(np.abs(P_hat[:, j] - ptrue))
            acc[mname]['pn'] += len(te)
            acc[mname]['tn'] += len(te) * T

            # 逐工况 CSV 的 relL2 改为相对真实时程本身，不再借训练均值时程作分母。
            for k, ci in enumerate(te):
                rows.append(dict(
                    model=mname, fold=fi + 1, case=ids[ci],
                    **{
                        f'relL2_{chan[j]}':
                        np.linalg.norm(Yh[k, :, j] - Yt[k, :, j]) /
                        (np.linalg.norm(Yt[k, :, j]) + 1e-30)
                        for j in range(C)
                    }
                ))

    def eng(v, u):
        return f"{v/1e3:,.2f} " + ('kN' if u == 'N' else 'kN·m')

    def table(title, k_sse, k_sst, k_abs, k_n):
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
        print(f"  {'模型':<15}" + ''.join(f"{c + ' R²':>11}" for c in chan))
        for m in mnames:
            v = 1 - acc[m][k_sse] / np.maximum(acc[m][k_sst], 1e-30)
            print(f"  {m:<15}" + ''.join(f"{x:>11.4f}" for x in v))
        print(f"\n  {'模型':<15}" + ''.join(f"{c + ' 误差':>15}" for c in chan))
        for m in mnames:
            e = (acc[m][k_abs] / max(acc[m][k_n], 1) if k_abs
                 else np.sqrt(acc[m][k_sse] / max(acc[m][k_n], 1)))
            print(f"  {m:<15}" + ''.join(f"{eng(e[j], unit[j]):>15}" for j in range(C)))

    # ------------------------------------------------------------------
    # 时程头
    # ------------------------------------------------------------------
    table(
        "时程分支：载荷时程（测试集 R² + RMSE；POD 与尺度在训练折内拟合）",
        'sse', 'sst', None, 'tn'
    )

    print("\n  ★ 联合归一化载荷空间的严格误差分解")
    print("     定义：分母 = ||F_test - μ_train||²；因此 ④² = ①² + ③²（仅联合空间成立）")
    tr_rel = np.sqrt(joint_trunc / max(joint_signal, 1e-30))
    print(f"     ① POD截断(r_L={rl}) = {tr_rel:>7.2%}   （所有回归器共同下界）")
    for m in mnames:
        reg_rel = np.sqrt(acc[m]['joint_reg'] / max(joint_signal, 1e-30))
        tot_rel = np.sqrt(acc[m]['joint_total'] / max(joint_signal, 1e-30))
        closure = ((acc[m]['joint_total'] - (joint_trunc + acc[m]['joint_reg'])) /
                   max(acc[m]['joint_total'], 1e-30))
        print(f"     {m:<15} ③ 回归={reg_rel:>7.2%}   ④ 合计={tot_rel:>7.2%}"
              f"   闭合残差={closure:+.2e}")

    # ------------------------------------------------------------------
    # 峰值头
    # ------------------------------------------------------------------
    table(
        "★ 峰值头：载荷峰值（标准测试集 R² + MAE）",
        'psse', 'psst', 'pabs', 'pn'
    )

    eval_idx = np.array(sorted(set(eval_indices)), dtype=int)
    mag = np.array([np.mean(np.abs(peak[eval_idx, j])) for j in range(C)])
    print("\n  评价集参考量级 " +
          '  '.join(f"{chan[j]}={eng(mag[j], unit[j])}" for j in range(C)))

    if is_holdout:
        pick_map = locked_peak
        pick_label = '训练集内 inner-CV 预先锁定；未看 holdout'
        title = '★ 结论行（峰值头：训练集内选模型后再评价 holdout）'
    else:
        pick_map = {j: min(mnames, key=lambda m: acc[m]['psse'][j]) for j in range(C)}
        pick_label = '同一外层CV上的探索性最优（用于模型比较，不是独立确认）'
        title = '★ 结论行（峰值头：同一五折上的探索性最优）'

    print(f"\n{'=' * 80}\n  {title}")
    print(f"  模型选择口径：{pick_label}")
    print("  R²口径：测试数据自身均值基线（standard test-set R²）")
    for j in range(C):
        bm = pick_map[j]
        r2 = safe_r2(acc[bm]['psse'][j], acc[bm]['psst'][j])
        mae = acc[bm]['pabs'][j] / max(acc[bm]['pn'], 1)
        print(f"     {chan[j]:<4} R²={r2:>7.4f}  MAE={eng(mae, unit[j]):>12}"
              f"  = 评价集量级的 {mae/max(mag[j], 1e-30):>6.1%}   ({bm})")

    if 'Mz' in chan:
        j = chan.index('Mz')
        bm = pick_map[j]
        r2 = safe_r2(acc[bm]['psse'][j], acc[bm]['psst'][j])
        mae = acc[bm]['pabs'][j] / max(acc[bm]['pn'], 1)
        mref = args.mref * 1e3
        print(f"\n  ★ RSI_O 峰值头（= |Mz| / Mref, Mref={args.mref} kN·m）")
        print(f"     RSI_O 评价集量级 = {mag[j]/mref:.5f}    预测 R² = {r2:.4f}"
              f"    MAE = {mae/mref:.5f}   ({bm})")
        if mag[j] / mref < 0.01:
            print(f"     ⚠ 该工况 RSI_O 仅 {mag[j]/mref:.5f} 量级，接近数值噪声；"
                  "组内排序不宜作物理规律。")
    print('=' * 80)

    # ------------------------------------------------------------------
    # summary CSV
    # ------------------------------------------------------------------
    if args.summary_append:
        split = (f"holdout:{args.holdout_col}={args.holdout_value}"
                 if is_holdout else f"{args.kfold}fold")
        head = [
            '工况', '划分', '通道', '量级', '单位', 'R2', 'MAE', 'MAE占量级',
            '报告模型', '模型选择口径', 'R2口径', 'r_L', '通道集'
        ]
        parent = _os.path.dirname(_os.path.abspath(args.summary_append))
        if parent:
            _os.makedirs(parent, exist_ok=True)
        new = not _os.path.exists(args.summary_append)
        with open(args.summary_append, 'a', newline='', encoding='utf-8-sig') as fo:
            w = _csv.writer(fo)
            if new:
                w.writerow(head)
            for j in range(C):
                bm = pick_map[j]
                r2 = safe_r2(acc[bm]['psse'][j], acc[bm]['psst'][j])
                mae = acc[bm]['pabs'][j] / max(acc[bm]['pn'], 1)
                uu = 'kN' if unit[j] == 'N' else 'kN·m'
                w.writerow([
                    args.type, split, chan[j], f"{mag[j]/1e3:.3f}", uu,
                    f"{r2:.4f}", f"{mae/1e3:.3f}",
                    f"{mae/max(mag[j], 1e-30):.4f}", bm, pick_label,
                    'test-mean standard R2', rl, '+'.join(chan)
                ])
            if 'Mz' in chan:
                j = chan.index('Mz')
                bm = pick_map[j]
                r2 = safe_r2(acc[bm]['psse'][j], acc[bm]['psst'][j])
                mae = acc[bm]['pabs'][j] / max(acc[bm]['pn'], 1)
                mref = args.mref * 1e3
                w.writerow([
                    args.type, split, 'RSI_O', f"{mag[j]/mref:.5f}", '-',
                    f"{r2:.4f}", f"{mae/mref:.5f}",
                    f"{mae/max(mag[j], 1e-30):.4f}", bm, pick_label,
                    'test-mean standard R2', rl, '+'.join(chan)
                ])
        print(f"  v3 定稿数字已追加 → {args.summary_append}")

    if args.csv and rows:
        with open(args.csv, 'w', newline='', encoding='utf-8-sig') as fo:
            w = _csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  逐工况结果 → {args.csv}")


if __name__ == '__main__':
    main()
