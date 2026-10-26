#!/usr/bin/env python3
"""
test_gram_math.py — 使用合成数据检查 Gram 投影与误差分解恒等式。

固定随机种子生成测试矩阵，无需外部数据。
用法：python test_gram_math.py
"""
import numpy as np

rng = np.random.default_rng(0)

# ---- 生成测试场: M 个网格点, n_cases 个工况 × T 个时刻 ----
M, n_cases, T, r = 300, 9, 12, 5
N = n_cases * T
X = rng.normal(size=(M, N)) @ np.diag(1 + rng.random(N))       # (M, N)
X += rng.normal(size=(M, 4)) @ rng.normal(size=(4, N)) * 3     # 叠加低秩结构
V = 0.1 + rng.random(M)                                        # 非均匀单元体积

# ---- 加权 Gram，与 gram_pod.py 的计算形式一致----
G = (X * V[:, None]).T @ X                                     # G[i,j] = <x_i, x_j>_V

cols = lambda cs: np.concatenate([np.arange(c*T, (c+1)*T) for c in cs])
tr_c, te_c = [0, 1, 2, 3, 4, 5], [6, 7, 8]
tau, eps = cols(tr_c), cols(te_c)


def pod_from_gram(G, tau, eps, r):
    n = len(tau)
    g = G[:, tau].sum(axis=1) / n
    gbar = float(G[np.ix_(tau, tau)].sum()) / (n * n)
    K_tt = G[np.ix_(tau, tau)] - g[tau][:, None] - g[tau][None, :] + gbar
    K_tt = 0.5 * (K_tt + K_tt.T)
    lam, Vec = np.linalg.eigh(K_tt)
    o = np.argsort(lam)[::-1]
    lam, Vec = np.clip(lam[o], 0, None), Vec[:, o]
    rr = int(min(r, np.sum(lam > 1e-12 * lam[0])))
    sig = np.sqrt(lam[:rr])
    A_tr = sig[:, None] * Vec[:, :rr].T
    K_te = G[np.ix_(tau, eps)] - g[tau][:, None] - g[eps][None, :] + gbar
    A_te = (Vec[:, :rr] / sig).T @ K_te
    Kdiag = np.diag(G)[eps] - 2 * g[eps] + gbar
    return A_tr, A_te, Kdiag, lam, Vec, sig, rr


A_tr, A_te, Kdiag, lam, Vec, sig, rr = pod_from_gram(G, tau, eps, r)

# ---- 直接法: 真的把场搬出来做一遍 ----
m = X[:, tau].mean(axis=1)
Xc = X[:, tau] - m[:, None]
Sw = np.sqrt(V)[:, None] * Xc
U = (Sw @ Vec[:, :rr]) / sig[None, :] / np.sqrt(V)[:, None]     # 物理模态

ok = True
def chk(name, err, tol=1e-8):
    global ok
    good = err < tol
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {name:<42} 误差={err:.3e}")

# 1) 模态在体积内积下正交归一
chk("U^T diag(V) U = I", np.abs(U.T @ (V[:, None] * U) - np.eye(rr)).max())

# 2) 训练系数一致
chk("A_tr == U^T diag(V) (X_tau - m)",
    np.abs(A_tr - U.T @ (V[:, None] * Xc)).max() / np.abs(A_tr).max())

# 3) 测试系数一致性：测试数据投影到训练基
A_te_direct = U.T @ (V[:, None] * (X[:, eps] - m[:, None]))
chk("A_te == U^T diag(V) (X_eps - m)",
    np.abs(A_te - A_te_direct).max() / np.abs(A_te_direct).max())

# 4) Kdiag == ||x_j - m||²_V
Kd_direct = np.einsum('mj,m,mj->j', X[:, eps] - m[:, None], V, X[:, eps] - m[:, None])
chk("K_jj == ||x_j - m||²_V", np.abs(Kdiag - Kd_direct).max() / Kd_direct.max())

# 5) 误差分解恒等式（核心）
a_hat = A_te + rng.normal(scale=0.3, size=A_te.shape)           # 假装是回归预测
recon = m[:, None] + U @ a_hat
err_direct = np.einsum('mj,m,mj->j', recon - X[:, eps], V, recon - X[:, eps])
err_gram = Kdiag - 2 * np.sum(a_hat * A_te, axis=0) + np.sum(a_hat ** 2, axis=0)
chk("||重构-真场||²_V == K_jj - 2â·a + ||â||²",
    np.abs(err_direct - err_gram).max() / err_direct.max())

# 6) 正交可加: 总误差 = 截断 + 回归
trunc = Kdiag - np.sum(A_te ** 2, axis=0)
reg = np.sum((a_hat - A_te) ** 2, axis=0)
chk("总误差 == 截断 + 回归", np.abs(err_gram - (trunc + reg)).max() / err_gram.max())

# 7) 截断项非负且等于直接算的残差
proj = m[:, None] + U @ A_te
tr_direct = np.einsum('mj,m,mj->j', proj - X[:, eps], V, proj - X[:, eps])
chk("截断项 == ||投影-真场||²_V", np.abs(trunc - tr_direct).max() / tr_direct.max())

print("\n" + ("全部数学一致性检查通过" if ok else "存在检查失败项"))
