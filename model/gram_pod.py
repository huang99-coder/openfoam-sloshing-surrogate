#!/usr/bin/env python3
"""
gram_pod.py — P1 第一步: 把某类工况的场"过一遍", 只产出一个 Gram 矩阵 G

为什么要这么做
--------------
折内建基（每折只用训练工况重新建 POD 基）听起来要把 6~17 GB 的场重跑 5 遍。
其实不用。快照法 POD 只依赖快照两两之间的内积, 也就是 Gram 矩阵
    G[i,j] = <x_i, x_j>_V
G 是 N×N 的（N = 工况数 × 时间步数, 通常几千）, 算一次只要几百 MB。
之后每一折的"去均值 + 建基 + 投影测试工况"全部是对 G 做切片, 不再碰原始场。

数学（kfold_infold.py 里用到, 这里先记下来）
--------------------------------------------
设训练列集合 tau, n=|tau|, 训练均值 m = (1/n) Σ_{k∈tau} x_k, 则对任意 i,j：
    g_i   = x_i^T m   = (1/n) Σ_{k∈tau} G[i,k]
    gbar  = m^T m     = (1/n²) Σ_{k,l∈tau} G[k,l]
    K[i,j] = <x_i - m, x_j - m> = G[i,j] - g_i - g_j + gbar
训练块 K[tau,tau] 特征分解 -> V, Lambda; sigma = sqrt(Lambda)
    训练系数 A_tr = sigma * V^T
    测试系数 A_te = (V/sigma)^T @ K[tau, eps]     ← 测试数据投影到训练折基
    测试快照到训练均值的距离平方 = K[j,j]

用法
----
  python gram_pod.py --input-dir all_h5_data --type slope      --workdir gram
  python gram_pod.py --input-dir all_h5_data --type overtaking --workdir gram
  # 内存/磁盘紧张时（stride=2 磁盘和内存都减半, 只影响数值精度不影响结论）
  python gram_pod.py --input-dir all_h5_data --type turning --workdir gram --stride 2
  # 体积加权（推荐, 与 build_shared_pod_v3 用同一份 cellV.npy）
  python gram_pod.py --input-dir all_h5_data --type slope --workdir gram --cell-volumes cellV.npy

产物
----
  gram/gram_<type>.npz     G, case_files, T, n_cases, 权重标记
  gram/stage_<type>.dat    暂存的场 memmap (M×N float32)。算完 G 后可删,
                           想在折内算空间模态或做场级可视化时才需要留着。
"""
import os
import glob
import argparse
import numpy as np
import h5py

from merge_pod_h5 import detect_case_type
from build_shared_pod import get_field_info, load_field_matrix


def collect(input_dir, ct, stride):
    files = sorted(glob.glob(os.path.join(input_dir, '**/*.h5'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, '*.h5')))
    files = [fp for fp in files if detect_case_type(fp) == ct]
    if not files:
        raise SystemExit(f"❌ {input_dir} 下没有 {ct} 类型的 H5")

    specs, M_ref, src_ref = [], None, None
    for fp in files:
        with h5py.File(fp, 'r') as f:
            info = get_field_info(f)
        Mc = (info['M'] + stride - 1) // stride if stride > 1 else info['M']
        if M_ref is None:
            M_ref, src_ref = Mc, info['source']
        elif Mc != M_ref:
            print(f"  ⚠ 跳过 {os.path.basename(fp)} (M={Mc:,} ≠ {M_ref:,})")
            continue
        specs.append((fp, info['n_ts']))
    return specs, M_ref, src_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='all_h5_data')
    ap.add_argument('--type', required=True,
                    choices=['slope', 'braking', 'turning', 'overtaking'])
    ap.add_argument('--workdir', default='gram')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--cell-volumes', default=None)
    ap.add_argument('--chunk', type=int, default=20000,
                    help='算 G 时每次读入的行(网格点)数; 内存紧就调小')
    ap.add_argument('--keep-stage', action='store_true',
                    help='算完 G 后保留暂存 memmap(默认保留, 加此参数无副作用)')
    ap.add_argument('--drop-stage', action='store_true',
                    help='算完 G 后删除暂存 memmap, 省磁盘')
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    ct = args.type

    specs, M, src = collect(args.input_dir, ct, args.stride)
    Ts = [t for _, t in specs]
    T = min(Ts)
    n_cases = len(specs)
    N = n_cases * T
    if len(set(Ts)) > 1:
        print(f"  ⚠ 各工况时间步不一致 {sorted(set(Ts))}, 统一截断到 T={T}"
              f"（与 kfold_eval 行为一致）")

    print(f"[{ct}] 工况 {n_cases}, 每工况 T={T}, 快照总数 N={N}, 网格点 M={M:,}, 取场={src}")
    print(f"     暂存文件 {M*N*4/1e9:.1f} GB   Gram 矩阵 {N*N*8/1e9:.2f} GB")

    w = None
    if args.cell_volumes:
        V = np.load(args.cell_volumes).astype(np.float64).ravel()
        if args.stride > 1:
            V = V[::args.stride]
        if V.shape[0] != M:
            raise SystemExit(f"单元体积长度 {V.shape[0]} ≠ 网格点数 {M}")
        w = V   # G 里直接用 V 做权, 不用开根号
        print(f"     ✓ 体积加权   V: max/min = {V.max()/V.min():.1f}")

    stage_path = os.path.join(args.workdir, f'stage_{ct}.dat')

    # ---------- 阶段 1: 暂存 ----------
    if os.path.exists(stage_path) and os.path.getsize(stage_path) == M * N * 4:
        print(f"  ✓ 复用已有暂存 {stage_path}")
        X = np.memmap(stage_path, dtype='float32', mode='r', shape=(M, N))
    else:
        print(f"  写暂存 {stage_path} ...")
        X = np.memmap(stage_path, dtype='float32', mode='w+', shape=(M, N))
        for c, (fp, _) in enumerate(specs):
            with h5py.File(fp, 'r') as f:
                arr = load_field_matrix(f, stride=args.stride)
            X[:, c*T:(c+1)*T] = arr[:, :T]
            del arr
            print(f"    + [{c+1}/{n_cases}] {os.path.basename(fp)}")
        X.flush()

    # ---------- 阶段 2: Gram ----------
    print(f"  算 Gram (行分块 {args.chunk}) ...")
    G = np.zeros((N, N), dtype=np.float64)
    n_blk = (M + args.chunk - 1) // args.chunk
    for bi, a in enumerate(range(0, M, args.chunk)):
        b = min(a + args.chunk, M)
        B = np.asarray(X[a:b, :], dtype=np.float64)
        if w is not None:
            G += (B * w[a:b, None]).T @ B
        else:
            G += B.T @ B
        del B
        if (bi + 1) % 5 == 0 or bi + 1 == n_blk:
            print(f"    块 {bi+1}/{n_blk}")

    out = os.path.join(args.workdir, f'gram_{ct}.npz')
    np.savez_compressed(
        out,
        G=G,
        case_files=np.array([os.path.basename(fp) for fp, _ in specs]),
        T=T, n_cases=n_cases, M=M, stride=args.stride,
        field_source=src, volume_weighted=bool(w is not None),
    )
    print(f"  ✅ {out}   ({os.path.getsize(out)/1e6:.1f} MB)")

    del X
    if args.drop_stage:
        os.remove(stage_path)
        print(f"  已删除暂存 {stage_path}")
    else:
        print(f"  暂存保留在 {stage_path}（折内算空间模态/画场图时要用；不用了就删）")


if __name__ == '__main__':
    main()
