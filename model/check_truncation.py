#!/usr/bin/env python3
"""
check_truncation.py — 把"能量占比"翻译成"场重构误差"

大白话
------
POD 的截断误差有一个精确关系（不需要跑任何场）：
    场相对 L2 误差 = sqrt(1 - 前 r 阶累计能量占比)
所以 85% 的能量占比 ≠ 15% 的误差, 而是 sqrt(0.15) = 38.7% 的误差。
这一步先把这个数字算出来, 心里有底再往下做。

前提: shared_*.h5 必须是 build_shared_pod_v3.py 产出的(带 eigenvalues_full)。
      老文件只有截断后的 sigma, 分母是错的, 会打印警告。

用法
----
  python check_truncation.py data/shared_slope.h5
  python check_truncation.py data/shared_*.h5          # 四类一起对比
"""
import sys
import numpy as np
import h5py


def report(path):
    with h5py.File(path, 'r') as f:
        meta = dict(f['metadata'].attrs)
        if 'eigenvalues_full' in f:
            ev = f['eigenvalues_full'][()]
            full = True
        else:
            sv = f['singular_values'][()]
            ev = np.asarray(sv, float) ** 2
            full = False
        r_saved = int(meta.get('n_modes', len(ev)))
        src = meta.get('field_source', '?')
        vw = bool(meta.get('volume_weighted', False))

    tot = ev.sum()
    print(f"\n{'='*74}")
    print(f"{path}")
    print(f"  取场={src}   体积加权={'是' if vw else '否'}   保存模态数 r={r_saved}   "
          f"谱长度={len(ev)}")
    if not full:
        print("  ⚠ 无 eigenvalues_full: 下面的分母是【保留的 r 阶】而不是全谱,")
        print("    累计能量被系统性高估, 误差被系统性低估。请用 v3 重建后再读。")
    print(f"{'-'*74}")
    print(f"  {'r':>4} {'本阶能量':>10} {'累计能量':>10} {'场相对L2误差':>14}")
    cum = np.cumsum(ev) / tot
    for r in [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 80]:
        if r <= len(ev):
            print(f"  {r:>4} {ev[r-1]/tot:>10.4f} {cum[r-1]:>10.4f} "
                  f"{np.sqrt(max(1-cum[r-1], 0)):>13.1%}")
    for target in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, target) + 1)
        if k <= len(ev):
            print(f"  → 达到 {target:.0%} 能量需要 r = {k}  "
                  f"(对应场误差 {np.sqrt(1-target):.1%})")
        else:
            print(f"  → 保存的谱内达不到 {target:.0%} 能量")
    print(f"{'='*74}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        report(p)
    print("\n说明：系数回归指标与 POD 截断误差分别报告。")
    print("      场重构精度同时取决于模态截断与系数回归；")
    print("      载荷代理使用低维特征时，按相应物理输出评价预测结果。")
