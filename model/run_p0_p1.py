#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_p0_p1.py — 共享 POD、Gram 矩阵及折内评价的运行入口。

使用 Python 统一组织各阶段命令；run_p0_p1.ps1 提供 PowerShell 包装入口。

用法
----
  python run_p0_p1.py --dry-run                    # 显示阶段命令，不执行计算
  python run_p0_p1.py                              # 运行四类工况
  python run_p0_p1.py --types slope braking        # 指定工况类型
  python run_p0_p1.py --cell-volumes cellV.npy     # 体积加权
  python run_p0_p1.py --stride 2                   # 空间采样步长
  python run_p0_p1.py --skip-rebuild               # 复用共享基，运行 Gram 与评价
  python run_p0_p1.py --only gram cv               # 指定阶段
  python run_p0_p1.py --fast                       # FCNN 使用 400 轮训练

阶段名：verify / build / augment / truncation / gram / cv
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# 四类的既定超参，来自《POD-NN 运行手册》第 2~5 节，已锁定
CFG = {
    'slope':      dict(nmodes=20, H=6,  model='fcnn', desc='data/shared_slope.h5'),
    'braking':    dict(nmodes=20, H=10, model='fcnn', desc='data/shared_braking.h5'),
    'turning':    dict(nmodes=20, H=10, model='fcnn', desc='data/shared_turning.h5'),
    'overtaking': dict(nmodes=40, H=12, model='krr',  desc='data/shared_overtaking_rich.h5'),
}
ALL_STAGES = ['verify', 'build', 'augment', 'truncation', 'gram', 'cv']


def banner(msg):
    print(f"\n{'=' * 74}\n{msg}\n{'=' * 74}", flush=True)


def run(cmd, dry, keep_going=False):
    printable = ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    if dry:
        return True
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=HERE)
    dt = time.time() - t0
    if rc != 0:
        print(f"\n  ✗ 退出码 {rc}（用时 {dt:.0f}s）", flush=True)
        if not keep_going:
            print("  已中止。修好上面的错误再跑，或加 --keep-going 跳过继续。")
            sys.exit(rc)
        return False
    print(f"  ✓ 完成（{dt:.0f}s）", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument('--types', nargs='+', default=list(CFG.keys()), choices=list(CFG.keys()))
    ap.add_argument('--cell-volumes', default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--input-dir', default='all_h5_data')
    ap.add_argument('--skip-rebuild', action='store_true', help='跳过 build + augment')
    ap.add_argument('--only', nargs='+', default=None, choices=ALL_STAGES)
    ap.add_argument('--fast', action='store_true', help='FCNN 用 400 轮快跑')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--keep-going', action='store_true')
    args = ap.parse_args()

    stages = args.only if args.only else list(ALL_STAGES)
    if args.skip_rebuild:
        stages = [s for s in stages if s not in ('build', 'augment')]

    cv = ['--cell-volumes', args.cell_volumes] if args.cell_volumes else []
    st = ['--stride', str(args.stride)] if args.stride > 1 else []
    dry = args.dry_run

    print(f"包目录 : {HERE}")
    print(f"解释器 : {PY}")
    print(f"工况   : {', '.join(args.types)}")
    print(f"阶段   : {' → '.join(stages)}")
    print(f"体积权 : {args.cell_volumes or '未启用体积加权'}")
    print(f"stride : {args.stride}")
    if dry:
        print("\n*** --dry-run：只打印命令，不执行 ***")

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)

    # ---------- 0 自检 ----------
    if 'verify' in stages:
        banner("0 / 环境与数学自检")
        run([PY, 'verify_env.py'], dry, args.keep_going)
        run([PY, 'test_gram_math.py'], dry, args.keep_going)

    # ---------- 1 P0 建基 ----------
    if 'build' in stages:
        banner("1 / P0 —— 重建共享基（多存 spatial_modes 与 eigenvalues_full）")
        for t in args.types:
            run([PY, 'build_shared_pod_v3.py', '--input-dir', args.input_dir,
                 '--types', t, '--n-modes', str(CFG[t]['nmodes'])] + st + cv,
                dry, args.keep_going)

    if 'augment' in stages and 'overtaking' in args.types:
        banner("1b / 变道专属 —— 注入真实物理输入（rich 文件）")
        run([PY, 'augment_inputs.py', '--shared-dir', 'data',
             '--input-dir', args.input_dir, '--types', 'overtaking'],
            dry, args.keep_going)

    # ---------- 2 截断误差阶梯 ----------
    if 'truncation' in stages:
        banner("2 / P0 —— 截断误差阶梯（★ 先看这张表再往下）")
        shared = [f'data/shared_{t}.h5' for t in args.types]
        exist = [p for p in shared if dry or os.path.exists(os.path.join(HERE, p))]
        if exist:
            run([PY, 'check_truncation.py'] + exist, dry, args.keep_going)
        else:
            print("  跳过：data/ 下还没有 shared_*.h5")

    # ---------- 3 Gram ----------
    if 'gram' in stages:
        banner("3 / P1 —— 算 Gram（最慢的一步，每类一次；之后不再碰原始场）")
        for t in args.types:
            run([PY, 'gram_pod.py', '--input-dir', args.input_dir,
                 '--type', t, '--workdir', 'gram'] + st + cv,
                dry, args.keep_going)

    # ---------- 4 折内建基 CV + 全局基对照 ----------
    if 'cv' in stages:
        banner("4 / P1 —— 折内建基 CV + 全局固定基对照")
        for t in args.types:
            c = CFG[t]
            base = [PY, 'kfold_infold.py',
                    '--gram', f'gram/gram_{t}.npz', '--desc', c['desc'],
                    '--model', c['model'], '--n-modes-use', '6',
                    '--n-harmonics', str(c['H'])]
            if c['model'] == 'fcnn':
                base += ['--hidden', '64', '48', '--dropout', '0.15']
                if args.fast:
                    base += ['--epochs', '400', '--patience', '60']
            print(f"\n--- {t} : 训练折内建基 ---")
            run(base + ['--basis', 'infold', '--csv', f'results/infold_{t}.csv'],
                dry, args.keep_going)
            print(f"\n--- {t} : 全局基（对照） ---")
            run(base + ['--basis', 'global'], dry, args.keep_going)

    banner("全部完成")
    print("逐工况结果  : results/infold_*.csv")
    print("两次 CV 的差 : 就是「表示层信息进入」带来的乐观偏差，做成一张表放进论文 5.4 节")
    print("\n下一步 P4 整档留出，例如：")
    print(f"  {os.path.basename(PY)} kfold_infold.py --gram gram/gram_slope.npz "
          f"--desc data/shared_slope.h5 --model fcnn --n-harmonics 6 "
          f"--hidden 64 48 --dropout 0.15 --holdout-col fill_level --holdout-value 70")


if __name__ == '__main__':
    main()
