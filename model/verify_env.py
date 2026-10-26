#!/usr/bin/env python3
"""
verify_env.py — 依赖、模块导入、文件名解析与数据目录检查。

检查内容：
  1. numpy、h5py、scikit-learn、torch、matplotlib 和 joblib。
  2. 本地模型与数据处理模块的导入。
  3. brake03g 的减速度解析结果（0.3g）。
  4. HDF5、共享基、Gram 和训练输出目录。

用法：python verify_env.py
"""
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ok = True


def line(tag, good, msg):
    global ok
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {tag:<34} {msg}")


print("=" * 74)
print("1. 外部依赖")
print("=" * 74)
for pkg, minv in [('numpy', None), ('h5py', None), ('sklearn', None),
                  ('torch', None), ('matplotlib', None), ('joblib', None)]:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', '?')
        line(pkg, True, f"v{v}")
    except Exception as e:
        line(pkg, False, f"缺失 → pip install {'scikit-learn' if pkg=='sklearn' else pkg}  ({e})")

print()
print("=" * 74)
print("2. 本地模块导入")
print("=" * 74)
MODULES = [
    'merge_pod_h5', 'pod_fourier_model', 'build_shared_pod', 'train_podnn',
    'kfold_eval', 'kfold_krr', 'train_krr', 'augment_inputs', 'predict_podnn',
    'check_split_energy', 'check_mode_ceiling',
    'build_shared_pod_v3', 'gram_pod', 'kfold_infold',
]
for mod in MODULES:
    try:
        importlib.import_module(mod)
        line(mod, True, "ok")
    except Exception as e:
        line(mod, False, f"{type(e).__name__}: {e}")

print()
print("=" * 74)
print("3. 减速度解析检查")
print("=" * 74)
src = open(os.path.join(HERE, 'merge_pod_h5.py'), encoding='utf-8', errors='replace').read()
has_patch = bool(re.search(r'if\s+v\s*>=\s*2\.0', src)) and 'v /= 10.0' in src
line('减速度解析规则', has_patch,
     "brake03g → 0.3g" if has_patch else "未检测到 0.3g 解析规则；请检查 merge_pod_h5.py 中的减速度处理。")

try:
    from merge_pod_h5 import parse_filename, detect_case_type
    p = parse_filename('pod_data_brake03g_fill50_water.h5', 'braking')
    d = float(p.get('deceleration', -1))
    line('实测解析 brake03g', 2.5 < d < 3.5, f"deceleration = {d:.3f} m/s²（期望 ≈2.943）")
except Exception as e:
    line('实测解析 brake03g', False, f"{type(e).__name__}: {e}")

print()
print("=" * 74)
print("4. 数据目录")
print("=" * 74)
for d, what in [('all_h5_data', '原始 CFD H5（每个算例一个）'),
                ('data', 'shared_*.h5 共享 POD 产物'),
                ('gram', 'gram_*.npz / stage_*.dat'),
                ('results', '训练产物 podnn_model.pth / krr_model.pkl')]:
    p = os.path.join(HERE, d)
    n = len([x for x in os.listdir(p) if not x.startswith('.')]) if os.path.isdir(p) else -1
    if n < 0:
        line(d + '/', False, "目录不存在")
    elif n == 0:
        print(f"  [ -- ] {d+'/':<34} 空（{what}）")
    else:
        line(d + '/', True, f"{n} 个文件")

print()
print("=" * 74)
if ok:
    print("环境检查通过。数学一致性检查：python test_gram_math.py")
else:
    print("环境检查存在失败项，详见以上输出。")
print("=" * 74)
