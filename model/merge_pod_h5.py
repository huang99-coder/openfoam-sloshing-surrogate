#!/usr/bin/env python3
"""
合并全部108组工况的H5文件，构建统一POD-NN训练数据集

V2 更新（统一模型版）：
  - 支持4类工况：制动(BR)、转弯(TR)、超车(OT)、上下坡(SL)
  - 输入特征从6维扩展为12维，新增：
    * deceleration:        制动减速度 (m/s²)，非制动工况=0
    * lateral_acceleration: 横向加速度 (m/s²)，非转弯/超车工况=0
    * lane_change_time:    变道时间 (s)，非超车工况=0
    * case_type:           工况类型编码 0=制动 1=转弯 2=超车 3=坡道
  - 自动从文件名和H5内部参数识别工况类型
  - 兼容旧版上下坡专用格式

用法:
    python merge_pod_h5.py --input-dir all_h5_data --output data/all_108_merged.h5
    python merge_pod_h5.py --input-dir all_h5_data --output data/all_108_merged.h5 --n-modes 10
    python merge_pod_h5.py --input-dir all_h5_data --explore-only
"""

import h5py
import numpy as np
import os
import re
import glob
import json


# ──────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────
TANK_LENGTH = 7.5
G = 9.81

# 统一模型的12维输入特征
FEATURE_NAMES = [
    'angle_deg',             # 坡度角(°)，坡道专用，其他=0
    'fill_level',            # 充装率 [0,1]
    'density',               # 液体密度 kg/m³
    'viscosity',             # 运动粘度 m²/s
    'sin_angle',             # sin(坡度角)，坡道专用，其他=0
    'cos_angle',             # cos(坡度角)，坡道=cos(θ)，其他=1
    'height_diff',           # 高度差(m)=L*sin(θ)，上坡>0下坡<0，其他=0
    'direction',             # 坡道方向 0=up 1=down，其他=0
    'deceleration',          # 制动减速度(m/s²)，制动专用，其他=0
    'lateral_acceleration',  # 横向加速度(m/s²)，转弯/超车专用，其他=0
    'lane_change_time',      # 变道时间(s)，超车专用，其他=0
    'case_type',             # 0=braking 1=turning 2=overtaking 3=slope
]

CASE_TYPE_MAP = {'braking': 0, 'turning': 1, 'overtaking': 2, 'slope': 3}


# ──────────────────────────────────────────────────────────
# 工况类型识别
# ──────────────────────────────────────────────────────────
def detect_case_type(fpath, h5_meta=None):
    """从文件名和H5元数据自动识别工况类型"""
    base = os.path.basename(fpath).lower()

    # 优先从H5 metadata读取
    if h5_meta and 'case_type' in h5_meta:
        ct = h5_meta['case_type']
        if ct in CASE_TYPE_MAP:
            return ct

    # 从文件名推断
    if any(k in base for k in ['brake', 'braking', '_br', 'br-', 'decel']):
        return 'braking'
    elif any(k in base for k in ['turn', 'turning', '_tr', 'tr-', 'r50', 'r100', 'r150']):
        return 'turning'
    elif any(k in base for k in ['overtake', 'overtaking', '_ot', 'ot-', 'lane', 'double']):
        return 'overtaking'
    elif any(k in base for k in ['slope', 'up', 'down', 'angle', '_sl', 'sl-']):
        return 'slope'

    print(f"    ⚠ 无法识别工况类型: {base}，默认slope")
    return 'slope'


# ──────────────────────────────────────────────────────────
# 文件名解析（扩展版，支持4类工况）
# ──────────────────────────────────────────────────────────
def parse_filename(fpath, case_type):
    """从文件名解析工况参数，返回统一的参数字典"""
    base = os.path.basename(fpath).lower()

    params = {
        'angle_deg': 0.0,
        'fill_level': 0.5,
        'density': 1000.0,
        'viscosity': 1e-6,
        'sin_angle': 0.0,
        'cos_angle': 1.0,
        'height_diff': 0.0,
        'direction': 0.0,
        'deceleration': 0.0,
        'lateral_acceleration': 0.0,
        'lane_change_time': 0.0,
        'case_type': float(CASE_TYPE_MAP.get(case_type, 3)),
    }

    # ── 充装率（所有工况通用）──
    m = re.search(r'fill(\d+)', base)
    if m:
        v = float(m.group(1))
        params['fill_level'] = v / 100.0 if v > 1 else v
    else:
        m = re.search(r'(\d+)p', base)
        if m:
            params['fill_level'] = float(m.group(1)) / 100.0

    # ── 液体类型 ──
    if 'oil' in base or 'diesel' in base:
        params['density'] = 850.0
        params['viscosity'] = 3.5e-6

    # ── 工况特有参数 ──
    if case_type == 'slope':
        # 坡度角
        m = re.search(r'angle(\d+)', base)
        if m:
            params['angle_deg'] = float(m.group(1))
        else:
            m = re.search(r'(\d+)deg', base)
            if m:
                params['angle_deg'] = float(m.group(1))

        # 方向
        if 'down' in base:
            params['direction'] = 1.0
        angle_rad = np.radians(params['angle_deg'])
        params['sin_angle'] = np.sin(angle_rad)
        params['cos_angle'] = np.cos(angle_rad)
        sign = -1.0 if params['direction'] > 0.5 else 1.0
        params['height_diff'] = TANK_LENGTH * params['sin_angle'] * sign

    elif case_type == 'braking':
        # 减速度
        m = re.search(r'(\d+\.?\d*)g', base)
        if m:
            v = float(m.group(1))
            if v >= 2.0:  # 文件名形如 03g/05g, 实际表示 0.3g/0.5g
                v /= 10.0
            params['deceleration'] = v * G
        else:
            m = re.search(r'decel(\d+\.?\d*)', base)
            if m:
                params['deceleration'] = float(m.group(1))

    elif case_type == 'turning':
        # 转弯半径和车速 → 横向加速度
        m_r = re.search(r'r(\d+)', base) or re.search(r't(\d+)m', base)
        m_v = re.search(r'(\d+)km', base)
        if m_r and m_v:
            R = float(m_r.group(1))
            v_kmh = float(m_v.group(1))
            v_ms = v_kmh / 3.6
            params['lateral_acceleration'] = v_ms ** 2 / R

    elif case_type == 'overtaking':
        # 变道时间
        m = re.search(r't(\d+\.?\d*)s', base)
        if m:
            params['lane_change_time'] = float(m.group(1))
        # 横向加速度峰值 = (3.5/2) * (π/t_lane)²
        if params['lane_change_time'] > 0:
            import math
            t = params['lane_change_time']
            params['lateral_acceleration'] = (3.5 / 2) * (math.pi / t) ** 2

    return params


# ──────────────────────────────────────────────────────────
# H5 工具函数
# ──────────────────────────────────────────────────────────
def explore_h5_structure(fpath):
    """打印H5文件结构"""
    print(f"\n{'='*60}")
    print(f"探索: {os.path.basename(fpath)}")
    print(f"{'='*60}")
    with h5py.File(fpath, 'r') as f:
        def _print(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 {name}: {obj.shape}  {obj.dtype}")
            else:
                print(f"{indent}📁 {name}/")
        f.visititems(_print)


def get_n_modes_from_file(fpath):
    """读取模态数"""
    try:
        with h5py.File(fpath, 'r') as f:
            return f['pod/alpha_water/time_coefficients'].shape[0]
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# 单文件读取
# ──────────────────────────────────────────────────────────
def read_single_h5(fpath, n_modes_target):
    """
    读取单个H5文件，返回 (X, y, param_dict)
    X: (n_timesteps, 12)   统一12维输入特征
    y: (n_timesteps, n_modes_target)  POD时间系数
    """
    # 读取H5元数据确定工况类型
    h5_meta = {}
    with h5py.File(fpath, 'r') as f:
        if 'metadata' in f:
            for k, v in f['metadata'].attrs.items():
                h5_meta[k] = v
        if 'parameters' in f:
            for k in f['parameters'].keys():
                try:
                    h5_meta[k] = float(f['parameters'][k][()])
                except:
                    pass
            # 读取字符串属性
            for k, v in f['parameters'].attrs.items():
                h5_meta[k] = v

    case_type = detect_case_type(fpath, h5_meta)
    params = parse_filename(fpath, case_type)

    # [MERGE-FIX-C] H5 字段名 alias:不同 preparer 历史上用过不同字段名,
    # 这里统一映射到 FEATURE_NAMES 里的标准名
    H5_KEY_ALIASES = {
        't_change':       'lane_change_time',     # overtaking 短名 → 标准全名
        'decel_ms2':      'deceleration',          # braking 内部名(若被写出)
        'lateral_accel':  'lateral_acceleration',  # turning 内部名(若被写出)
        'fill_pct_value': 'fill_pct',              # 占位,以防遗留
    }
    for old_key, new_key in H5_KEY_ALIASES.items():
        if old_key in h5_meta and new_key not in h5_meta:
            h5_meta[new_key] = h5_meta[old_key]

    # 用H5内部参数覆盖文件名解析（更准确）
    # [MERGE-FIX-A] 扩展覆盖范围:工况特有参数也优先用 H5 写入的值,
    # 避免文件名正则 (\d+\.?\d*)g 在 'brake07g' 上把 0.7g 误读成 7g 这类问题
    for key in ['density', 'viscosity', 'deceleration',
                'lateral_acceleration', 'lane_change_time',
                'fill_level']:
        if key in h5_meta:
            try:
                params[key] = float(h5_meta[key])
            except (TypeError, ValueError):
                pass
    # H5 里的 fill_level 是 0-1 浮点;旧版 fill_pct (0-100) 兜底
    if 'fill_pct' in h5_meta and 'fill_level' not in h5_meta:
        try:
            params['fill_level'] = float(h5_meta['fill_pct']) / 100.0
        except (TypeError, ValueError):
            pass

    with h5py.File(fpath, 'r') as f:
        # POD时间系数
        tc = f['pod/alpha_water/time_coefficients'][()]  # (n_modes, n_ts)
        y_full = tc.T  # (n_ts, n_modes)
        n_ts, n_m = y_full.shape
        y = y_full[:, :n_modes_target]

        # 时间轴
        try:
            times = f['snapshots/alpha_water/times'][()]
            t0, t1 = times.min(), times.max()
            t_norm = (times - t0) / (t1 - t0) if t1 > t0 else np.linspace(0, 1, n_ts)
        except Exception:
            t_norm = np.linspace(0, 1, n_ts)

    # 构建12维输入矩阵
    # 注意: 'time'不在FEATURE_NAMES中了（Case-to-Fourier不用时间作输入）
    # 但为了兼容merge后train_podnn.py的读取方式，保持与原版一致的格式
    param_row = [params[k] for k in FEATURE_NAMES]
    X = np.column_stack([
        np.tile(param_row, (n_ts, 1)),  # (n_ts, 12)
        t_norm.reshape(-1, 1),           # (n_ts, 1) 时间列
    ])  # → (n_ts, 13)

    param_dict = {
        'file': os.path.basename(fpath),
        **params,                          # params 里 case_type 是 float (0/1/2/3)
        'case_type': case_type,            # [MERGE-FIX-B] 用字符串覆盖 float,
                                           # 这样末尾"工况类型分布"统计才不会全 0
        'case_type_code': float(CASE_TYPE_MAP.get(case_type, 3)),  # 训练用编码也保留一份
    }

    return X, y, param_dict


# ──────────────────────────────────────────────────────────
# 主合并函数
# ──────────────────────────────────────────────────────────
def merge_all_h5_files(h5_dir, output_path='data/all_108_merged.h5',
                       n_modes_fixed=None):
    """合并目录中所有工况类型的H5文件"""

    # 查找文件
    h5_files = sorted(glob.glob(os.path.join(h5_dir, '**/*.h5'), recursive=True))
    if not h5_files:
        h5_files = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
    if not h5_files:
        print(f"❌ 在 {h5_dir} 中未找到H5文件")
        return None

    print(f"找到 {len(h5_files)} 个H5文件")

    # ── Pass 1: 扫描 ──
    print("\n[Pass 1] 扫描...")
    modes_list = []
    valid_files = []
    type_counts = {'braking': 0, 'turning': 0, 'overtaking': 0, 'slope': 0}

    for fpath in h5_files:
        n = get_n_modes_from_file(fpath)
        if n is not None:
            modes_list.append(n)
            valid_files.append(fpath)
            ct = detect_case_type(fpath)
            type_counts[ct] = type_counts.get(ct, 0) + 1
            print(f"  {os.path.basename(fpath):50s}  {n:2d}modes  [{ct}]")
        else:
            print(f"  ❌ 无法读取: {os.path.basename(fpath)}")

    if not modes_list:
        print("❌ 没有有效文件")
        return None

    print(f"\n工况分布: {type_counts}")

    if n_modes_fixed is not None:
        n_modes_target = n_modes_fixed
        valid_files = [f for f, n in zip(valid_files, modes_list) if n >= n_modes_target]
        modes_list = [n for n in modes_list if n >= n_modes_target]
        print(f"指定模态数: {n_modes_target}，有效文件: {len(valid_files)}")
    else:
        n_modes_target = min(modes_list)
        print(f"模态数: min={min(modes_list)} max={max(modes_list)} → 截取前{n_modes_target}个")

    # ── Pass 2: 合并 ──
    print(f"\n[Pass 2] 读取并合并 {len(valid_files)} 个文件...")

    all_X, all_y, case_info, case_boundaries = [], [], [], []
    total_samples = 0

    for i, fpath in enumerate(valid_files):
        fname = os.path.basename(fpath)
        try:
            X, y, pd_ = read_single_h5(fpath, n_modes_target)

            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print(f"    ⚠ {fname}: 含NaN，跳过")
                continue
            if np.any(np.isinf(X)) or np.any(np.isinf(y)):
                print(f"    ⚠ {fname}: 含Inf，跳过")
                continue

            print(f"  [{i+1}/{len(valid_files)}] {fname:50s} X={X.shape} y={y.shape} [{pd_['case_type']}]")

            case_boundaries.append((total_samples, total_samples + X.shape[0]))
            total_samples += X.shape[0]
            all_X.append(X)
            all_y.append(y)
            case_info.append(pd_)

        except Exception as e:
            print(f"    ❌ {fname}: {e}")

    if not all_X:
        print("\n❌ 没有成功读取任何数据")
        return None

    X_merged = np.vstack(all_X)
    y_merged = np.vstack(all_y)

    # 完整特征名（12个工况参数 + 1个时间）
    full_feature_names = FEATURE_NAMES + ['time']

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"合并完成")
    print(f"  总工况数:  {len(all_X)}")
    print(f"  总样本数:  {X_merged.shape[0]}")
    print(f"  输入维度:  {X_merged.shape[1]}  ({len(FEATURE_NAMES)} params + 1 time)")
    print(f"  输出维度:  {y_merged.shape[1]} modes")
    print(f"\n  输入特征范围:")
    for j, name in enumerate(full_feature_names):
        if j < X_merged.shape[1]:
            col = X_merged[:, j]
            uniq = len(np.unique(np.round(col, 6)))
            print(f"    [{j:2d}] {name:<25s}  [{col.min():+10.4f}, {col.max():+10.4f}]  uniq={uniq}")

    print(f"\n  工况类型分布:")
    for ct in ['braking', 'turning', 'overtaking', 'slope']:
        n = sum(1 for d in case_info if d['case_type'] == ct)
        if n > 0:
            print(f"    {ct:12s}: {n} 组")
    print(f"{'='*60}")

    # ── 保存 ──
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('X', data=X_merged, compression='gzip')
        f.create_dataset('y', data=y_merged, compression='gzip')

        meta = f.create_group('metadata')
        meta.attrs['n_cases'] = len(all_X)
        meta.attrs['n_samples'] = X_merged.shape[0]
        meta.attrs['n_features'] = X_merged.shape[1]
        meta.attrs['n_modes'] = n_modes_target
        meta.attrs['case_type'] = 'unified_108'
        meta.attrs['feature_names'] = json.dumps(full_feature_names)

        f.create_dataset('case_boundaries', data=np.array(case_boundaries))

        grp = f.create_group('case_info')
        for i, info in enumerate(case_info):
            cg = grp.create_group(f'case_{i:03d}')
            for k, v in info.items():
                try:
                    cg.attrs[k] = v if isinstance(v, str) else float(v)
                except:
                    pass

    fsize = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n✅ 已保存: {output_path}  ({fsize:.2f} MB)")
    return output_path


# ──────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='合并全部工况H5文件（统一模型版）')
    parser.add_argument('--input-dir', type=str, default='all_h5_data',
                        help='H5文件目录（支持子目录递归搜索）')
    parser.add_argument('--output', type=str, default='data/all_108_merged.h5',
                        help='输出路径')
    parser.add_argument('--n-modes', type=int, default=None,
                        help='指定统一模态数')
    parser.add_argument('--explore-only', action='store_true',
                        help='仅探索文件结构')
    args = parser.parse_args()

    if args.explore_only:
        files = sorted(glob.glob(os.path.join(args.input_dir, '**/*.h5'), recursive=True))
        for fp in files[:5]:
            explore_h5_structure(fp)
    else:
        merge_all_h5_files(args.input_dir, args.output, args.n_modes)
