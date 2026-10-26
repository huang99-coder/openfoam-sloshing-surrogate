#!/usr/bin/env python3
"""
POD-NN 预测新工况（统一四类工况版）

支持: 制动(braking) / 转弯(turning) / 超车(overtaking) / 上下坡(slope)

用法:
  # 上下坡
  python predict_podnn.py --model results/podnn_model.pth slope --angle 7.5 --fill 55 --direction up

  # 制动
  python predict_podnn.py --model results/podnn_model.pth braking --decel 0.3 --fill 40 --liquid oil

  # 转弯
  python predict_podnn.py --model results/podnn_model.pth turning --radius 50 --speed 30 --fill 40

  # 超车
  python predict_podnn.py --model results/podnn_model.pth overtaking --lane-time 1.5 --fill 50

  # 输出到指定文件
  python predict_podnn.py --model results/podnn_model.pth braking --decel 0.5 --fill 60 --output my_pred.png
"""

import math
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pod_fourier_model import ParamPredictorMLP

TANK_LENGTH = 7.5
G = 9.81
LANE_WIDTH = 3.5

# 必须与 merge_pod_h5.py 中的 FEATURE_NAMES 顺序一致
FEATURE_NAMES = [
    'angle_deg',             # [0]  坡度角
    'fill_level',            # [1]  充装率 [0,1]
    'density',               # [2]  液体密度
    'viscosity',             # [3]  运动粘度
    'sin_angle',             # [4]  sin(坡度角)
    'cos_angle',             # [5]  cos(坡度角)
    'height_diff',           # [6]  高度差
    'direction',             # [7]  坡道方向
    'deceleration',          # [8]  制动减速度
    'lateral_acceleration',  # [9]  横向加速度
    'lane_change_time',      # [10] 变道时间
    'case_type',             # [11] 工况类型编码
]

CASE_TYPE_MAP = {'braking': 0, 'turning': 1, 'overtaking': 2, 'slope': 3}

# 常见液体物性
LIQUID_DB = {
    'water':    {'rho': 1000.0, 'nu': 1e-6},
    'oil':      {'rho': 850.0,  'nu': 3.5e-6},
    'diesel':   {'rho': 850.0,  'nu': 3.5e-6},
    'gasoline': {'rho': 720.0,  'nu': 7.4e-7},
}


# ──────────────────────────────────────────────────────────
# Fourier 重构
# ──────────────────────────────────────────────────────────

def fourier_basis(t, n_harmonics):
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    return np.column_stack(cols)


def reconstruct(theta_flat, T, K, n_harmonics):
    t_norm = np.linspace(0.0, 1.0, T)
    Phi = fourier_basis(t_norm, n_harmonics)
    n_th = 1 + 2 * n_harmonics
    y_rec = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        lo = k * n_th
        hi = (k + 1) * n_th
        y_rec[:, k] = Phi @ theta_flat[lo:hi]
    return y_rec


# ──────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────

def load_model(path):
    if path.endswith('.pkl'):
        # KRR 模型 (train_krr.py 产出)
        import joblib
        ckpt = joblib.load(path)
        ckpt['backend'] = 'krr'
        return ckpt['model'], ckpt
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['model_config']
    model = ParamPredictorMLP(
        cfg['n_input'], cfg['n_output'],
        tuple(cfg['hidden_dims']), dropout=0.0)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


# ──────────────────────────────────────────────────────────
# 特征构建（四类工况统一）
# ──────────────────────────────────────────────────────────

def build_feature_row(case_type, fill_pct, rho, nu, **kwargs):
    """
    构建与 merge_pod_h5.py FEATURE_NAMES 对齐的12维特征向量。

    参数:
        case_type:  'slope' / 'braking' / 'turning' / 'overtaking'
        fill_pct:   充装率 (%)
        rho:        液体密度 (kg/m³)
        nu:         运动粘度 (m²/s)

    工况特有参数 (kwargs):
        slope:      angle_deg, direction ('up'/'down')
        braking:    decel_g (减速度, 以g为单位)
        turning:    radius (m), speed_kmh (km/h)
        overtaking: lane_time (s)
    """
    feat = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    # 通用参数
    feat[1] = fill_pct / 100.0     # fill_level
    feat[2] = rho                   # density
    feat[3] = nu                    # viscosity
    feat[5] = 1.0                   # cos_angle (默认=1, 非坡道)
    feat[11] = float(CASE_TYPE_MAP.get(case_type, 3))  # case_type

    if case_type == 'slope':
        angle = kwargs.get('angle_deg', 0)
        direction = kwargs.get('direction', 'up')
        ar = np.radians(abs(angle))
        sign = 1.0 if direction == 'up' else -1.0

        feat[0] = abs(angle)                        # angle_deg
        feat[4] = np.sin(ar)                        # sin_angle
        feat[5] = np.cos(ar)                        # cos_angle
        feat[6] = TANK_LENGTH * np.sin(ar) * sign   # height_diff
        feat[7] = 0.0 if direction == 'up' else 1.0 # direction

    elif case_type == 'braking':
        decel_g = kwargs.get('decel_g', 0.3)
        feat[8] = decel_g * G                       # deceleration (m/s²)

    elif case_type == 'turning':
        radius = kwargs.get('radius', 100)
        speed_kmh = kwargs.get('speed_kmh', 30)
        v_ms = speed_kmh / 3.6
        feat[9] = v_ms ** 2 / radius                # lateral_acceleration

    elif case_type == 'overtaking':
        lane_time = kwargs.get('lane_time', 3.0)
        feat[9] = (LANE_WIDTH / 2) * (math.pi / lane_time) ** 2  # peak lateral_accel
        feat[10] = lane_time                         # lane_change_time

    return feat


# ──────────────────────────────────────────────────────────
# 预测
# ──────────────────────────────────────────────────────────

def predict(model, ckpt, feature_row):
    """用训练好的模型预测 POD 时间系数

    feature_row:
      旧版模型 → 12维 ndarray, 按 ckpt['param_idx'] 索引;
      rich 模型 (ckpt 带 input_cols) → dict {列名: 值}, 按 input_cols 顺序拼。
    """
    if ckpt.get('backend') == 'krr':
        row = np.array([[float(feature_row[c]) for c in ckpt['input_cols']]],
                       dtype=np.float64)
        row_sc = ckpt['scaler_X'].transform(row)
        theta = ckpt['scaler_T'].inverse_transform(model.predict(row_sc))
        return reconstruct(theta[0], ckpt['n_timesteps'],
                           ckpt['n_modes'], ckpt['n_harmonics'])
    if ckpt.get('input_cols'):
        row = np.array([[float(feature_row[c]) for c in ckpt['input_cols']]],
                       dtype=np.float64)
    else:
        row = feature_row[ckpt['param_idx']].reshape(1, -1)
    row_sc = (row - ckpt['scaler_X_mean']) / ckpt['scaler_X_scale']
    with torch.no_grad():
        theta_sc = model(torch.FloatTensor(row_sc)).numpy()
    theta = theta_sc * ckpt['scaler_T_scale'] + ckpt['scaler_T_mean']
    return reconstruct(
        theta[0], ckpt['n_timesteps'], ckpt['n_modes'], ckpt['n_harmonics'])


# ──────────────────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────────────────

def plot_prediction(y, title_str, output_path):
    """绘制 POD 时间系数预测结果"""
    T, K = y.shape
    t = np.arange(T)
    n_show = min(6, K)

    n_rows = (n_show + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i in range(n_show):
        ax = axes_flat[i]
        ax.plot(t, y[:, i], 'b-', lw=1.5)
        ax.set(xlabel='Time step', ylabel=f'$a_{{{i+1}}}(t)$')
        ax.set_title(f'POD Mode {i+1}', fontsize=11)
        ax.grid(alpha=0.3)

    for i in range(n_show, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle(title_str, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='POD-NN 预测新工况（统一四类工况版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict_podnn.py --model results/podnn_model.pth slope --angle 7.5 --fill 55 --direction up
  python predict_podnn.py --model results/podnn_model.pth braking --decel 0.3 --fill 40 --liquid oil
  python predict_podnn.py --model results/podnn_model.pth turning --radius 50 --speed 30 --fill 40
  python predict_podnn.py --model results/podnn_model.pth overtaking --lane-time 1.5 --fill 50
""")

    # 全局参数
    parser.add_argument('--model', default='results/podnn_model.pth',
                        help='模型文件路径')
    parser.add_argument('--output', default=None,
                        help='输出图片路径 (默认自动命名)')

    # 子命令：四类工况
    sub = parser.add_subparsers(dest='case_type', help='工况类型')
    sub.required = True

    # ── slope ──
    p_slope = sub.add_parser('slope', help='上下坡工况')
    p_slope.add_argument('--angle', type=float, required=True, help='坡度角 (°)')
    p_slope.add_argument('--fill', type=float, required=True, help='充装率 (%%)')
    p_slope.add_argument('--direction', default='up', choices=['up', 'down'])
    p_slope.add_argument('--liquid', default='water',
                         choices=list(LIQUID_DB.keys()), help='液体类型')

    # ── braking ──
    p_brake = sub.add_parser('braking', help='紧急制动工况')
    p_brake.add_argument('--decel', type=float, required=True,
                         help='减速度 (g), 例如 0.3 表示 0.3g')
    p_brake.add_argument('--fill', type=float, required=True, help='充装率 (%%)')
    p_brake.add_argument('--liquid', default='water',
                         choices=list(LIQUID_DB.keys()))

    # ── turning ──
    p_turn = sub.add_parser('turning', help='稳态转弯工况')
    p_turn.add_argument('--radius', type=float, required=True,
                        help='转弯半径 (m)')
    p_turn.add_argument('--speed', type=float, required=True,
                        help='车速 (km/h)')
    p_turn.add_argument('--fill', type=float, required=True, help='充装率 (%%)')
    p_turn.add_argument('--liquid', default='water',
                        choices=list(LIQUID_DB.keys()))

    # ── overtaking ──
    p_ot = sub.add_parser('overtaking', help='超车变道工况')
    p_ot.add_argument('--lane-time', type=float, default=None,
                      help='变道时间 (s) — 旧版12维模型用')
    p_ot.add_argument('--peak-gx', type=float, default=None,
                      help='峰值横向加速度 peak_gx — rich模型用, 单位与训练数据一致')
    p_ot.add_argument('--total-time', type=float, default=None,
                      help='机动总时长 total_time (s) — rich模型用')
    p_ot.add_argument('--t-change', type=float, default=None,
                      help='变道时间 t_change (s) — rich模型用')
    p_ot.add_argument('--double', action='store_true',
                      help='双超车 (不加则为单超车) — rich模型用')
    p_ot.add_argument('--fill', type=float, required=True, help='充装率 (%%)')
    p_ot.add_argument('--liquid', default='water',
                      choices=list(LIQUID_DB.keys()))

    args = parser.parse_args()

    # 加载模型
    model, ckpt = load_model(args.model)
    print(f"Model: n_harmonics={ckpt['n_harmonics']}  "
          f"n_modes={ckpt['n_modes']}  n_ts={ckpt['n_timesteps']}")

    # 液体物性
    liq = LIQUID_DB.get(args.liquid, LIQUID_DB['water'])
    rho, nu = liq['rho'], liq['nu']

    # 构建特征向量
    ct = args.case_type
    if ct == 'slope':
        feat = build_feature_row(ct, args.fill, rho, nu,
                                 angle_deg=args.angle,
                                 direction=args.direction)
        title = (f"POD-NN | {args.direction} {args.angle}°  "
                 f"fill={args.fill}%  {args.liquid}")
        default_out = (f"pred_slope_{args.direction}{args.angle}"
                       f"_fill{int(args.fill)}.png")

    elif ct == 'braking':
        feat = build_feature_row(ct, args.fill, rho, nu,
                                 decel_g=args.decel)
        title = (f"POD-NN | braking {args.decel}g  "
                 f"fill={args.fill}%  {args.liquid}")
        default_out = (f"pred_brake_{args.decel}g"
                       f"_fill{int(args.fill)}.png")

    elif ct == 'turning':
        v_ms = args.speed / 3.6
        a_lat = v_ms ** 2 / args.radius
        feat = build_feature_row(ct, args.fill, rho, nu,
                                 radius=args.radius,
                                 speed_kmh=args.speed)
        title = (f"POD-NN | R={args.radius}m v={args.speed}km/h "
                 f"(a={a_lat:.2f}m/s²)  fill={args.fill}%  {args.liquid}")
        default_out = (f"pred_turn_R{int(args.radius)}m_{int(args.speed)}km"
                       f"_fill{int(args.fill)}.png")

    elif ct == 'overtaking':
        if ckpt.get('input_cols'):
            # rich 模型: 直接提供真实物理量, 按 input_cols 顺序拼输入
            vals = {
                'fill_level': args.fill / 100.0,
                'peak_gx': args.peak_gx,
                'total_time': args.total_time,
                't_change': args.t_change,
                'is_double_overtake': 1.0 if args.double else 0.0,
            }
            missing = [c for c in ckpt['input_cols'] if vals.get(c) is None]
            if missing:
                parser.error(f"rich 模型缺少输入: {missing} "
                             f"(用 --peak-gx / --total-time / --t-change 提供)")
            feat = vals
            tag = 'double' if args.double else 'single'
            title = (f"POD-NN | overtaking {tag}  gx={args.peak_gx} "
                     f"T={args.total_time}s tc={args.t_change}s  "
                     f"fill={args.fill}%  {args.liquid}")
            default_out = (f"pred_overtake_{tag}_T{int(args.total_time)}s"
                           f"_fill{int(args.fill)}.png")
        else:
            if args.lane_time is None:
                parser.error("旧版12维模型需要 --lane-time")
            a_peak = (LANE_WIDTH / 2) * (math.pi / args.lane_time) ** 2
            feat = build_feature_row(ct, args.fill, rho, nu,
                                     lane_time=args.lane_time)
            title = (f"POD-NN | lane change {args.lane_time}s "
                     f"(a_peak={a_peak:.2f}m/s²)  fill={args.fill}%  "
                     f"{args.liquid}")
            default_out = (f"pred_overtake_t{args.lane_time}s"
                           f"_fill{int(args.fill)}.png")

    output_path = args.output or default_out

    # 打印特征向量
    print(f"\nCase: {ct}")
    if isinstance(feat, dict):
        print(f"Feature vector ({len(feat)}D, rich):")
        for name, val in feat.items():
            print(f"  {name:25s} = {val:.6g}")
    else:
        print(f"Feature vector ({len(feat)}D):")
        for i, (name, val) in enumerate(zip(FEATURE_NAMES, feat)):
            if val != 0:
                print(f"  [{i:2d}] {name:25s} = {val:.6g}")

    # 预测
    y = predict(model, ckpt, feat)
    print(f"\nPrediction: {y.shape[0]} timesteps × {y.shape[1]} modes")

    # 绘图
    plot_prediction(y, title, output_path)


if __name__ == '__main__':
    main()