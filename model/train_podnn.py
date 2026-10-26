#!/usr/bin/env python3
"""
POD-NN 训练 (Case-to-Fourier) - 统一108组模型版

V2更新：
  - CASE_PARAM_COLS 从6维扩展为12维，覆盖4类工况参数
  - 其余训练逻辑、网络结构、Fourier参数化方法完全不变

用法：
  # 使用旧版上下坡数据（自动兼容，多余的列全为0会被scaler处理）
  python train_podnn.py --data data/slope_pod_merged.h5

  # 使用新版108组统一数据
  python train_podnn.py --data data/all_108_merged.h5

  # 推荐参数
  python train_podnn.py --data data/all_108_merged.h5 --n-harmonics 6 --n-modes-use 8 --hidden 128 128 64
"""

import os
import json
import argparse
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pod_fourier_model import create_model

TANK_LENGTH = 7.5

# V2: 扩展为12维工况参数（不含时间列）
# 对于旧版6维数据，新增的列全为0，StandardScaler会自动处理（方差为0的列不影响训练）
CASE_PARAM_COLS = [
    'angle_deg',             # 坡度角
    'fill_level',            # 充装率
    'density',               # 液体密度
    'viscosity',             # 运动粘度
    'sin_angle',             # sin(坡度角)
    'cos_angle',             # cos(坡度角)
    'height_diff',           # 高度差
    'direction',             # 坡道方向
    'deceleration',          # 制动减速度        ← 新增
    'lateral_acceleration',  # 横向加速度        ← 新增
    'lane_change_time',      # 变道时间          ← 新增
    'case_type',             # 工况类型编码      ← 新增
]


def fourier_basis(t, n_harmonics):
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    return np.column_stack(cols)


def fit_fourier_coeffs(y_seq, n_harmonics):
    T, K = y_seq.shape
    t_norm = np.linspace(0.0, 1.0, T)
    Phi = fourier_basis(t_norm, n_harmonics)
    theta_list = []
    for k in range(K):
        theta, *_ = np.linalg.lstsq(Phi, y_seq[:, k], rcond=None)
        theta_list.append(theta)
    return np.concatenate(theta_list, axis=0)


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


class FourierCaseTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.scaler_X = StandardScaler()
        self.scaler_T = StandardScaler()
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_r2': [], 'val_r2': [], 'lr': []
        }

    def load_data(self, h5_path):
        print(f"\n加载数据: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            X_all = f['X'][()]
            y_all = f['y'][()]
            boundaries = f['case_boundaries'][()]
            meta = f['metadata']
            feat_names = json.loads(meta.attrs['feature_names'])
            n_cases = int(meta.attrs['n_cases'])
            n_modes = int(meta.attrs['n_modes'])
            # rich 文件: 自带真实物理输入列清单 (augment_inputs.py 写入)
            input_cols = None
            if 'input_cols' in meta.attrs:
                input_cols = json.loads(meta.attrs['input_cols'])

        # [SANITY] 各 case 的 n_timesteps 可能不一致 (如某个 case 只有 99 步),
        # 统一截断到最短 T, 与打补丁后的 kfold_eval.py 行为一致
        case_lengths = [int(b[1] - b[0]) for b in boundaries]
        unique_lengths = sorted(set(case_lengths))
        n_timesteps = min(case_lengths)
        if len(unique_lengths) > 1:
            from collections import Counter
            print(f"  ⚠ 各 case 时间步不一致 {dict(Counter(case_lengths))}, "
                  f"统一截断到 T={n_timesteps}")
        else:
            print(f"  ✓ 所有 {len(case_lengths)} 个 case 都是 {n_timesteps} 时间步")

        # V2: 匹配可用的列（兼容旧版和新版数据）
        # rich 文件用它自带的 input_cols(真实物理输入), 否则走 CASE_PARAM_COLS 白名单
        sel_cols = input_cols if input_cols else CASE_PARAM_COLS
        param_idx = []
        param_names = []
        for col_name in sel_cols:
            if col_name in feat_names:
                param_idx.append(feat_names.index(col_name))
                param_names.append(col_name)

        if not param_idx:
            raise ValueError(f"无法匹配任何参数列！\n  CASE_PARAM_COLS: {CASE_PARAM_COLS}\n  H5 features: {feat_names}")

        n_modes_use = min(self.config.get('n_modes_use') or n_modes, n_modes)

        X_cases, y_cases = [], []
        for c in range(n_cases):
            s, e = boundaries[c]
            X_cases.append(X_all[s, param_idx])
            y_cases.append(y_all[s:s + n_timesteps, :n_modes_use])

        X_cases = np.asarray(X_cases, dtype=np.float64)
        y_cases = np.asarray(y_cases, dtype=np.float64)

        print(f"  工况数: {n_cases}   每工况时间步: {n_timesteps}")
        print(f"  匹配到的输入列 ({len(param_idx)}): {param_names}")
        print(f"  工况参数矩阵: {X_cases.shape}")
        print(f"  时序矩阵:     {y_cases.shape}")

        # 检查是否有常量列（如旧版数据中新增列全为0）
        for j, name in enumerate(param_names):
            col = X_cases[:, j]
            if col.std() < 1e-10:
                print(f"    ⚠ 列 '{name}' 方差≈0（所有值={col[0]:.4f}），对训练无贡献")

        self.X_cases = X_cases
        self.y_cases = y_cases
        self.n_cases = n_cases
        self.n_timesteps = n_timesteps
        self.n_modes = n_modes_use
        self.param_idx = param_idx
        self.param_names = param_names
        self.feat_names = feat_names
        self.input_cols = input_cols          # rich 文件才有, 否则 None
        return X_cases, y_cases

    def build_targets(self, y_cases):
        n_harmonics = self.config['n_harmonics']
        thetas = np.array([
            fit_fourier_coeffs(seq, n_harmonics) for seq in y_cases
        ], dtype=np.float64)
        self.n_theta_per_mode = 1 + 2 * n_harmonics
        self.n_output = thetas.shape[1]

        y_recon = np.array([
            reconstruct(theta, self.n_timesteps, self.n_modes, n_harmonics)
            for theta in thetas
        ])
        ss_res = np.sum((y_cases - y_recon) ** 2)
        ss_tot = np.sum((y_cases - y_cases.mean()) ** 2)
        r2_fit = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        print(f"\nFourier 参数化:")
        print(f"  n_harmonics: {n_harmonics}")
        print(f"  每模态参数数: {self.n_theta_per_mode}")
        print(f"  总输出维度:   {self.n_output}")
        print(f"  纯 Fourier 拟合 R²: {r2_fit:.4f}")

        self.theta_cases = thetas
        return thetas

    def prepare_data(self, X_cases, theta_cases, n_test=3, n_val=3):
        n = len(X_cases)
        # V2: 对108组数据，增加测试/验证集大小
        if n > 50:
            n_test = max(n_test, int(n * 0.15))
            n_val = max(n_val, int(n * 0.15))
            print(f"  (108组模式: 自动调整 n_test={n_test}, n_val={n_val})")

        rng = np.random.default_rng(self.config.get('seed', 42))
        idx = rng.permutation(n)
        tr_idx = idx[:n - n_test - n_val]
        va_idx = idx[n - n_test - n_val:n - n_test]
        te_idx = idx[n - n_test:]

        X_train, X_val, X_test = X_cases[tr_idx], X_cases[va_idx], X_cases[te_idx]
        T_train, T_val, T_test = theta_cases[tr_idx], theta_cases[va_idx], theta_cases[te_idx]

        print(f"\n数据集划分:")
        print(f"  训练: {len(tr_idx)} 工况")
        print(f"  验证: {len(va_idx)} 工况")
        print(f"  测试: {len(te_idx)} 工况")

        X_tr_sc = self.scaler_X.fit_transform(X_train)
        X_va_sc = self.scaler_X.transform(X_val)
        X_te_sc = self.scaler_X.transform(X_test)

        T_tr_sc = self.scaler_T.fit_transform(T_train)
        T_va_sc = self.scaler_T.transform(T_val)
        T_te_sc = self.scaler_T.transform(T_test)

        bs = min(self.config.get('batch_size', 8), len(tr_idx))
        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_sc), torch.FloatTensor(T_tr_sc)),
            batch_size=bs, shuffle=True)
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_va_sc), torch.FloatTensor(T_va_sc)),
            batch_size=bs, shuffle=False)

        self.X_test_sc = torch.FloatTensor(X_te_sc)
        self.T_test_sc = torch.FloatTensor(T_te_sc)
        self.theta_test_raw = T_test
        self.y_test_raw = self.y_cases[te_idx]
        self.te_idx = te_idx
        self.tr_idx = tr_idx
        self.va_idx = va_idx

    def build_model(self):
        hidden_dims = tuple(self.config.get('hidden', [128, 128, 64]))
        dropout = float(self.config.get('dropout', 0.25))
        self.model = create_model(
            n_input=self.X_cases.shape[1],
            n_output=self.n_output,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ratio = len(self.tr_idx) / n_params

        print(f"\n模型 (Case-to-Fourier)")
        print(f"  输入: {self.X_cases.shape[1]} 维工况参数")
        print(f"  输出: {self.n_output} 维 Fourier 参数")
        print(f"  隐层: {hidden_dims}  dropout={dropout}")
        print(f"  参数量: {n_params:,}")
        print(f"  训练工况/参数比: {ratio:.4f}")

    def _predict_theta_scaled(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                yp = self.model(xb.to(self.device)).cpu().numpy()
                preds.append(yp)
                trues.append(yb.numpy())
        return np.vstack(preds), np.vstack(trues)

    def _theta_to_sequence(self, theta_scaled):
        theta = self.scaler_T.inverse_transform(theta_scaled)
        return np.array([
            reconstruct(row, self.n_timesteps, self.n_modes, self.config['n_harmonics'])
            for row in theta
        ])

    def _r2_seq(self, loader):
        pred_sc, true_sc = self._predict_theta_scaled(loader)
        y_pred = self._theta_to_sequence(pred_sc)
        y_true = self._theta_to_sequence(true_sc)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    def train(self):
        epochs = int(self.config.get('epochs', 1500))
        lr = float(self.config.get('lr', 1e-3))
        patience = int(self.config.get('patience', 80))
        weight_decay = float(self.config.get('weight_decay', 3e-2))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(50, min(epochs // 3, 200)), T_mult=2, eta_min=1e-6)
        criterion = nn.MSELoss()

        best_val = float('inf')
        best_ep = 0
        no_imp = 0
        best_path = self.config.get('best_path', 'best_podnn.pth')

        print(f"\n训练: epochs={epochs}  lr={lr}  patience={patience}  wd={weight_decay}")
        print(f"{'=' * 72}")

        for ep in range(epochs):
            self.model.train()
            tr_losses = []
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_losses.append(loss.item())
            scheduler.step()

            self.model.eval()
            va_losses = []
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    va_losses.append(criterion(self.model(xb), yb).item())

            tr_loss = float(np.mean(tr_losses))
            va_loss = float(np.mean(va_losses))
            tr_r2 = self._r2_seq(self.train_loader)
            va_r2 = self._r2_seq(self.val_loader)
            cur_lr = optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(va_loss)
            self.history['train_r2'].append(tr_r2)
            self.history['val_r2'].append(va_r2)
            self.history['lr'].append(cur_lr)

            if va_loss < best_val:
                best_val = va_loss
                best_ep = ep
                no_imp = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                no_imp += 1

            if (ep + 1) % 50 == 0 or ep == 0:
                print(
                    f"Epoch {ep + 1:5d}/{epochs} | "
                    f"Loss {tr_loss:.5f}/{va_loss:.5f} | "
                    f"R² {tr_r2:.4f}/{va_r2:.4f} | "
                    f"LR {cur_lr:.1e}"
                )

            if no_imp >= patience:
                print(f"\n早停: 连续 {patience} epochs 无改善")
                break

        self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        print(f"\n最佳模型: Epoch {best_ep + 1}  Val Loss: {best_val:.6f}")
        return self.history

    def evaluate(self):
        print(f"\n{'=' * 72}\n测试集评估（{len(self.te_idx)} 个工况）\n{'=' * 72}")
        self.model.eval()
        with torch.no_grad():
            theta_pred_sc = self.model(self.X_test_sc.to(self.device)).cpu().numpy()

        theta_pred = self.scaler_T.inverse_transform(theta_pred_sc)
        y_pred = np.array([
            reconstruct(row, self.n_timesteps, self.n_modes, self.config['n_harmonics'])
            for row in theta_pred
        ])
        y_true = self.y_test_raw

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2_total = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        r2m, rmse, mae = [], [], []
        for k in range(self.n_modes):
            yt = y_true[:, :, k].reshape(-1)
            yp = y_pred[:, :, k].reshape(-1)
            sr = np.sum((yt - yp) ** 2)
            st = np.sum((yt - yt.mean()) ** 2)
            r2m.append(1.0 - sr / st if st > 1e-12 else 0.0)
            rmse.append(float(np.sqrt(np.mean((yt - yp) ** 2))))
            mae.append(float(np.mean(np.abs(yt - yp))))

        print(f"\n总体 R²: {r2_total:.4f}")
        print(f"\n{'模态':>5} {'R2':>10} {'RMSE':>12} {'MAE':>12}")
        print('-' * 44)
        for i, r2 in enumerate(r2m):
            flag = 'OK' if r2 > 0.9 else ('~' if r2 > 0.7 else 'X')
            print(f"  a_{i+1:<2} {r2:>10.4f} {rmse[i]:>12.6f} {mae[i]:>12.6f}  {flag}")

        print("\n逐工况总体 R²:")
        r2_case = []
        for ci, cid in enumerate(self.te_idx):
            yt = y_true[ci].reshape(-1)
            yp = y_pred[ci].reshape(-1)
            sr = np.sum((yt - yp) ** 2)
            st = np.sum((yt - yt.mean()) ** 2)
            r2 = 1.0 - sr / st if st > 1e-12 else 0.0
            r2_case.append(float(r2))
            print(f"  Case {cid:2d}: R²={r2:.4f}")

        self.eval_results = {
            'r2_total': float(r2_total),
            'r2_per_mode': [float(x) for x in r2m],
            'rmse_per_mode': rmse,
            'mae_per_mode': mae,
            'r2_per_case': r2_case,
            'test_case_ids': self.te_idx.tolist(),
            'n_modes': self.n_modes,
            'n_harmonics': int(self.config['n_harmonics']),
        }
        self.y_pred_test = y_pred
        return self.eval_results

    def plot_results(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        plt.rcParams.update({'font.size': 12, 'figure.dpi': 200})

        # Training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].semilogy(self.history['train_loss'], lw=1.2, label='Train')
        axes[0].semilogy(self.history['val_loss'], lw=1.2, label='Val')
        axes[0].set(xlabel='Epoch', ylabel='MSE Loss', title='Training loss')
        axes[0].legend(); axes[0].grid(alpha=0.25)

        axes[1].plot(self.history['train_r2'], lw=1.2, label='Train')
        axes[1].plot(self.history['val_r2'], lw=1.2, label='Val')
        axes[1].axhline(0.9, color='r', ls='--', lw=1, alpha=0.6, label='R²=0.90')
        axes[1].set(xlabel='Epoch', ylabel='R²', title='R²', ylim=[-0.05, 1.05])
        axes[1].legend(); axes[1].grid(alpha=0.25)

        axes[2].semilogy(self.history['lr'], lw=1.2, color='g')
        axes[2].set(xlabel='Epoch', ylabel='LR', title='Learning rate')
        axes[2].grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), bbox_inches='tight')
        plt.close()

        # R² per mode
        r2m = self.eval_results['r2_per_mode']
        fig, ax = plt.subplots(figsize=(max(4, len(r2m)*1.5+1), 4))
        clrs = ['#2ecc71' if r>0.9 else '#f39c12' if r>0.7 else '#e74c3c' for r in r2m]
        bars = ax.bar(range(1, len(r2m)+1), r2m, color=clrs, alpha=0.85, edgecolor='k', lw=0.5)
        ax.axhline(0.9, color='r', ls='--', lw=1, alpha=0.6)
        for b, r2 in zip(bars, r2m):
            ax.text(b.get_x()+b.get_width()/2, r2+0.02, f'{r2:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax.set(xlabel='POD mode', ylabel='R²', title=f'R² per mode (overall={self.eval_results["r2_total"]:.4f})',
               xticks=range(1, len(r2m)+1), ylim=[0, 1.12])
        ax.grid(alpha=0.25, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'r2_per_mode.png'), bbox_inches='tight')
        plt.close()

        # Time series comparison
        n_test = self.y_pred_test.shape[0]
        n_show = min(4, self.n_modes)
        t = np.arange(self.n_timesteps)

        for ci in range(min(n_test, 6)):
            cid = self.te_idx[ci]
            fig, axes2 = plt.subplots(2, 2, figsize=(12, 8))
            axes2 = axes2.flatten()
            for mi in range(n_show):
                ax = axes2[mi]
                yt = self.y_test_raw[ci, :, mi]
                yp = self.y_pred_test[ci, :, mi]
                sr = np.sum((yt-yp)**2); st = np.sum((yt-yt.mean())**2)
                r2 = 1-sr/st if st>1e-12 else 0
                ax.plot(t, yt, 'b-', lw=1.5, label='CFD')
                ax.plot(t, yp, 'r--', lw=1.2, label='POD-NN')
                ax.fill_between(t, yt, yp, alpha=0.1, color='r')
                ax.set(xlabel='Time step', ylabel=f'a_{mi+1}(t)', title=f'Mode {mi+1} (R²={r2:.3f})')
                ax.legend(); ax.grid(alpha=0.25)
            for mi in range(n_show, 4):
                axes2[mi].set_visible(False)
            plt.suptitle(f'Test case {cid}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'time_series_case{cid}.png'), bbox_inches='tight')
            plt.close()

        print(f"OK Plots saved to: {save_dir}/")

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        hidden = self.config.get('hidden', [128, 128, 64])
        path = os.path.join(save_dir, 'podnn_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_input': int(self.X_cases.shape[1]),
                'n_output': int(self.n_output),
                'hidden_dims': list(hidden),
                'dropout': float(self.config.get('dropout', 0.25)),
            },
            'scaler_X_mean': self.scaler_X.mean_,
            'scaler_X_scale': self.scaler_X.scale_,
            'scaler_T_mean': self.scaler_T.mean_,
            'scaler_T_scale': self.scaler_T.scale_,
            'param_idx': self.param_idx,
            'param_names': self.param_names,
            'input_cols': self.input_cols,
            'feat_names': self.feat_names,
            'n_timesteps': int(self.n_timesteps),
            'n_modes': int(self.n_modes),
            'n_harmonics': int(self.config['n_harmonics']),
            'n_theta_per_mode': int(self.n_theta_per_mode),
            'eval_results': self.eval_results,
            'config': self.config,
            'tank_length': TANK_LENGTH,
        }, path)

        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(self.eval_results, f, indent=2, ensure_ascii=False)
        print(f"OK Model saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='POD-NN Case-to-Fourier (统一108组模型)')
    parser.add_argument('--data', default='data/all_108_merged.h5')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 128, 64])
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=16)  # V2: 8→16 for 108 cases
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--weight-decay', type=float, default=3e-2)
    parser.add_argument('--n-harmonics', type=int, default=6)  # V2: 5→6 for more complex dynamics
    parser.add_argument('--n-modes-use', type=int, default=None)
    parser.add_argument('--n-test', type=int, default=16)  # V2: 3→16 for 108 cases (15%)
    parser.add_argument('--n-val', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = vars(args)

    print(f"{'=' * 72}")
    print("POD-NN 训练 (Case-to-Fourier) — 统一108组模型")
    print(f"  [{datetime.now():%Y-%m-%d %H:%M:%S}]")
    print(f"{'=' * 72}")

    trainer = FourierCaseTrainer(config)
    X, y = trainer.load_data(args.data)
    theta = trainer.build_targets(y)
    trainer.prepare_data(X, theta, n_test=args.n_test, n_val=args.n_val)
    trainer.build_model()
    trainer.train()
    results = trainer.evaluate()
    trainer.plot_results(args.output_dir)
    trainer.save_model(args.output_dir)

    r2t = results['r2_total']
    n_good = sum(1 for r in results['r2_per_mode'] if r > 0.9)
    print(f"\n{'=' * 72}")
    print(f"Complete | Total R²: {r2t:.4f} | R²>0.90: {n_good}/{len(results['r2_per_mode'])}")
    if r2t >= 0.95:
        print('  Excellent!')
    elif r2t >= 0.90:
        print('  Good - target reached.')
    else:
        print('  Try: --n-harmonics 8 --n-modes-use 6 --hidden 256 128 64 --dropout 0.30')
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()