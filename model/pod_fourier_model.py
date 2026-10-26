#!/usr/bin/env python3
"""
POD-NN 模型定义 (Case-to-Fourier)

输入:  工况参数 (n_params)
输出:  每个 POD 模态的 Fourier 系数 theta
       维度 = n_modes * (1 + 2 * n_harmonics)

通过低维时间参数化显著减少输出自由度，
适合小样本场景（如 30 个工况）。
"""

import torch
import torch.nn as nn


class ParamPredictorMLP(nn.Module):
    """工况参数 -> Fourier 参数向量"""

    def __init__(self, n_input, n_output,
                 hidden_dims=(128, 128, 64), dropout=0.25):
        super().__init__()
        layers = []
        in_dim = n_input
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_model(n_input, n_output,
                 hidden_dims=(128, 128, 64), dropout=0.25, **kwargs):
    return ParamPredictorMLP(
        n_input=n_input,
        n_output=n_output,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
