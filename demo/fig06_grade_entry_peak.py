"""图6：进坡冲击峰特征——(a) 余弦级数直接拟合的表示误差；(b) 上坡工况峰值随坡度变化。"""
import numpy as np
import pandas as pd
import sys
sys.dont_write_bytecode = True
from common import DATA, plt, format_ax, panel_label, save
from style import COLORS, LINES, LATIN_FONT_FAMILY, apply_style, figure_size, pt

N = 2000


def cosine_fit(t, F, K):
    tg = np.linspace(t[0], t[-1], N); y = np.interp(tg, t, F); tau = (tg - tg[0]) / (tg[-1] - tg[0])
    B = np.stack([np.cos(np.pi * k * tau) for k in range(K + 1)], 1)
    return tg, y, B @ np.linalg.lstsq(B, y, rcond=None)[0]


def main(output_dir=None):
    apply_style(); r = pd.read_csv(DATA / 'grade_entry_GT-SL-01_raw.csv'); t, F = r.time_s.to_numpy(), r.Fz_kN.to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=figure_size('fig06_grade_entry_peak'))
    fig.subplots_adjust(left=.075, right=.985, bottom=.27, top=.95, wspace=.24)
    ax = axs[0]; w = (t > 6.5) & (t < 7.6); ax.plot(t[w], F[w], label='原始 CFD', **LINES['cfd'])
    errs = {}
    for K, sty in [(12, dict(color=COLORS['gray'], lw=.95, ls=(0, (1, 1.6)))), (40, dict(color=COLORS['blue'], lw=1.1, ls=(0, (4, 2.2)))), (120, dict(color=COLORS['orange'], lw=1.0, ls='-.'))]:
        tg, y, p = cosine_fit(t, F, K); errs[K] = (np.abs(p).max() - np.abs(y).max()) / np.abs(y).max() * 100
        m = (tg > 6.5) & (tg < 7.6); ax.plot(tg[m], p[m], label=f'余弦拟合 $K$={K}', **sty)
    ax.set_xlim(6.5, 7.6); ax.set_xlabel(r'时间 $t$/s', labelpad=1.5); ax.set_ylabel(r'$F_{z}$/kN', labelpad=2)
    format_ax(ax); ax.legend(loc='lower left', fontsize=pt(7.5), handletextpad=.4)
    panel_label(ax, '(a) 局部峰值与级数表示', dy=-0.30)
    ax = axs[1]; p = pd.read_csv(DATA / 'grade_entry_uphill_peaks.csv')
    marks = {40: 'o', 50: 's', 60: '^', 70: 'D', 80: 'v'}
    for f, g in p.groupby('fill_pct'):
        g = g.sort_values('angle_deg'); ax.plot(g.angle_deg, g.peak_abs_Fz_kN, marker=marks[f], ms=3.4, mfc='white', lw=.8, label=f'{f}%')
    ax.plot([5], [np.abs(np.interp(np.linspace(t[0], t[-1], N), t, F)).max()], marker='*', ms=8.5, color=COLORS['orange'], ls='none', label='GT-SL-01（62%，重采样）')
    ax.set_xlabel(r'坡度 $\theta$/(°)', labelpad=1.5); ax.set_ylabel(r'峰值 $|F_{z}|_{\max}$/kN', labelpad=2)
    format_ax(ax); ax.set_xticks(range(3, 10)); ax.set_ylim(0, 108)
    for label in ax.get_xticklabels(): label.set_fontfamily(LATIN_FONT_FAMILY)
    ax.legend(loc='upper left', ncol=3, fontsize=pt(7.5), handletextpad=.3, columnspacing=.7, borderaxespad=.2)
    panel_label(ax, '(b) 上坡峰值', dy=-0.30)
    return save(fig, 'fig06_grade_entry_peak', output_dir, dict(source='data/grade_entry_GT-SL-01_raw.csv, data/grade_entry_uphill_peaks.csv',
        peak_definition='Panel a: raw CFD; panel b star and projection errors: 2000-point resampled series; definitions unchanged',
                                                               projection_peak_error_pct={str(k): round(v, 2) for k, v in errs.items()}))


if __name__ == '__main__':
    print('已生成：', main()['stem'])
