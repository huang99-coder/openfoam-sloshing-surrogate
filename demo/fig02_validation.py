"""fig02_validation: 可独立运行；仅绘图，不重新训练或滤波。"""
import numpy as np
import pandas as pd
import sys
sys.dont_write_bytecode = True
from common import DATA, plt, format_ax, save
from style import COLORS, apply_style, figure_size, pt


def main(output_dir=None):
    apply_style()
    # 图2只显示探针可解析区间，不作平移或重新归零。
    cfd = pd.read_csv(DATA / 'validation_cfd.csv')
    cfd=cfd[(cfd.t_s>=0)&(cfd.t_s<=6.68)&cfd.eta_m.notna()]
    exp = pd.read_csv(DATA / 'validation_experiment.csv')
    assert len(exp)==94
    fig,ax=plt.subplots(figsize=figure_size('fig02_validation'))
    fig.subplots_adjust(left=.18,right=.975,bottom=.18,top=.84)
    ax.plot(cfd.t_s,cfd.eta_m,color=COLORS['black'],lw=1.05,label='CFD')
    ax.plot(exp.t_s,exp.eta_experiment_m,ls='none',marker='o',ms=2.65,
            mfc='white',mec=COLORS['orange'],mew=.65,label='实验（数字化）')
    ax.set_xlim(0,6.78)  # 仅在右侧留白；实际绘图数据仍不超过6.68 s。
    ax.set_xlabel(r'时间 $t$/s',labelpad=2)
    ax.set_ylabel(r'液面位移 $\eta$/m',labelpad=3)
    ax.margins(y=.075)
    format_ax(ax)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.54,.985),ncol=2,fontsize=pt(7.5),
               handletextpad=.45,columnspacing=1.0)
    return save(fig,'fig02_validation', output_dir, dict(
        sources=['data/validation_cfd.csv', 'data/validation_experiment.csv'],
        interval_s=[0,6.68],experimental_points=len(exp),
        cfd_points=len(cfd),eta_reference_m=.15,
        processing='alpha=0.5 crossing minus 0.15 m; no shift or fit; no over-range points'))


if __name__ == '__main__':
    result = main()
    print('已生成：', result['stem'])
