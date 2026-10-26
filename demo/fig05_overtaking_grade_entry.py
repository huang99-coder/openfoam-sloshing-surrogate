"""图5：双次变道与上坡独立工况的载荷时程。"""
import sys
sys.dont_write_bytecode = True
from common import plt, load_gt, plot_case, panel_label, save
from style import apply_style, figure_size, pt

PANELS = [('GT-OT-01', 'x', '(a) GT-OT-01：双次变道，2.2 s，55%'), ('GT-SL-01', 'z', '(b) GT-SL-01：上坡5°，62%')]


def main(output_dir=None):
    apply_style(); curves, metrics = load_gt()
    fig, axs = plt.subplots(1, 2, figsize=figure_size('fig05_overtaking_grade_entry'))
    fig.subplots_adjust(left=.078, right=.985, bottom=.29, top=.84, wspace=.24)
    for ax, (case, comp, label) in zip(axs, PANELS):
        plot_case(ax, case, curves, metrics, comp); panel_label(ax, label, dy=-0.34)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.0), fontsize=pt(7.5))
    return save(fig, 'fig05_overtaking_grade_entry', output_dir, dict(source='data/gt_curves.csv, data/gt_metrics.csv', smoothing='none'))


if __name__ == '__main__':
    print('已生成：', main()['stem'])
