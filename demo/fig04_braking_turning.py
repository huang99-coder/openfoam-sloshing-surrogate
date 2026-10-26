"""图4：制动与转弯独立工况的载荷时程。"""
import sys
sys.dont_write_bytecode = True
from common import plt, load_gt, plot_case, panel_label, save
from style import apply_style, figure_size, pt

PANELS = [('GT-BR-01', 'z', r'(a) GT-BR-01：制动0.45$g$，55%，水'), ('GT-BR-02', 'z', r'(b) GT-BR-02：制动0.65$g$，45%，柴油'),
          ('GT-TR-01', 'x', '(c) GT-TR-01：75 m，45 km/h，65%'), ('GT-TR-02', 'x', '(d) GT-TR-02：55 m，52 km/h，48%')]


def main(output_dir=None):
    apply_style(); curves, metrics = load_gt()
    fig, axs = plt.subplots(2, 2, figsize=figure_size('fig04_braking_turning'))
    fig.subplots_adjust(left=.078, right=.985, bottom=.16, top=.90, hspace=.90, wspace=.24)
    for ax, (case, comp, label) in zip(axs.ravel(), PANELS):
        plot_case(ax, case, curves, metrics, comp); panel_label(ax, label, dy=-.40)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.0), fontsize=pt(7.5))
    return save(fig, 'fig04_braking_turning', output_dir, dict(source='data/gt_curves.csv, data/gt_metrics.csv', smoothing='none'))


if __name__ == '__main__':
    print('已生成：', main()['stem'])
