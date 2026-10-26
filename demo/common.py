"""公共函数：路径、坐标轴格式、子图标注与保存。脚本按自身位置定位文件，不依赖工作目录。"""
from pathlib import Path
import json
from io import BytesIO
from matplotlib import patheffects
from matplotlib.text import Text
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, ticker
from style import DPI, FORMATS, FIGURE_SIZES_MM, LATIN_FONT_FAMILY, LINES, PEAK_MARKER, pt, chinese_font, PANEL_TITLE_SIZE

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'figures'


def format_ax(ax, nx=5, ny=5):
    """开放式坐标框、刻度朝内；刻度数字使用 Times 体。"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', top=False, right=False, pad=2)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nx, min_n_ticks=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=ny, min_n_ticks=4))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(LATIN_FONT_FAMILY)


def panel_label(ax, text, dy=-0.36):
    """子图名置于子图下方居中，符合中文期刊惯例。"""
    ax.text(0.5, dy, text, transform=ax.transAxes, ha='center', va='top', fontsize=pt(PANEL_TITLE_SIZE))


def save(fig, stem, output_dir, details):
    output = Path(output_dir) if output_dir is not None else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    width, height = FIGURE_SIZES_MM[stem]
    assert np.allclose(fig.get_size_inches() * 25.4, [width, height])
    # Final-size alignment; the canvas and all plotted sample values stay fixed.
    fig.align_ylabels()
    fig.align_xlabels()
    fig.canvas.draw()
    texts = fig.findobj(match=Text)
    original_effects = [text.get_path_effects() for text in texts]
    for suffix in FORMATS:
        # Normal path effect routes PDF text through glyph outlines, preventing
        # OTF/Type42 CID mapping failures without rasterizing the curves.
        if suffix == 'pdf':
            for text in texts:
                text.set_path_effects([patheffects.Normal()])
        else:
            for text, effects in zip(texts, original_effects):
                text.set_path_effects(effects)
        buffer = BytesIO()
        fig.savefig(buffer, format=suffix, dpi=DPI, facecolor='white', bbox_inches=None)
        target = output / (stem + '.' + suffix)
        temporary = target.with_name(target.name + '.tmp')
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(target)
    for text, effects in zip(texts, original_effects):
        text.set_path_effects(effects)
    plt.close(fig)
    audit = dict(stem=stem, width_mm=width, height_mm=height, png_dpi=DPI,
                 formats=list(FORMATS), chinese_font=chinese_font(),
                 pdf_text='vector glyph outlines', svg_text='vector glyph outlines', **details)
    (output / (stem + '.json')).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit


def load_gt():
    return pd.read_csv(DATA / 'gt_curves.csv'), pd.read_csv(DATA / 'gt_metrics.csv').set_index('case_id')


def plot_case(ax, case, curves, metrics, comp):
    """CFD、傅里叶基准模型、本文时程模型及峰值直接预测（×，标在CFD峰值时刻）。"""
    d = curves[curves.case_id == case]
    t, y = d.time_s.to_numpy(), d.cfd_kN.to_numpy()
    ax.plot(t, y, label='CFD', **LINES['cfd'])
    ax.plot(t, d.baseline_kN, label='傅里叶基准模型', **LINES['baseline'])
    ax.plot(t, d.proposed_kN, label='本文时程模型', **LINES['proposed'])
    i = int(np.argmax(np.abs(y)))
    ax.plot(t[i], np.sign(y[i]) * metrics.loc[case, 'peak_model_pred_kN'], label='峰值直接预测', **PEAK_MARKER)
    ax.set_xlim(t[0], t[-1]); ax.set_xlabel(r'时间 $t$/s', labelpad=1.5)
    ax.set_ylabel(r'$F_{%s}$/kN' % comp, labelpad=2)
    format_ax(ax)
