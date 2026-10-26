"""统一改图入口：颜色、字体、线型、尺寸、分辨率和图 1 几何参数。"""
from pathlib import Path
from matplotlib import pyplot as plt, font_manager

DPI = 1200
FORMATS = ('png', 'pdf', 'svg')  # 可改为 ('png', 'pdf', 'svg')；PDF、SVG 为矢量格式。
COLORS = {'black': '#111111', 'orange': '#D55E00', 'blue': '#0072B2', 'gray': '#7F7F7F'}
# 线型兼顾黑白印刷：CFD 实线、本文模型虚线、基准模型点线，峰值用"×"标记。
LINES = {
    'cfd': dict(color='#111111', lw=1.05, ls='-'),
    'proposed': dict(color='#0072B2', lw=1.10, ls=(0, (4, 2.2))),
    'baseline': dict(color='#7F7F7F', lw=0.95, ls=(0, (1, 1.6))),
}
PEAK_MARKER = dict(marker='x', color='#D55E00', ms=5.5, mew=1.3, ls='none')
FIGURE_SIZES_MM = {
    'fig01_geometry': (174, 65),
    'fig02_validation': (84, 67),
    'fig04_braking_turning': (174, 112),
    'fig05_overtaking_grade_entry': (174, 68),
    'fig06_grade_entry_peak': (174, 74),
}
GEOMETRY = {
    'a_m': 1.2,
    'b_m': 0.9,
    'length_m': 7.5,
    'view_elevation_deg': 16.0,
    'view_rotation_deg': 65.0,
}
FONT_FAMILY = 'FangSong'  # 优先使用仿宋，回退字体为 Noto Serif CJK SC。
LATIN_FONT_CANDIDATES = ('Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif')  # 刻度数字用 Times 体
FONT_FILE = Path(__file__).resolve().parent / 'fonts' / 'NotoSerifCJKsc-Regular.otf'
FONT_SCALE = 1.0  # 全部图中文字同比放大；例如 1.10 表示增大 10%。
BASE_FONT_SIZE = 8.5
PANEL_TITLE_SIZE = 8.5
CFD_LINE_WIDTH = 1.05
PREDICTION_LINE_WIDTH = 1.10
PREDICTION_DASHES = (0, (4, 2.4))


def pt(size):
    return size * FONT_SCALE


def figure_size(stem):
    """以毫米设定最终画布；不使用 tight bbox，避免插入 Word 时比例漂移。"""
    return tuple(value / 25.4 for value in FIGURE_SIZES_MM[stem])


def chinese_font():
    """按候选顺序选择已安装字体，并注册项目内的中文字体。"""
    if FONT_FILE.is_file():
        font_manager.fontManager.addfont(str(FONT_FILE))
    candidates = (FONT_FAMILY, '仿宋', 'FangSong_GB2312', '仿宋_GB2312',
                  'Noto Serif CJK SC', 'Source Han Serif SC')
    for candidate in candidates:
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            return candidate
        except ValueError:
            continue
    raise RuntimeError('未找到中文字体。请保留随包 fonts 文件夹，或在 style.py 指定已安装的中文字体。')


def latin_font():
    for candidate in LATIN_FONT_CANDIDATES:
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            return candidate
        except ValueError:
            continue
    return 'DejaVu Serif'


LATIN_FONT_FAMILY = latin_font()


def apply_style():
    plt.rcdefaults()
    chinese = chinese_font()
    plt.rcParams.update({
        'font.family': chinese, 'font.size': pt(BASE_FONT_SIZE),
        'axes.labelsize': pt(8.5), 'axes.titlesize': pt(8.5), 'axes.titlepad': 4.0,
        'legend.fontsize': pt(7.5), 'legend.frameon': False, 'legend.handlelength': 1.8,
        'xtick.labelsize': pt(7.5), 'ytick.labelsize': pt(7.5),
        'axes.linewidth': .8, 'lines.linewidth': 1.15, 'lines.markersize': 3.6,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'xtick.major.size': 3., 'ytick.major.size': 3.,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': True, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'svg.fonttype': 'path',
        'path.simplify': False,
    })


def apply_geometry_style():
    """优先保留图 1 原字体；Windows 无 Nimbus Roman 时自动回退。"""
    plt.rcdefaults()
    roman = 'DejaVu Serif'
    for candidate in ('Nimbus Roman', 'Times New Roman', 'DejaVu Serif'):
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            roman = candidate
            break
        except ValueError:
            continue
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [roman, 'DejaVu Serif'],
        'font.size': pt(9.5), 'mathtext.fontset': 'custom',
        'mathtext.rm': roman, 'mathtext.it': roman + ':italic',
        'mathtext.bf': roman + ':bold', 'mathtext.cal': roman + ':italic',
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'path',
    })
