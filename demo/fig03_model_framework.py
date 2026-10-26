"""KRR/SVR load-history and peak-regression architecture diagram.

The editable SVG retains text; the companion SVG/PDF use outlined lettering.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager, patheffects
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FONT = ROOT / "fonts" / "NotoSerifCJKsc-Regular.otf"


def main(output_dir=None):
    if not FONT.is_file():
        raise FileNotFoundError("请保留随包 fonts 文件夹：" + str(FONT))
    font_manager.fontManager.addfont(str(FONT))
    output = Path(output_dir) if output_dir is not None else ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Serif CJK SC','font.size':9.5,
        'mathtext.fontset':'stix','axes.unicode_minus':False,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.transparent':False})
    
    W,H=174,60
    NAVY='#294F6B'; TEAL='#3C7773'; INK='#20272C'; GRAY='#6C757B'
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
    
    def text(x,y,s,size=9.5,color=INK):
        return ax.text(x,y,s,ha='center',va='center',fontsize=size,color=color)
    def box(x,y,w,h,color=GRAY):
        ax.add_patch(Rectangle((x,y),w,h,edgecolor=color,facecolor='white',lw=.65))
    def line(x1,y1,x2,y2,color=GRAY):
        ax.plot([x1,x2],[y1,y2],color=color,lw=.7,solid_capstyle='butt')
    def arrow(points,color):
        for a,b in zip(points[:-2],points[1:-1]): line(*a,*b,color)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
            mutation_scale=6.2,color=color,lw=.7,shrinkA=0,shrinkB=0))
    
    # Six nodes, one shared input, and two independent output branches.
    box(4,20,28,20)
    text(18,26,'工况参数')
    text(18,34.2,r'$\mathbf{x}$',14)
    
    for y,color in [(6,NAVY),(36,TEAL)]:
        box(46,y,34,18,color)
        box(144,y,26,18,color)
    
    box(97,6,30,18,NAVY)
    text(63,11.8,'KRR',10.8,NAVY)
    text(63,18.3,'系数回归')
    text(112,15,'级数重构')
    text(157,11.8,'载荷时程')
    text(157,18.7,r'$\hat{F}(t)$',12.5,NAVY)
    
    text(63,41.8,'SVR / KRR',10.2,TEAL)
    text(63,48.3,'峰值回归')
    text(157,41.8,'峰值')
    text(157,48.7,r'$\hat{F}_{\mathrm{peak}}$',12.5,TEAL)
    
    line(32,30,38,30)
    arrow([(38,30),(38,15),(46,15)],NAVY)
    arrow([(38,30),(38,45),(46,45)],TEAL)
    arrow([(80,15),(97,15)],NAVY)
    arrow([(127,15),(144,15)],NAVY)
    arrow([(80,45),(144,45)],TEAL)
    
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflow=[]
    for t in ax.texts:
        b=t.get_window_extent(renderer)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:
            overflow.append(t.get_text())
    assert not overflow,overflow
    fig.savefig(output/'fig03_model_framework.png',dpi=900,facecolor='white')
    plt.rcParams['svg.fonttype']='none'
    fig.savefig(output/'fig03_model_framework_editable.svg',facecolor='white')
    plt.rcParams['svg.fonttype']='path'
    fig.savefig(output/'fig03_model_framework.svg',facecolor='white')
    for t in ax.texts: t.set_path_effects([patheffects.Normal()])
    fig.savefig(output/'fig03_model_framework.pdf',facecolor='white')
    plt.close(fig)
    audit = dict(stem="fig03_model_framework", width_mm=W, height_mm=H,
        png_dpi=900, formats=["png","pdf","svg"], chinese_font="Noto Serif CJK SC",
        minimum_text_pt=9.5, nodes=6, training_performed=False, data_curves_used=False,
        diagram_scope="inference architecture only",
        editable_svg="fig03_model_framework_editable.svg")
    (output / "fig03_model_framework.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    return audit


if __name__ == "__main__":
    print("已生成：", main()["stem"])
