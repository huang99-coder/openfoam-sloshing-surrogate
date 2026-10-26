"""生成六幅研究图件，输出至 figures/。"""
from pathlib import Path
import importlib, importlib.metadata, json, struct, sys, traceback
sys.dont_write_bytecode = True
from check_data import verify_data

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'figures'
FIGURES = [
    (1,'fig01_geometry','椭圆罐几何与坐标'),
    (2,'fig02_validation','矩形容器自由液面验证'),
    (3,'fig03_model_framework','载荷时程与峰值双分支模型'),
    (4,'fig04_braking_turning','制动与转弯独立工况'),
    (5,'fig05_overtaking_grade_entry','变道与坡道独立工况'),
    (6,'fig06_grade_entry_peak','坡道瞬态峰值与级数表示'),
]


def main():
    OUTPUT.mkdir(exist_ok=True)
    checks = verify_data()
    print('数据校验通过：7个CSV，6个独立载荷工况。',flush=True)
    print('R2均值=%.3f；峰值直接回归MARE=%.1f%%。' %
          (checks['mean_time_R2'], checks['direct_peak_MARE_pct']),flush=True)
    audits=[]
    for number,module_name,title in FIGURES:
        audit=importlib.import_module(module_name).main(OUTPUT)
        paths={fmt:OUTPUT / (audit['stem']+'.'+fmt) for fmt in ('png','pdf','svg')}
        for path in paths.values():
            if not path.is_file() or path.stat().st_size==0:
                raise RuntimeError('缺少绘图输出：'+str(path))
        width_px,height_px=struct.unpack('>II',paths['png'].read_bytes()[16:24])
        audit.update(figure=number,title=title,width_px=width_px,height_px=height_px,
            relative_paths={k:p.relative_to(ROOT).as_posix() for k,p in paths.items()})
        audits.append(audit)
        print('已生成 图%d：%s（PNG %d dpi；PDF、SVG矢量）' %
              (number,audit['stem'],audit['png_dpi']),flush=True)
    manifest=dict(figures=audits, training_performed=False,
        data_source='随包data；沿用原绘图数据，不生成新模型预测',
        environment={'python':sys.version.split()[0],**{
            name:importlib.metadata.version(name) for name in ('numpy','pandas','matplotlib')}})
    (OUTPUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUTPUT/'data_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print('完成。输出目录：'+str(OUTPUT),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        print('绘图失败：请根据上述错误检查依赖、字体或数据路径。',file=sys.stderr)
        sys.exit(1)
