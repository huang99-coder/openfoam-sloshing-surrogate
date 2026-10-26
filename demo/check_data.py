"""校验 CSV 指纹、独立工况 ID 和载荷评价统计。"""
from pathlib import Path
import csv, hashlib, json

ROOT = Path(__file__).resolve().parent


def verify_data():
    expected = json.loads((ROOT / 'data_sha256.json').read_text(encoding='utf-8'))
    mismatch = []
    for name, digest in expected.items():
        file = ROOT / 'data' / name
        actual = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
        if actual != digest:
            mismatch.append(name)
    if mismatch:
        raise ValueError('随包数据与原始指纹不一致，请核对来源：' + ', '.join(mismatch))
    with (ROOT / 'data' / 'gt_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [row['case_id'] for row in rows]
    expected_ids = ['GT-BR-01','GT-BR-02','GT-TR-01','GT-TR-02','GT-OT-01','GT-SL-01']
    if sorted(ids) != sorted(expected_ids):
        raise ValueError('独立工况清单与小论文6组不一致')
    avg = lambda xs: sum(xs) / len(xs)
    r2 = avg([float(row['proposed_R2']) for row in rows])
    mare = avg([abs(float(row['peak_model_err_pct'])) for row in rows])
    five = [abs(float(row['peak_model_err_pct'])) for row in rows if row['case_id'] != 'GT-SL-01']
    result = dict(csv_count=len(expected), csv_hashes_match=True, independent_cases=ids,
        mean_time_R2=r2, direct_peak_MARE_pct=mare,
        non_grade_direct_peak_MARE_pct=avg(five), non_grade_max_abs_error_pct=max(five),
        note='汇总表的既有结果；gt_curves.csv每组1000个绘图点，不等同于2000点评价预测。')
    return result


if __name__ == '__main__':
    print(json.dumps(verify_data(),ensure_ascii=False,indent=2))
