#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_overtaking_v5_offgrid.py
=================================
读取“预先锁定”的 v5 预测 CSV + 已完成的新增 CFD case 目录，
从 forcesTotal/moment.dat 提取 max|Mz|，输出真正独立验证结果。

默认不训练、不调参、不重新选择架构。
"""
from __future__ import annotations
import argparse, csv, hashlib, re
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

FLOAT_RE=re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def read_predictions(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def moment_files(case):
    cands=[]
    for base in ['forcesTotal','forces']:
        p=case/'postProcessing'/base
        if p.exists():
            cands += list(p.glob('**/moment.dat'))
    return sorted(set(cands))

def read_max_abs_mz_kNm(case):
    fs=moment_files(case)
    if not fs:
        raise FileNotFoundError(f"{case}: 未找到 postProcessing/**/moment.dat")
    vals=[]
    for fp in fs:
        for line in fp.read_text(encoding='utf-8',errors='ignore').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            nums=[float(x) for x in FLOAT_RE.findall(line)]
            if len(nums)>=4:
                # 标准 moment.dat: time Mx My Mz；若括号存在，正则仍提取同样的前4个数
                vals.append(nums[3])
    if not vals:
        raise ValueError(f"{case}: moment.dat 无有效 Mz")
    return max(abs(np.asarray(vals,dtype=float)))/1000.0

def metric(rows,key_true,key_pred):
    y=np.array([float(r[key_true]) for r in rows])
    p=np.array([float(r[key_pred]) for r in rows])
    r2=float(r2_score(y,p)) if np.var(y)>1e-30 else float('nan')
    mae=float(mean_absolute_error(y,p))
    mean_abs=max(float(np.mean(np.abs(y))),1e-30)
    return r2,mae,mae/mean_abs,mean_abs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',default='results/overtaking_v5_offgrid_predictions_LOCKED.csv')
    ap.add_argument('--cases-dir',required=True,help='16 个新增 case 文件夹所在根目录')
    ap.add_argument('--mref',type=float,default=273.6)
    ap.add_argument('--out-prefix',default='results/overtaking_v5_offgrid_independent')
    args=ap.parse_args()

    pred=read_predictions(Path(args.predictions))
    root=Path(args.cases_dir)
    outrows=[]
    failures=[]
    for r in pred:
        case=root/r['folder']
        try:
            true=read_max_abs_mz_kNm(case)
            rr=dict(r)
            rr['Mz_abs_true_kNm']=f'{true:.10f}'
            rr['RSI_O_true']=f'{true/args.mref:.10f}'
            rr['abs_error_kNm']=f'{abs(true-float(r["Mz_abs_pred_kNm"])):.10f}'
            rr['RSI_O_abs_error']=f'{abs(true/args.mref-float(r["RSI_O_pred"])):.10f}'
            outrows.append(rr)
        except Exception as e:
            failures.append((r['validation_id'],r['folder'],str(e)))

    if failures:
        for x in failures: print("FAIL",*x)
        raise SystemExit(f"❌ {len(failures)} 个 case 缺结果；先完成全部 CFD 再做最终评价。")

    op=Path(args.out_prefix); op.parent.mkdir(parents=True,exist_ok=True)
    cases_csv=Path(str(op)+'_cases.csv')
    with open(cases_csv,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outrows[0].keys()))
        w.writeheader(); w.writerows(outrows)

    groups={
        'ALL':outrows,
        'single':[r for r in outrows if int(float(r['is_double_overtake']))==0],
        'double':[r for r in outrows if int(float(r['is_double_overtake']))==1],
        'high_fill_75':[r for r in outrows if abs(float(r['fill_level'])-0.75)<1e-9],
    }
    sums=[]
    for name,rr in groups.items():
        r2,mae,rel,mean_abs=metric(rr,'Mz_abs_true_kNm','Mz_abs_pred_kNm')
        sums.append({
            'subset':name,'n':len(rr),'R2':f'{r2:.8f}','MAE_kNm':f'{mae:.8f}',
            'MAE_over_mean_abs':f'{rel:.8f}','mean_abs_Mz_kNm':f'{mean_abs:.8f}',
            'RSI_O_MAE':f'{mae/args.mref:.8f}'
        })
    summary_csv=Path(str(op)+'_summary.csv')
    with open(summary_csv,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sums[0].keys()))
        w.writeheader(); w.writerows(sums)

    print("Independent off-grid validation:")
    for s in sums:
        print(f"{s['subset']:<12} n={s['n']:<2} R²={float(s['R2']):.4f} "
              f"MAE={float(s['MAE_kNm']):.3f} kN·m "
              f"MAE/量级={float(s['MAE_over_mean_abs']):.1%} "
              f"RSI_O MAE={float(s['RSI_O_MAE']):.5f}")
    print(f"cases   -> {cases_csv}")
    print(f"summary -> {summary_csv}")
    print("评价对象为预先冻结模型；依据本批结果重新选模后，应另设独立确认数据。")

if __name__=='__main__':
    main()
