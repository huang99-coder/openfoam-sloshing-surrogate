#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_overtaking_v5_offgrid.py — 冻结模型的独立工况预登记预测。

使用 overtaking_v5_final_model.joblib 生成新增 off-grid 工况的预测 CSV。
独立评价采用 CFD 结果获取前固定的模型与预测记录。

默认路径：
  model    = results/overtaking_v5_final_model.joblib
  manifest = data/overtaking_offgrid_manifest.csv
  out      = results/overtaking_v5_offgrid_predictions_LOCKED.csv

PAIR-DELTA：
  single -> base
  double -> base + delta

默认保留已有 LOCKED CSV；--force 显式覆盖已有输出。覆盖后的文件应作为
新的预测版本管理，独立评价仍以取得 CFD 结果前的冻结记录为准。
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

def sha256(path: Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def load_manifest(path: Path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def fval(row,name):
    aliases={
        'fill_level':'fill_level',
        'peak_gx':'peak_gx',
        'total_time':'total_time',
        't_change':'t_change',
        'is_double_overtake':'is_double_overtake',
    }
    if name not in aliases:
        raise KeyError(f"manifest 不支持模型特征 {name}; 当前仅支持 {sorted(aliases)}")
    return float(row[aliases[name]])

def predict_one(obj,row):
    arch=obj['architecture']
    if arch=='RAW-POOLED':
        names=obj['feature_names']
        x=np.array([[fval(row,n) for n in names]],dtype=float)
        return float(np.asarray(obj['model'].predict(x)).reshape(-1)[0])
    if arch=='PAIR-DELTA':
        bn=obj['base_feature_names']
        dn=obj['delta_feature_names']
        xb=np.array([[fval(row,n) for n in bn]],dtype=float)
        base=float(np.asarray(obj['base_model'].predict(xb)).reshape(-1)[0])
        if float(row['is_double_overtake']) < 0.5:
            return base
        xd=np.array([[fval(row,n) for n in dn]],dtype=float)
        delta=float(np.asarray(obj['delta_model'].predict(xd)).reshape(-1)[0])
        return base+delta
    raise ValueError(f"未知 architecture={arch}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='results/overtaking_v5_final_model.joblib')
    ap.add_argument('--manifest',default='data/overtaking_offgrid_manifest.csv')
    ap.add_argument('--out',default='results/overtaking_v5_offgrid_predictions_LOCKED.csv')
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()

    mp=Path(args.model); mf=Path(args.manifest); out=Path(args.out)
    if not mp.exists():
        raise SystemExit(f"❌ 缺少冻结模型 {mp}\n需要 p3_overtaking_v5.py 生成的 results/overtaking_v5_final_model.joblib。")
    if not mf.exists():
        raise SystemExit(f"❌ 缺少 manifest {mf}")
    if out.exists() and not args.force:
        raise SystemExit(f"❌ {out} 已存在。为保持独立验证证据，默认禁止覆盖。")
    out.parent.mkdir(parents=True,exist_ok=True)

    obj=joblib.load(mp)
    rows=load_manifest(mf)
    stamp=datetime.now(timezone.utc).isoformat()
    model_hash=sha256(mp)
    manifest_hash=sha256(mf)
    mref=float(obj.get('mref_kNm',273.6))

    outrows=[]
    for r in rows:
        pred=predict_one(obj,r)
        rr=dict(r)
        rr.update({
            'Mz_abs_pred_kNm':f'{pred:.10f}',
            'RSI_O_pred':f'{pred/mref:.10f}',
            'model_architecture':obj['architecture'],
            'model_version':str(obj.get('version','unknown')),
            'prediction_locked_utc':stamp,
            'model_sha256':model_hash,
            'manifest_sha256':manifest_hash,
        })
        outrows.append(rr)

    with open(out,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outrows[0].keys()))
        w.writeheader(); w.writerows(outrows)

    lock=out.with_suffix(out.suffix+'.sha256.txt')
    lock.write_text(
        f"prediction_file_sha256={sha256(out)}\n"
        f"model_sha256={model_hash}\nmanifest_sha256={manifest_hash}\n"
        f"locked_utc={stamp}\narchitecture={obj['architecture']}\n",
        encoding='utf-8'
    )

    print(f"architecture={obj['architecture']}")
    print(f"model sha256={model_hash}")
    print(f"manifest sha256={manifest_hash}")
    print(f"LOCKED predictions -> {out}")
    print(f"lock record       -> {lock}")
    print("预测已冻结。独立评价使用新增 CFD 结果与此版本预测进行比较。")

if __name__=='__main__':
    main()
