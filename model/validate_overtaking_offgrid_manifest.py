#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查新增 off-grid manifest 是否与 v5 所需 5 个特征兼容。"""
import csv, math, argparse
from pathlib import Path
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='data/overtaking_offgrid_manifest.csv')
    a=ap.parse_args()
    rows=list(csv.DictReader(open(a.manifest,encoding='utf-8-sig')))
    req=['fill_level','peak_gx','total_time','t_change','is_double_overtake']
    for c in req:
        if c not in rows[0]: raise SystemExit(f'缺列 {c}')
    pairs={}
    for r in rows:
        k=r['pair_id']; pairs.setdefault(k,[]).append(r)
        T=float(r['t_change']); gx=float(r['peak_gx'])
        exp=math.pi**2*3.5/(2*T*T)
        if abs(gx-exp)>2e-4: raise SystemExit(f"{r['validation_id']} peak_gx mismatch")
    if len(rows)!=16 or len(pairs)!=8 or any(len(v)!=2 for v in pairs.values()):
        raise SystemExit(f'设计应为 8 pair / 16 cases；实际 pairs={len(pairs)} rows={len(rows)}')
    print('PASS: manifest = 8 pairs / 16 cases; v5 features complete.')
if __name__=='__main__': main()
