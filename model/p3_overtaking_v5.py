#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p3_overtaking_v5.py
====================
超车/变道工况 max|Mz| / RSI_O 专用代理 v5。

目标与候选架构
--------------
RSI_O 采用绝对力矩峰值 max|Mz|。当前实现比较以下两个候选架构：
A. RAW-POOLED
   使用原始 5 个描述符联合建模：
   fill_level, peak_gx, total_time, t_change, is_double_overtake。
   重点扩展低 gamma 的 RBF-SVR 网格，同时保留 Ridge/KRR 作为竞争基线。

B. PAIR-DELTA
   利用单/双工况天然成对这一结构：
       y_double = y_single + Delta
   先学习单次基础峰值，再学习双次相对同参数单次的增量 Delta。
   基础模型与增量模型利用单次、双次工况的配对关系共享信息。

评价协议
--------
- 所有模型/超参数/架构选择只在当前外层训练集内部完成；外层测试集不参与。
- 默认可同时运行两种开发性评价：
  1) pair-random：成对安全的随机组五折。每个测试折含多个 T 档，反映参数域内插值能力。
  2) t-blocked：每次整档留出一个 t_change。可直接观察内部 T 档插值和边界 T 档外推。
- 还可指定 --holdout-col/--holdout-value 做结构化留出。
- R² 始终采用评价集自身均值作为基线；同时报告 MAE 和 MAE/评价集平均 |Mz|。

数据用途
--------
用于模型选择的 50 组工况属于开发数据。独立确认采用模型冻结后新增的
off-grid CFD 工况，开发数据上的交叉评价与独立确认结果分别报告。

典型用法
--------
# 一次跑 pair-random + t-blocked，并在全部 50 组上锁定最终模型
python p3_overtaking_v5.py \
  --loads loads_all.npz \
  --extra-desc data/shared_overtaking_rich.h5 \
  --protocol all \
  --out-prefix results/overtaking_v5

# 只跑 t_change 整档留出五折
python p3_overtaking_v5.py \
  --loads loads_all.npz \
  --extra-desc data/shared_overtaking_rich.h5 \
  --protocol t-blocked \
  --out-prefix results/overtaking_v5_tblocked

# 留出 70% 充装率
python p3_overtaking_v5.py \
  --loads loads_all.npz \
  --extra-desc data/shared_overtaking_rich.h5 \
  --holdout-col fill_level --holdout-value 0.7 \
  --out-prefix results/overtaking_v5_fill07

# 快速冒烟测试（缩小候选网格，不用于最终论文数字）
python p3_overtaking_v5.py \
  --loads loads_all.npz \
  --extra-desc data/shared_overtaking_rich.h5 \
  --protocol t-blocked --fast \
  --out-prefix results/overtaking_v5_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object


@dataclass
class EvalResult:
    variant: str
    y_true: List[float]
    y_pred: List[float]
    chosen: List[str]
    inner_sse: List[float]


def _decode_strings(a: np.ndarray) -> List[str]:
    return [str(x.decode() if isinstance(x, (bytes, bytearray)) else x) for x in a]


def load_overtaking_data(loads_path: str, extra_desc: str):
    z = np.load(loads_path, allow_pickle=True)
    case_type = np.asarray(_decode_strings(z["case_type"]))
    idx = np.where(case_type == "overtaking")[0]
    if len(idx) == 0:
        raise SystemExit(f"❌ {loads_path} 中没有 case_type=overtaking")

    channels = _decode_strings(z["channels"])
    if "Mz" not in channels:
        raise SystemExit(f"❌ 通道中没有 Mz；实际为 {channels}")
    j_mz = channels.index("Mz")

    case_ids_all = _decode_strings(z["case_id"])
    case_ids = [case_ids_all[i] for i in idx]

    signed = np.asarray(z["raw_peak_signed"], dtype=float)[idx, j_mz] / 1e3
    if "raw_peak_abs" in z.files:
        y_abs = np.asarray(z["raw_peak_abs"], dtype=float)[idx, j_mz] / 1e3
    else:
        y_abs = np.abs(signed)
        print("⚠ loads 文件没有 raw_peak_abs，已退化为 abs(raw_peak_signed)")

    import h5py
    with h5py.File(extra_desc, "r") as f:
        Xall = np.asarray(f["X"][()], dtype=float)
        boundaries = np.asarray(f["case_boundaries"][()], dtype=int)
        feat_names = json.loads(f["metadata"].attrs["feature_names"])
        if "input_cols" in f["metadata"].attrs:
            names = json.loads(f["metadata"].attrs["input_cols"])
        else:
            names = [c for c in feat_names if c != "time"]
        cols = [feat_names.index(c) for c in names]
        X = np.asarray([Xall[s, cols] for s, _ in boundaries], dtype=float)

    if len(X) != len(idx):
        raise SystemExit(
            f"❌ extra-desc 工况数 {len(X)} 与 loads 中 overtaking 工况数 {len(idx)} 不一致"
        )

    required = ["fill_level", "peak_gx", "total_time", "t_change", "is_double_overtake"]
    miss = [c for c in required if c not in names]
    if miss:
        raise SystemExit(f"❌ extra-desc 缺少 {miss}；实际列为 {names}")

    return X, names, y_abs, signed, case_ids


def col(X: np.ndarray, names: Sequence[str], name: str) -> np.ndarray:
    if name not in names:
        raise SystemExit(f"❌ 描述符缺少 {name}；实际列为 {list(names)}")
    return X[:, names.index(name)]


def make_pair_groups(X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    fill = col(X, names, "fill_level")
    tc = col(X, names, "t_change")
    keys = [f"{f:.10f}|{t:.10f}" for f, t in zip(fill, tc)]
    _, groups = np.unique(keys, return_inverse=True)
    return groups


def wrap(reg) -> object:
    # X 和 y 都只在当前训练子集内标准化。
    p = Pipeline([
        ("xscale", StandardScaler()),
        ("reg", reg),
    ])
    return TransformedTargetRegressor(regressor=p, transformer=StandardScaler())


def make_candidates(seed: int, fast: bool = False) -> List[Candidate]:
    """v5 重点扩展低 gamma SVR；fast 仅用于检查脚本能否跑通。"""
    out: List[Candidate] = []

    ridge_alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    for a in ridge_alphas:
        out.append(Candidate(f"Ridge(a={a:g})", wrap(Ridge(alpha=a))))

    if fast:
        krr_a = (0.01, 0.1)
        krr_g = (0.05, 0.1)
        svr_C = (30.0, 100.0)
        svr_g = (0.03, 0.05, 0.1)
        svr_e = (0.03, 0.05)
    else:
        krr_a = (0.001, 0.01, 0.1, 1.0)
        krr_g = (0.02, 0.03, 0.05, 0.08, 0.10, 0.20)
        svr_C = (10.0, 30.0, 100.0, 300.0, 1000.0)
        svr_g = (0.02, 0.03, 0.05, 0.08, 0.10, 0.20)
        svr_e = (0.01, 0.03, 0.05, 0.10)

    for a in krr_a:
        for g in krr_g:
            out.append(Candidate(
                f"KRR(a={a:g},g={g:g})",
                wrap(KernelRidge(kernel="rbf", alpha=a, gamma=g)),
            ))

    for C in svr_C:
        for g in svr_g:
            for e in svr_e:
                out.append(Candidate(
                    f"SVR(C={C:g},g={g:g},e={e:g})",
                    wrap(SVR(kernel="rbf", C=C, gamma=g, epsilon=e)),
                ))
    return out


def random_group_folds(groups: np.ndarray, n_splits: int, seed: int):
    ug = np.unique(groups)
    k = min(n_splits, len(ug))
    if k < 2:
        raise ValueError("至少需要 2 个独立 group")
    rng = np.random.default_rng(seed)
    shuffled = ug.copy()
    rng.shuffle(shuffled)
    chunks = np.array_split(shuffled, k)
    all_idx = np.arange(len(groups))
    splits = []
    for chunk in chunks:
        mask = np.isin(groups, chunk)
        te = all_idx[mask]
        tr = all_idx[~mask]
        splits.append((tr, te))
    return splits


def t_blocked_folds(X: np.ndarray, names: Sequence[str]):
    tc = col(X, names, "t_change")
    vals = np.sort(np.unique(tc))
    all_idx = np.arange(len(tc))
    return [
        (all_idx[~np.isclose(tc, v)], all_idx[np.isclose(tc, v)], float(v))
        for v in vals
    ]


def inner_splits(
    X: np.ndarray,
    names: Sequence[str],
    groups: np.ndarray,
    protocol: str,
    n_splits: int,
    seed: int,
):
    if protocol == "t-blocked":
        tc = col(X, names, "t_change")
        vals = np.sort(np.unique(tc))
        if len(vals) >= 2 and len(vals) <= n_splits:
            all_idx = np.arange(len(tc))
            return [
                (all_idx[~np.isclose(tc, v)], all_idx[np.isclose(tc, v)])
                for v in vals
            ]
        # 当 T 档多于 inner-folds 时，按 T 档随机分块，仍不会拆开同一 T 档。
        _, tgroups = np.unique(tc, return_inverse=True)
        return random_group_folds(tgroups, n_splits, seed)
    if protocol == "pair-random":
        return random_group_folds(groups, n_splits, seed)
    raise ValueError(protocol)


def select_candidate(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    candidates: Sequence[Candidate],
):
    best: Optional[Candidate] = None
    best_sse = float("inf")
    scored: List[Tuple[float, Candidate]] = []
    for cand in candidates:
        sse = 0.0
        valid = True
        for tr, va in splits:
            if len(tr) < 3 or len(va) == 0:
                valid = False
                break
            est = clone(cand.estimator)
            est.fit(X[tr], y[tr])
            pred = np.asarray(est.predict(X[va]), dtype=float).reshape(-1)
            sse += float(np.sum((pred - y[va]) ** 2))
        if not valid:
            continue
        scored.append((sse, cand))
        if sse < best_sse:
            best_sse = sse
            best = cand
    if best is None:
        raise RuntimeError("没有候选模型得到有效 inner-CV 结果")
    scored.sort(key=lambda x: x[0])
    return best, best_sse, scored


def evaluate_metrics(y: np.ndarray, pred: np.ndarray):
    r2 = float(r2_score(y, pred)) if np.var(y) > 1e-30 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    mean_abs = max(float(np.mean(np.abs(y))), 1e-30)
    rel = mae / mean_abs
    return r2, mae, rel, mean_abs


def pair_table(
    X: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
):
    """每个 pair group 必须正好含 1 个 single + 1 个 double。"""
    d = col(X, names, "is_double_overtake") >= 0.5
    fill = col(X, names, "fill_level")
    tc = col(X, names, "t_change")

    base_cols = ["fill_level", "peak_gx", "t_change"]
    delta_cols = ["fill_level", "peak_gx", "total_time", "t_change"]
    ibase = [names.index(c) for c in base_cols]
    idelta = [names.index(c) for c in delta_cols]

    rows = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        si = idx[~d[idx]]
        di = idx[d[idx]]
        if len(si) != 1 or len(di) != 1:
            raise RuntimeError(
                f"pair group={g} 不是 1 single + 1 double；single={len(si)}, double={len(di)}"
            )
        s, dd = int(si[0]), int(di[0])
        rows.append((g, s, dd))

    pg = np.asarray([r[0] for r in rows], dtype=int)
    sidx = np.asarray([r[1] for r in rows], dtype=int)
    didx = np.asarray([r[2] for r in rows], dtype=int)
    Xbase = X[sidx][:, ibase]
    Xdelta = X[didx][:, idelta]
    ys = y[sidx]
    yd = y[didx]
    delta = yd - ys
    pt = tc[sidx]
    pf = fill[sidx]
    return {
        "pair_groups": pg,
        "single_idx": sidx,
        "double_idx": didx,
        "Xbase": Xbase,
        "Xdelta": Xdelta,
        "y_single": ys,
        "y_double": yd,
        "delta": delta,
        "t_change": pt,
        "fill_level": pf,
        "base_cols": base_cols,
        "delta_cols": delta_cols,
    }


def pair_inner_splits(ptab, protocol: str, n_splits: int, seed: int):
    n = len(ptab["pair_groups"])
    if protocol == "t-blocked":
        tc = np.asarray(ptab["t_change"], dtype=float)
        vals = np.sort(np.unique(tc))
        if len(vals) >= 2 and len(vals) <= n_splits:
            all_idx = np.arange(n)
            return [
                (all_idx[~np.isclose(tc, v)], all_idx[np.isclose(tc, v)])
                for v in vals
            ]
        _, tg = np.unique(tc, return_inverse=True)
        return random_group_folds(tg, n_splits, seed)
    return random_group_folds(np.arange(n), n_splits, seed)


def select_pair_delta(
    X: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Sequence[Candidate],
    protocol: str,
    inner_folds_n: int,
    seed: int,
    top_k: int,
):
    p = pair_table(X, names, y, groups)
    splits = pair_inner_splits(p, protocol, inner_folds_n, seed)

    base_best, _, base_scores = select_candidate(
        p["Xbase"], p["y_single"], splits, candidates
    )
    delta_best, _, delta_scores = select_candidate(
        p["Xdelta"], p["delta"], splits, candidates
    )

    base_top = [c for _, c in base_scores[: min(top_k, len(base_scores))]]
    delta_top = [c for _, c in delta_scores[: min(top_k, len(delta_scores))]]

    best_combo = None
    best_sse = float("inf")
    for bc in base_top:
        for dc in delta_top:
            sse = 0.0
            for tr, va in splits:
                eb = clone(bc.estimator)
                ed = clone(dc.estimator)
                eb.fit(p["Xbase"][tr], p["y_single"][tr])
                ed.fit(p["Xdelta"][tr], p["delta"][tr])
                pb = np.asarray(eb.predict(p["Xbase"][va]), dtype=float).reshape(-1)
                pd = np.asarray(ed.predict(p["Xdelta"][va]), dtype=float).reshape(-1)
                # 对每个验证 pair 同时评价 single 与 double。
                sse += float(np.sum((pb - p["y_single"][va]) ** 2))
                sse += float(np.sum((pb + pd - p["y_double"][va]) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_combo = (bc, dc)

    if best_combo is None:
        raise RuntimeError("PAIR-DELTA 无可用组合")
    return best_combo[0], best_combo[1], best_sse, p


def fit_predict_pair_delta(
    Xtr: np.ndarray,
    names: Sequence[str],
    ytr: np.ndarray,
    gtr: np.ndarray,
    Xte: np.ndarray,
    candidates: Sequence[Candidate],
    protocol: str,
    inner_folds_n: int,
    seed: int,
    top_k: int,
):
    bc, dc, inner_sse, p = select_pair_delta(
        Xtr, names, ytr, gtr, candidates, protocol, inner_folds_n, seed, top_k
    )
    eb = clone(bc.estimator)
    ed = clone(dc.estimator)
    eb.fit(p["Xbase"], p["y_single"])
    ed.fit(p["Xdelta"], p["delta"])

    base_cols = p["base_cols"]
    delta_cols = p["delta_cols"]
    ibase = [names.index(c) for c in base_cols]
    idelta = [names.index(c) for c in delta_cols]
    dte = col(Xte, names, "is_double_overtake") >= 0.5

    pred = np.asarray(eb.predict(Xte[:, ibase]), dtype=float).reshape(-1)
    if np.any(dte):
        corr = np.asarray(ed.predict(Xte[dte][:, idelta]), dtype=float).reshape(-1)
        pred[dte] += corr
    label = f"base:{bc.name} | delta:{dc.name}"
    return pred, label, inner_sse


def fit_predict_raw(
    Xtr: np.ndarray,
    names: Sequence[str],
    ytr: np.ndarray,
    gtr: np.ndarray,
    Xte: np.ndarray,
    candidates: Sequence[Candidate],
    select_protocol: str,
    inner_folds_n: int,
    seed: int,
):
    splits = inner_splits(
        Xtr, names, gtr, select_protocol, inner_folds_n, seed
    )
    cand, inner_sse, _ = select_candidate(Xtr, ytr, splits, candidates)
    est = clone(cand.estimator)
    est.fit(Xtr, ytr)
    pred = np.asarray(est.predict(Xte), dtype=float).reshape(-1)
    return pred, cand.name, inner_sse


def common_label(labels: Sequence[str]) -> str:
    if not labels:
        return "-"
    c = Counter(labels)
    lab, n = c.most_common(1)[0]
    return lab if n == len(labels) else f"最常见({n}/{len(labels)}): {lab}"


def evaluate_outer(
    protocol_name: str,
    outer_splits,
    X: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    case_ids: Sequence[str],
    candidates: Sequence[Candidate],
    args,
    out_prefix: str,
):
    variants = ["RAW-POOLED", "PAIR-DELTA", "AUTO-LOCKED"]
    results = {v: EvalResult(v, [], [], [], []) for v in variants}
    rows = []

    print("\n" + "=" * 96)
    print(f"外层评价协议：{protocol_name}")
    print("=" * 96)

    for k, item in enumerate(outer_splits, 1):
        if len(item) == 3:
            tr, te, tag = item
        else:
            tr, te = item
            tag = ""
        print(f"\n--- 外层 {k}/{len(outer_splits)}：训练 {len(tr)}，测试 {len(te)}" +
              (f"，测试档={tag:g}" if isinstance(tag, (int, float)) and tag != "" else "") + " ---")

        raw_pred, raw_lab, raw_sse = fit_predict_raw(
            X[tr], names, y[tr], groups[tr], X[te], candidates,
            args.select_protocol, args.inner_folds, args.seed + 101 * k,
        )
        pair_pred, pair_lab, pair_sse = fit_predict_pair_delta(
            X[tr], names, y[tr], groups[tr], X[te], candidates,
            args.select_protocol, args.inner_folds, args.seed + 211 * k,
            args.pair_top_k,
        )
        print(f"RAW-POOLED       inner-SSE={raw_sse:10.4f}  锁定={raw_lab}")
        print(f"PAIR-DELTA       inner-SSE={pair_sse:10.4f}  锁定={pair_lab}")

        if raw_sse <= pair_sse:
            auto_pred, auto_arch, auto_lab, auto_sse = raw_pred, "RAW-POOLED", raw_lab, raw_sse
        else:
            auto_pred, auto_arch, auto_lab, auto_sse = pair_pred, "PAIR-DELTA", pair_lab, pair_sse
        print(f"AUTO-LOCKED      -> {auto_arch}")

        fold_data = {
            "RAW-POOLED": (raw_pred, raw_lab, raw_sse),
            "PAIR-DELTA": (pair_pred, pair_lab, pair_sse),
            "AUTO-LOCKED": (auto_pred, f"{auto_arch} :: {auto_lab}", auto_sse),
        }
        for v, (pred, lab, sse) in fold_data.items():
            results[v].y_true.extend(y[te].tolist())
            results[v].y_pred.extend(pred.tolist())
            results[v].chosen.append(lab)
            results[v].inner_sse.append(sse)

        fill = col(X, names, "fill_level")
        tc = col(X, names, "t_change")
        dbl = col(X, names, "is_double_overtake")
        for j, ci in enumerate(te):
            rows.append({
                "protocol": protocol_name,
                "fold": k,
                "test_tag": tag,
                "case_id": case_ids[ci],
                "fill_level": float(fill[ci]),
                "t_change": float(tc[ci]),
                "is_double": int(dbl[ci] >= 0.5),
                "Mz_abs_true_kNm": float(y[ci]),
                "RSI_O_true": float(y[ci] / args.mref),
                "RAW_pred_kNm": float(raw_pred[j]),
                "PAIR_DELTA_pred_kNm": float(pair_pred[j]),
                "AUTO_variant": auto_arch,
                "AUTO_pred_kNm": float(auto_pred[j]),
                "AUTO_RSI_O_pred": float(auto_pred[j] / args.mref),
            })

    print("\n最终汇总：")
    summary = []
    for v in variants:
        yt = np.asarray(results[v].y_true)
        yp = np.asarray(results[v].y_pred)
        r2, mae, rel, mean_abs = evaluate_metrics(yt, yp)
        print(
            f"{v:<16} R²={r2:8.4f}  MAE={mae:7.3f} kN·m  "
            f"MAE/量级={rel:7.1%}  RSI_O MAE={mae/args.mref:.5f}"
        )
        print(f"{'':<16} 锁定配置：{common_label(results[v].chosen)}")
        summary.append({
            "protocol": protocol_name,
            "variant": v,
            "R2": f"{r2:.8f}",
            "MAE_kNm": f"{mae:.8f}",
            "MAE_over_mean_abs": f"{rel:.8f}",
            "RSI_O_MAE": f"{mae/args.mref:.8f}",
            "mean_abs_Mz_kNm": f"{mean_abs:.8f}",
            "locked_config": common_label(results[v].chosen),
        })

    # t-blocked 时额外打印 AUTO 每个 T 档，区分内部插值与边界外推。
    if protocol_name == "t-blocked":
        print("\nAUTO-LOCKED 分 T 档结果：")
        tc_all = col(X, names, "t_change")
        auto_rows = rows
        tvals = sorted({float(r["t_change"]) for r in auto_rows})
        tmin, tmax = min(tvals), max(tvals)
        for tv in tvals:
            rr = [r for r in auto_rows if np.isclose(float(r["t_change"]), tv)]
            yt = np.asarray([float(r["Mz_abs_true_kNm"]) for r in rr])
            yp = np.asarray([float(r["AUTO_pred_kNm"]) for r in rr])
            r2, mae, rel, _ = evaluate_metrics(yt, yp)
            kind = "边界外推" if np.isclose(tv, tmin) or np.isclose(tv, tmax) else "内部档位"
            print(f"  T={tv:g}s  {kind:<8} R²={r2:8.4f}  MAE={mae:6.3f}  MAE/量级={rel:6.1%}")

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    sum_path = out_prefix + "_summary.csv"
    case_path = out_prefix + "_cases.csv"
    with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with open(case_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n汇总 → {sum_path}")
    print(f"逐工况 → {case_path}")
    return summary, rows


def fit_final_model(
    X: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Sequence[Candidate],
    args,
    out_prefix: str,
):
    """在全部 50 组上仅用内部 CV 锁定并训练最终模型，供新增 CFD 工况的独立评价使用。"""
    raw_splits = inner_splits(
        X, names, groups, args.select_protocol, args.inner_folds, args.seed + 9001
    )
    raw_cand, raw_sse, _ = select_candidate(X, y, raw_splits, candidates)

    bc, dc, pair_sse, p = select_pair_delta(
        X, names, y, groups, candidates, args.select_protocol,
        args.inner_folds, args.seed + 9011, args.pair_top_k
    )

    if raw_sse <= pair_sse:
        arch = "RAW-POOLED"
        est = clone(raw_cand.estimator)
        est.fit(X, y)
        obj = {
            "version": "v5",
            "architecture": arch,
            "target": "max_abs_Mz_kNm",
            "mref_kNm": args.mref,
            "feature_names": list(names),
            "model_label": raw_cand.name,
            "model": est,
            "selection_inner_sse": raw_sse,
            "selection_protocol": args.select_protocol,
        }
        label = raw_cand.name
        sse = raw_sse
    else:
        eb = clone(bc.estimator); ed = clone(dc.estimator)
        eb.fit(p["Xbase"], p["y_single"])
        ed.fit(p["Xdelta"], p["delta"])
        obj = {
            "version": "v5",
            "architecture": "PAIR-DELTA",
            "target": "max_abs_Mz_kNm",
            "mref_kNm": args.mref,
            "feature_names": list(names),
            "base_feature_names": p["base_cols"],
            "delta_feature_names": p["delta_cols"],
            "base_model_label": bc.name,
            "delta_model_label": dc.name,
            "base_model": eb,
            "delta_model": ed,
            "selection_inner_sse": pair_sse,
            "selection_protocol": args.select_protocol,
        }
        arch = "PAIR-DELTA"
        label = f"base:{bc.name} | delta:{dc.name}"
        sse = pair_sse

    path = out_prefix + "_final_model.joblib"
    joblib.dump(obj, path)
    sel_path = out_prefix + "_final_selection.csv"
    with open(sel_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["architecture", "locked_config", "inner_sse", "selection_protocol"])
        w.writeheader(); w.writerow({
            "architecture": arch,
            "locked_config": label,
            "inner_sse": f"{sse:.8f}",
            "selection_protocol": args.select_protocol,
        })
    print("\n" + "=" * 96)
    print("最终模型冻结（供未来新增 off-grid CFD 独立验证）")
    print("=" * 96)
    print(f"架构={arch}")
    print(f"配置={label}")
    print(f"模型文件 → {path}")
    print(f"选择记录 → {sel_path}")
    print("独立评价使用此冻结版本；依据新增结果调整模型后，应另设独立确认数据。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", default="loads_all.npz")
    ap.add_argument("--extra-desc", default="data/shared_overtaking_rich.h5")
    ap.add_argument("--protocol", choices=["all", "pair-random", "t-blocked"], default="all")
    ap.add_argument("--select-protocol", choices=["pair-random", "t-blocked"], default="t-blocked",
                    help="每个外层训练集内部用于锁定模型/架构的协议")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--pair-top-k", type=int, default=5,
                    help="PAIR-DELTA 中基础/增量候选各取前 K，再做 K×K 联合比较")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mref", type=float, default=273.6)
    ap.add_argument("--holdout-col", default=None)
    ap.add_argument("--holdout-value", type=float, default=None)
    ap.add_argument("--fast", action="store_true", help="缩小候选网格，仅做冒烟测试")
    ap.add_argument("--no-fit-final", action="store_true", help="不在全部 50 组上冻结最终模型")
    ap.add_argument("--out-prefix", default="results/overtaking_v5")
    args = ap.parse_args()

    if (args.holdout_col is None) ^ (args.holdout_value is None):
        raise SystemExit("❌ --holdout-col 与 --holdout-value 必须同时给出")

    X, names, y, signed, case_ids = load_overtaking_data(args.loads, args.extra_desc)
    groups = make_pair_groups(X, names)
    d = col(X, names, "is_double_overtake") >= 0.5
    tc = col(X, names, "t_change")

    print("=" * 96)
    print("P3 超车/变道峰值代理 v5：RAW-POOLED vs PAIR-DELTA")
    print("目标 = max|Mz|；RSI_O = max|Mz| / Mref")
    print("=" * 96)
    print(f"工况数={len(y)}，single={np.sum(~d)}，double={np.sum(d)}，pair={len(np.unique(groups))}")
    print(f"T 档={np.sort(np.unique(tc))}")
    print(f"|Mz|max：均值={np.mean(y):.3f} kN·m，范围={np.min(y):.3f}~{np.max(y):.3f} kN·m")
    print(f"RSI_O 量级均值={np.mean(y)/args.mref:.5f}")
    print(f"raw_peak_signed：正={np.sum(signed>=0)} / 负={np.sum(signed<0)}；v5 不再用其符号作目标")

    candidates = make_candidates(args.seed, args.fast)
    print(f"候选配置={len(candidates)} 个" + ("（FAST 冒烟网格）" if args.fast else "（正式网格）"))
    print("重点：SVR 搜索 C=10~1000、gamma=0.02~0.20、epsilon=0.01~0.10（正式网格）。")
    print(f"外层训练集内部选模协议={args.select_protocol}；测试折不会参与超参数或架构选择。")

    if args.holdout_col is not None:
        if args.holdout_col not in names:
            raise SystemExit(f"❌ {args.holdout_col} 不在描述符 {names}")
        v = col(X, names, args.holdout_col)
        te = np.where(np.isclose(v, args.holdout_value))[0]
        tr = np.where(~np.isclose(v, args.holdout_value))[0]
        if len(te) == 0:
            raise SystemExit(f"❌ 没有 {args.holdout_col}={args.holdout_value}；实际值={np.unique(v)}")
        evaluate_outer(
            f"holdout:{args.holdout_col}={args.holdout_value}",
            [(tr, te)], X, names, y, groups, case_ids, candidates, args,
            args.out_prefix,
        )
    else:
        if args.protocol in ("all", "pair-random"):
            splits = random_group_folds(groups, args.outer_folds, args.seed + 17)
            evaluate_outer(
                "pair-random", splits, X, names, y, groups, case_ids, candidates,
                args, args.out_prefix + "_paircv"
            )
        if args.protocol in ("all", "t-blocked"):
            splits = t_blocked_folds(X, names)
            evaluate_outer(
                "t-blocked", splits, X, names, y, groups, case_ids, candidates,
                args, args.out_prefix + "_tblocked"
            )

    if not args.no_fit_final and args.holdout_col is None:
        fit_final_model(X, names, y, groups, candidates, args, args.out_prefix)

    print("\n评价协议说明：")
    print("1) pair-random AUTO-LOCKED：按单/双配对分组，评价参数域内预测。")
    print("2) t-blocked：T=2.0/2.5/3.0 为内部档位留出；T=1.5/4.0 为边界档位外推评价。")
    print("3) RAW-POOLED 与 PAIR-DELTA 的架构选择依据训练集内部 inner-SSE。")
    print("4) R²、MAE 与目标量级联合报告，并注明相应数据划分。")
    print("5) 独立确认：固定 *_final_model.joblib 与预测记录后，使用新增 off-grid CFD 评价。")


if __name__ == "__main__":
    main()
