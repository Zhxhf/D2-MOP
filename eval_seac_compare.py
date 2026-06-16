"""Evaluate SEAC-VQA against its preserved baseline.

This script reads an output JSON produced by article2_seac_vqa.py and reports:
- SEAC final accuracy
- baseline accuracy stored in the same JSON
- W->R / R->W counts caused by SEAC selective arbitration
- changed count and override precision
- optional NExT-QA type breakdown is already handled by eval.py during main run.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_data(obj):
    if isinstance(obj, dict) and 'data' in obj:
        return obj['data']
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_path', required=True)
    ap.add_argument('--out_dir', default='')
    args = ap.parse_args()

    obj = load_json(args.pred_path)
    data = get_data(obj)
    if not isinstance(data, dict):
        raise ValueError('Expected dict output or eval dict with a data field.')

    total = 0
    base_correct = 0
    seac_correct = 0
    changed = 0
    w2r = 0
    r2w = 0
    changed_correct = 0
    reasons = Counter()
    baseline_modes = Counter()
    margins = []

    rows = []
    for qid, x in data.items():
        if 'truth' not in x or 'pred' not in x or 'baseline_pred' not in x:
            continue
        total += 1
        truth = int(x.get('truth', -1))
        pred = int(x.get('pred', -1))
        base = int(x.get('baseline_pred', -1))
        bc = base == truth
        sc = pred == truth
        base_correct += int(bc)
        seac_correct += int(sc)
        ch = pred != base
        changed += int(ch)
        if ch and (not bc) and sc:
            w2r += 1
        if ch and bc and (not sc):
            r2w += 1
        if ch and sc:
            changed_correct += 1
        dec = x.get('seac_decision', {}) or x.get('caper_decision', {}) or {}
        reasons[dec.get('reason', 'unknown')] += 1
        baseline_modes[x.get('baseline_mode', 'unknown')] += 1
        if dec.get('margin_vs_baseline') is not None:
            margins.append(float(dec.get('margin_vs_baseline')))
        rows.append({
            'qid': qid,
            'truth': truth,
            'baseline_pred': base,
            'pred': pred,
            'baseline_correct': bc,
            'seac_correct': sc,
            'changed': ch,
            'reason': dec.get('reason', ''),
            'margin_vs_baseline': dec.get('margin_vs_baseline', ''),
        })

    summary = {
        'total': total,
        'baseline_correct': base_correct,
        'seac_correct': seac_correct,
        'baseline_acc': base_correct / total if total else 0,
        'seac_acc': seac_correct / total if total else 0,
        'gain': (seac_correct - base_correct) / total if total else 0,
        'changed': changed,
        'W_to_R': w2r,
        'R_to_W': r2w,
        'changed_correct': changed_correct,
        'override_precision': changed_correct / changed if changed else 0,
        'reason_counts': dict(reasons),
        'baseline_mode_counts': dict(baseline_modes),
        'avg_margin_vs_baseline': sum(margins) / len(margins) if margins else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pred_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.pred_path).stem
    with open(out_dir / f'{stem}_seac_compare_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / f'{stem}_seac_compare_rows.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['qid','truth','baseline_pred','pred','baseline_correct','seac_correct','changed','reason','margin_vs_baseline']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f'Wrote {out_dir / (stem + "_seac_compare_summary.json")}')
    print(f'Wrote {out_dir / (stem + "_seac_compare_rows.csv")}')


if __name__ == '__main__':
    main()
