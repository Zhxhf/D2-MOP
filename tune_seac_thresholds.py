"""Tune/apply SEAC selective-arbitration thresholds on a calibration split.

Use this ONLY on a development/calibration subset, not on the final test split.
It reads an article2_seac_vqa.py output file containing option_ledgers and baseline_pred.

Typical workflow:
1) Run SEAC once with any thresholds to save option_ledgers.
2) Tune thresholds on a small calibration subset.
3) Apply the chosen thresholds to a separate evaluation subset/output.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(x, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(x, f, indent=2, ensure_ascii=False)


def get_data(obj):
    return obj.get('data', obj) if isinstance(obj, dict) else obj


def decide(x, margin, min_support, max_missing, max_contradiction):
    ledgers = x.get('option_ledgers', [])
    base = int(x.get('baseline_pred', -1))
    if not ledgers:
        return int(x.get('pred', base)), 'no_ledgers'
    best = max(ledgers, key=lambda z: float(z.get('calibrated_score', -999)))
    alt = int(best.get('index', -1))
    if not (0 <= base < len(ledgers)):
        return alt, 'baseline_invalid'
    base_score = float(ledgers[base].get('calibrated_score', -999))
    m = float(best.get('calibrated_score', -999)) - base_score
    ok = (
        alt != base and
        m > margin and
        float(best.get('support', 0)) >= min_support and
        float(best.get('missing', 1)) <= max_missing and
        float(best.get('contradiction', 1)) <= max_contradiction
    )
    return (alt if ok else base), ('override' if ok else 'keep')


def evaluate(data, margin, min_support, max_missing, max_contradiction, keys=None):
    total = base_c = seac_c = changed = w2r = r2w = 0
    keys = keys or list(data.keys())
    for k in keys:
        x = data[k]
        if 'truth' not in x or 'baseline_pred' not in x:
            continue
        truth = int(x.get('truth', -1))
        base = int(x.get('baseline_pred', -1))
        pred, _ = decide(x, margin, min_support, max_missing, max_contradiction)
        bc, sc = (base == truth), (pred == truth)
        total += 1; base_c += bc; seac_c += sc
        if pred != base:
            changed += 1
            if (not bc) and sc: w2r += 1
            if bc and (not sc): r2w += 1
    return {
        'total': total,
        'baseline_correct': base_c,
        'seac_correct': seac_c,
        'baseline_acc': base_c / total if total else 0,
        'seac_acc': seac_c / total if total else 0,
        'gain': (seac_c - base_c) / total if total else 0,
        'changed': changed,
        'W_to_R': w2r,
        'R_to_W': r2w,
        'override_precision': (w2r / changed) if changed else 0,
        'margin_threshold': margin,
        'min_support': min_support,
        'max_missing': max_missing,
        'max_contradiction': max_contradiction,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_path', required=True)
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--apply_output', default='', help='Optional path to write a new JSON with tuned thresholds applied.')
    ap.add_argument('--calib_first_n', type=int, default=-1, help='Tune only on the first N samples; use -1 for all.')
    ap.add_argument('--margin_grid', default='0.10,0.12,0.16,0.20,0.25,0.30,0.40,0.50,0.65')
    ap.add_argument('--min_support_grid', default='0.10,0.12,0.16,0.20,0.25,0.30,0.35,0.40')
    ap.add_argument('--max_missing_grid', default='0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95')
    ap.add_argument('--max_contradiction_grid', default='0.30,0.45,0.55,0.70')
    args = ap.parse_args()

    obj = load_json(args.pred_path)
    data = get_data(obj)
    keys = list(data.keys())
    calib_keys = keys[:args.calib_first_n] if args.calib_first_n and args.calib_first_n > 0 else keys

    margins = [float(x) for x in args.margin_grid.split(',') if x.strip()]
    mins = [float(x) for x in args.min_support_grid.split(',') if x.strip()]
    maxmiss = [float(x) for x in args.max_missing_grid.split(',') if x.strip()]
    maxcons = [float(x) for x in args.max_contradiction_grid.split(',') if x.strip()]

    results = []
    for m, s, miss, con in itertools.product(margins, mins, maxmiss, maxcons):
        results.append(evaluate(data, m, s, miss, con, keys=calib_keys))
    results.sort(key=lambda r: (r['gain'], r['override_precision'], -r['R_to_W'], r['changed']), reverse=True)
    best = results[0] if results else {}

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pred_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({'best': best, 'top20': results[:20], 'note': 'Tune only on a development/calibration subset; do not tune on final test.'}, out_dir / 'seac_threshold_tuning.json')
    print(json.dumps({'best': best, 'top5': results[:5]}, indent=2, ensure_ascii=False))

    if args.apply_output:
        new_obj = copy.deepcopy(obj)
        new_data = get_data(new_obj)
        for k, x in new_data.items():
            pred, reason = decide(x, best['margin_threshold'], best['min_support'], best['max_missing'], best['max_contradiction'])
            x['pred'] = int(pred)
            x['response'] = ['A','B','C','D','E'][pred] if 0 <= pred < 5 else 'INVALID'
            x['seac_decision_tuned'] = {
                'reason': reason,
                'margin_threshold': best['margin_threshold'],
                'min_support': best['min_support'],
                'max_missing': best['max_missing'],
                'max_contradiction': best['max_contradiction'],
            }
        save_json(new_obj, args.apply_output)
        print(f'Wrote applied output: {args.apply_output}')


if __name__ == '__main__':
    main()
