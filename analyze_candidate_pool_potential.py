"""Analyze the potential of auxiliary candidate pools relative to a baseline.
This is a diagnostic only; oracle numbers must NOT be reported as final method results.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from util import load_json
LETTERS='ABCDE'

def loadp(p):
    d=load_json(p)
    return d.get('data', d) if isinstance(d,dict) else d

def parse(s):
    return [x.strip() for x in str(s or '').split(',') if x.strip()]

ap=argparse.ArgumentParser()
ap.add_argument('--baseline', required=True)
ap.add_argument('--candidates', required=True, help='comma-separated prediction files')
ap.add_argument('--min_votes', type=int, default=2)
ap.add_argument('--risk_letters', default='E')
args=ap.parse_args()
base=loadp(args.baseline); cands=[loadp(p) for p in parse(args.candidates)]
keys=list(base)
for min_votes in sorted(set([1,2,3,args.min_votes])):
  for scope in ['ALL', args.risk_letters or 'NONE']:
    total=len(keys); b_correct=0; oracle=0; triggers=0; w2r=0; danger=0
    for k in keys:
      y=int(base[k].get('truth',-1)); b=int(base[k].get('pred',-1)); b_correct += int(b==y)
      votes=Counter()
      for d in cands:
        if k in d:
          try: p=int(d[k].get('pred',-1))
          except: p=-1
          if 0<=p<5 and p!=b: votes[p]+=1
      pool={p for p,v in votes.items() if v>=min_votes}
      if scope!='ALL' and (not (0<=b<5) or LETTERS[b] not in set(scope)):
        pool=set()
      if pool: triggers+=1
      if any(p==y for p in pool):
        oracle += 1; w2r += int(b!=y)
      else:
        oracle += int(b==y); danger += int(pool and b==y)
    print({
      'min_votes':min_votes,'scope':scope,'triggers':triggers,'baseline_acc':round(b_correct/total,4),
      'oracle_acc_if_perfect_selector':round(oracle/total,4),'oracle_net_gain_samples':oracle-b_correct,
      'oracle_net_gain_pp':round((oracle-b_correct)/total*100,2),'potential_W_to_R':w2r,
      'baseline_right_triggered_without_correct_alt':danger})
