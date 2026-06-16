from __future__ import annotations
import argparse,json,os,itertools
from collections import Counter
from pathlib import Path
LETTERS='ABCDE'

def load(p):
 d=json.load(open(p)); return d.get('data',d) if isinstance(d,dict) else d

def eval_apply(data, baseline_weight=1, min_override_votes=2, min_vote_margin=1, risk_letters='', q_types=None, apply=False):
 out={}; total=basec=corr=ch=w=r=0
 risk=set(str(risk_letters).upper()) if risk_letters else None
 qset=set(q_types) if q_types else None
 for k,x in data.items():
  if 'truth' not in x or 'baseline_pred' not in x: continue
  total+=1; y=int(x['truth']); b=int(x['baseline_pred']); pred=b
  basec += b==y
  prompt_preds=[int(p) for p in x.get('prompt_preds',[]) if isinstance(p,int) or str(p).lstrip('-').isdigit()]
  cnt=Counter(p for p in prompt_preds if 0<=p<5)
  if 0<=b<5: cnt[b]+=baseline_weight
  if cnt:
   top,topv=cnt.most_common(1)[0]; bv=cnt.get(b,0)
   risk_ok=(risk is None) or (0<=b<5 and LETTERS[b] in risk)
   q_ok=(qset is None) or (x.get('q_type') in qset)
   if top!=b and risk_ok and q_ok and topv>=min_override_votes and (topv-bv)>=min_vote_margin:
    pred=top
  corr += pred==y
  if pred!=b:
   ch+=1; w+=(b!=y and pred==y); r+=(b==y and pred!=y)
  if apply:
   xx=dict(x); xx['pred']=int(pred); xx['response']=LETTERS[pred] if 0<=pred<5 else 'INVALID'
   xx['seac_decision']={'reason':'mop_tuned_vote_override' if pred!=b else 'keep_baseline_tuned_vote_gate','baseline_pred':b,'final_pred':pred,'changed':pred!=b}
   out[k]=xx
 return {'total':total,'baseline_correct':basec,'seac_correct':corr,'baseline_acc':basec/total,'seac_acc':corr/total,'gain':(corr-basec)/total,'changed':ch,'W_to_R':w,'R_to_W':r,'override_precision':(w/ch if ch else 0)}, out

ap=argparse.ArgumentParser()
ap.add_argument('--pred_path',required=True)
ap.add_argument('--out_dir',default='')
ap.add_argument('--apply_output',default='')
ap.add_argument('--search_qtypes',action='store_true')
args=ap.parse_args()
data=load(args.pred_path)
qtypes=sorted({x.get('q_type') for x in data.values() if isinstance(x,dict) and x.get('q_type')})
configs=[]
qsets=[None]
if args.search_qtypes:
 for r in range(1,len(qtypes)+1):
  for comb in itertools.combinations(qtypes,r): qsets.append(comb)
for bw in [0,1,2]:
 for mov in [1,2,3,4]:
  for margin in [0,1,2,3]:
   for risk in ['', 'E']:
    for qset in qsets:
     s,_=eval_apply(data,bw,mov,margin,risk,qset,False)
     s.update({'baseline_weight':bw,'min_override_votes':mov,'min_vote_margin':margin,'risk_letters':risk,'q_types':list(qset) if qset else []})
     configs.append(s)
configs=sorted(configs,key=lambda x:(x['seac_correct'], -x['changed']), reverse=True)
print(json.dumps({'best':configs[0],'top10':configs[:10]},indent=2,ensure_ascii=False))
out_dir=Path(args.out_dir or Path(args.pred_path).parent); out_dir.mkdir(parents=True,exist_ok=True)
with open(out_dir/'mop_vote_tuning_top10.json','w',encoding='utf-8') as f: json.dump({'top10':configs[:10]},f,indent=2,ensure_ascii=False)
if args.apply_output:
 best=configs[0]
 _,out=eval_apply(data,best['baseline_weight'],best['min_override_votes'],best['min_vote_margin'],best['risk_letters'],best['q_types'],True)
 Path(args.apply_output).parent.mkdir(parents=True,exist_ok=True)
 with open(args.apply_output,'w',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False)
 print('Wrote applied:',args.apply_output)
