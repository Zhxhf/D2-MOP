#!/usr/bin/env python3
import argparse, json
from collections import Counter, defaultdict
LETTERS='ABCDE'

def load(path):
    data=json.load(open(path,'r',encoding='utf-8'))
    if isinstance(data,dict) and 'data' in data: data=data['data']
    return data

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',default='output/article2_nextqa_full/00_llovi_direct_-1.json')
    args=ap.parse_args()
    d=load(args.baseline)
    pred=Counter(); truth=Counter(); corr=defaultdict(lambda:[0,0]); conf=Counter(); qtype=defaultdict(lambda:[0,0])
    for x in d.values():
        t=int(x.get('truth',-1)); p=int(x.get('pred',-1)); pred[p]+=1; truth[t]+=1; corr[p][0]+=1; corr[p][1]+= int(p==t); conf[(t,p)]+=1
        qt=x.get('q_type',''); qtype[qt][0]+=1; qtype[qt][1]+=int(p==t)
    print('Total:',len(d))
    print('Prediction distribution:')
    for i in [-1,0,1,2,3,4]:
        if pred[i]:
            n,c=corr[i]
            name='INVALID' if i<0 else LETTERS[i]
            print(f'  {name}: pred={n}, correct={c}, acc={c/n:.4f}')
    print('\nTruth distribution:')
    for i in range(5): print(f'  {LETTERS[i]}: {truth[i]}')
    print('\nQuestion type baseline acc:')
    for k,(n,c) in sorted(qtype.items()): print(f'  {k}: {c}/{n} = {c/n:.4f}')
    print('\nConfusion matrix rows=truth, cols=pred A-E:')
    for t in range(5): print(LETTERS[t], [conf[(t,p)] for p in range(5)])
if __name__=='__main__': main()
