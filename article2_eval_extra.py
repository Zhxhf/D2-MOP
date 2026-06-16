
import argparse
from util import load_json, save_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_path', required=True)
    p.add_argument('--out_path', default='')
    args = p.parse_args()
    data = load_json(args.pred_path)
    if 'data' in data: data = data['data']
    n_claim = n_unsup = 0
    n = correct = 0
    temporal = ['before','after','again','repeat','return','leave','enter','stay','then']
    t_n = t_correct = 0
    for k,v in data.items():
        if 'pred' in v and 'truth' in v:
            n += 1
            if v['pred'] == v['truth'] or str(v['pred']) == str(v['truth']): correct += 1
            q = str(v.get('question','')).lower()
            if any(w in q for w in temporal):
                t_n += 1
                if v['pred'] == v['truth'] or str(v['pred']) == str(v['truth']): t_correct += 1
        meta = v.get('repair_meta') or {}
        claims = meta.get('claims') or []
        unsup = meta.get('unsupported') or []
        n_claim += len(claims); n_unsup += len(unsup)
    stat = {
        'acc': correct/max(1,n), 'num_total': n,
        'temporal_acc': t_correct/max(1,t_n), 'num_temporal': t_n,
        'unsupported_claim_rate': n_unsup/max(1,n_claim), 'num_claims': n_claim, 'num_unsupported': n_unsup,
    }
    print(stat)
    if args.out_path: save_json(stat, args.out_path)
if __name__ == '__main__': main()
