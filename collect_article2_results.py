import argparse, json, csv
from pathlib import Path

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_md', required=True)
    args = ap.parse_args()
    rows=[]
    for p in sorted(Path(args.input_dir).glob('*_extra.json')):
        if p.name.startswith('summary'):
            continue
        d=load_json(p)
        name=p.name.replace('_extra.json','')
        rows.append({
            'experiment': name,
            'acc_percent': round(float(d.get('acc',0))*100, 2),
            'num_total': d.get('num_total',0),
            'temporal_acc_percent': round(float(d.get('temporal_acc',0))*100, 2),
            'num_temporal': d.get('num_temporal',0),
            'unsupported_claim_rate_percent': round(float(d.get('unsupported_claim_rate',0))*100, 2),
            'num_claims': d.get('num_claims',0),
            'num_unsupported': d.get('num_unsupported',0),
        })
    fieldnames=['experiment','acc_percent','num_total','temporal_acc_percent','num_temporal','unsupported_claim_rate_percent','num_claims','num_unsupported']
    with open(args.out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    with open(args.out_md,'w') as f:
        f.write('| Experiment | Acc (%) | N | Temporal Acc (%) | Temporal N | Unsupported Claim Rate (%) | Claims | Unsupported |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['experiment']} | {r['acc_percent']} | {r['num_total']} | {r['temporal_acc_percent']} | {r['num_temporal']} | {r['unsupported_claim_rate_percent']} | {r['num_claims']} | {r['num_unsupported']} |\n")
    print(f'Wrote {args.out_csv} and {args.out_md}')

if __name__ == '__main__':
    main()
