#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/videomind/VideoMind/LANZHOUhuiyi/D2p/LLoVi_article2_trustqa_4090

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate uav

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODEL:-/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2.5-7B-Instruct}
N=${N:--1}

DATA=data/intentqa/llava1.5_fps1.json
ANNO=data/intentqa/val.csv
DUR=data/intentqa/durations.json

BASE_OUT=output/intentqa_llovi_direct
MOP_OUT=output/intentqa_mop_v4_full
ANALYSIS_OUT=output/intentqa_mop_v4_analysis
LOG_OUT=output/intentqa_mop_v4_logs

mkdir -p "$BASE_OUT" "$MOP_OUT" "$ANALYSIS_OUT" "$LOG_OUT"

echo "===== check IntentQA data ====="
ls -lh "$DATA" "$ANNO" "$DUR"

echo "===== run IntentQA LLoVi-direct baseline ====="
python main.py \
  --dataset intentqa \
  --data_path "$DATA" \
  --anno_path "$ANNO" \
  --duration_path "$DUR" \
  --fps 0.5 \
  --caption_every 2 \
  --prompt_type qa_next \
  --task qa \
  --model "$MODEL" \
  --load_in_4bit \
  --num_examples_to_run "$N" \
  --output_base_path "$BASE_OUT" \
  --output_filename llovi_direct_intentqa.json \
  --start_from_scratch \
  --save_every 20

echo "===== run IntentQA MOP-VQA v4 ====="
python article2_prompt_ensemble_vqa.py \
  --dataset intentqa \
  --data_path "$DATA" \
  --anno_path "$ANNO" \
  --duration_path "$DUR" \
  --caption_every 2 \
  --num_examples_to_run "$N" \
  --model "$MODEL" \
  --load_in_4bit \
  --baseline_pred_path "$BASE_OUT/llovi_direct_intentqa.json" \
  --prompt_modes direct,verify,eliminate,temporal,contrastive \
  --baseline_weight 1 \
  --min_override_votes 2 \
  --min_vote_margin 1 \
  --max_new_tokens 128 \
  --output_base_path "$MOP_OUT" \
  --output_filename mop_intentqa.json \
  --start_from_scratch \
  --save_every 10

echo "===== compare IntentQA baseline vs MOP ====="
python eval_seac_compare.py \
  --pred_path "$MOP_OUT/mop_intentqa.json" \
  --out_dir "$MOP_OUT"

cat "$MOP_OUT/mop_intentqa_seac_compare_summary.json"

echo "===== generate IntentQA analysis files ====="
python - <<'PY'
import json, csv, os
from collections import defaultdict, Counter

pred_path = "output/intentqa_mop_v4_full/mop_intentqa.json"
out_dir = "output/intentqa_mop_v4_analysis"
os.makedirs(out_dir, exist_ok=True)

LET = "ABCDE"

def load_data(path):
    obj = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj

data = load_data(pred_path)

def get_int(x, key, default=-1):
    try:
        return int(x.get(key, default))
    except Exception:
        s = str(x.get(key, "")).strip().upper()
        return LET.index(s[0]) if s and s[0] in LET else default

def letter(i):
    return LET[i] if isinstance(i, int) and 0 <= i < len(LET) else str(i)

def get_type(x):
    for k in ["type", "q_type", "qtype", "question_type", "category"]:
        if k in x and str(x[k]).strip():
            return str(x[k]).strip()
    return "UNK"

def get_options(x):
    opts = []
    for k in ["a0", "a1", "a2", "a3", "a4"]:
        if k in x:
            opts.append(x.get(k, ""))
    if not opts and isinstance(x.get("options"), list):
        opts = x["options"]
    return opts

def options_json(x):
    return json.dumps(get_options(x), ensure_ascii=False)

# W2R / R2W cases
for mode, filename in [("w2r", "w2r_cases.csv"), ("r2w", "r2w_cases.csv")]:
    rows = []
    for qid, x in data.items():
        y = get_int(x, "truth")
        b = get_int(x, "baseline_pred")
        p = get_int(x, "pred")

        cond = (b != y and p == y) if mode == "w2r" else (b == y and p != y)
        if not cond:
            continue

        rows.append({
            "qid": qid,
            "video": x.get("video", x.get("video_id", "")),
            "type": get_type(x),
            "question": x.get("question", ""),
            "truth": letter(y),
            "baseline": letter(b),
            "mop": letter(p),
            "options": options_json(x),
            "prompt_preds": json.dumps(x.get("prompt_preds", []), ensure_ascii=False),
            "decision": json.dumps(x.get("seac_decision", x.get("mop_info", {})), ensure_ascii=False)[:2000],
        })

    out_csv = os.path.join(out_dir, filename)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["qid", "video", "type", "question", "truth", "baseline", "mop", "options", "prompt_preds", "decision"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(mode.upper(), len(rows), "saved:", out_csv)

def summarize(pred_getter):
    total = base_correct = mop_correct = changed = w2r = r2w = 0
    by_type = defaultdict(lambda: {"n":0, "base":0, "mop":0, "w2r":0, "r2w":0})

    for qid, x in data.items():
        y = get_int(x, "truth")
        b = get_int(x, "baseline_pred")
        p = pred_getter(x, b)
        t = get_type(x)

        total += 1
        base_correct += int(b == y)
        mop_correct += int(p == y)
        changed += int(p != b)
        w2r += int(b != y and p == y)
        r2w += int(b == y and p != y)

        by_type[t]["n"] += 1
        by_type[t]["base"] += int(b == y)
        by_type[t]["mop"] += int(p == y)
        by_type[t]["w2r"] += int(b != y and p == y)
        by_type[t]["r2w"] += int(b == y and p != y)

    return {
        "total": total,
        "baseline_correct": base_correct,
        "mop_correct": mop_correct,
        "baseline_acc": base_correct / total if total else 0,
        "mop_acc": mop_correct / total if total else 0,
        "gain": (mop_correct - base_correct) / total if total else 0,
        "changed": changed,
        "W_to_R": w2r,
        "R_to_W": r2w,
        "net": w2r - r2w,
        "by_type": by_type,
    }

official = summarize(lambda x, b: get_int(x, "pred"))

with open(os.path.join(out_dir, "intentqa_mop_summary.json"), "w", encoding="utf-8") as f:
    json.dump({k:v for k,v in official.items() if k != "by_type"}, f, ensure_ascii=False, indent=2)

print("===== official summary =====")
print(json.dumps({k:v for k,v in official.items() if k != "by_type"}, ensure_ascii=False, indent=2))

# by type
qtype_csv = os.path.join(out_dir, "intentqa_by_qtype.csv")
with open(qtype_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["type", "n", "baseline_acc", "mop_acc", "gain", "W2R", "R2W", "net"])
    for t, s in sorted(official["by_type"].items()):
        n = s["n"]
        ba = s["base"] / n * 100 if n else 0
        ma = s["mop"] / n * 100 if n else 0
        w.writerow([t, n, f"{ba:.2f}", f"{ma:.2f}", f"{ma-ba:+.2f}", s["w2r"], s["r2w"], s["w2r"] - s["r2w"]])
print("saved:", qtype_csv)

# prompt ablation
def get_prompt_pred(x, mode):
    d = x.get("mop_prompt_responses", {})
    if isinstance(d, dict) and mode in d:
        try:
            return int(d[mode].get("pred", -1))
        except Exception:
            pass

    modes = x.get("mop_prompt_modes", [])
    preds = x.get("prompt_preds", [])
    if mode in modes:
        idx = modes.index(mode)
        if idx < len(preds):
            try:
                return int(preds[idx])
            except Exception:
                return -1
    return -1

def vote_pred(x, b, modes, baseline_weight=1, min_override_votes=2, min_vote_margin=1):
    cnt = Counter()
    for m in modes:
        p = get_prompt_pred(x, m)
        if 0 <= p < 5:
            cnt[p] += 1

    if baseline_weight > 0 and 0 <= b < 5:
        cnt[b] += baseline_weight

    if not cnt:
        return b

    top, tv = cnt.most_common(1)[0]
    bv = cnt.get(b, 0)

    if top != b and tv >= min_override_votes and (tv - bv) >= min_vote_margin:
        return top
    return b

def direct_only(x, b):
    p = get_prompt_pred(x, "direct")
    return p if 0 <= p < 5 else b

ablations = [
    ("baseline", lambda x,b: b),
    ("direct_prompt_only", direct_only),
    ("mop_3_direct_verify_eliminate", lambda x,b: vote_pred(x,b,["direct","verify","eliminate"],1,2,1)),
    ("mop_4_add_temporal", lambda x,b: vote_pred(x,b,["direct","verify","eliminate","temporal"],1,2,1)),
    ("mop_5_full_gate", lambda x,b: vote_pred(x,b,["direct","verify","eliminate","temporal","contrastive"],1,2,1)),
    ("mop_5_no_baseline_weight", lambda x,b: vote_pred(x,b,["direct","verify","eliminate","temporal","contrastive"],0,2,0)),
]

abl_csv = os.path.join(out_dir, "intentqa_prompt_ablation.csv")
with open(abl_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["setting", "total", "baseline_acc", "method_acc", "gain", "changed", "W2R", "R2W", "net"])
    for name, fn in ablations:
        s = summarize(fn)
        w.writerow([name, s["total"], f"{s['baseline_acc']*100:.2f}", f"{s['mop_acc']*100:.2f}", f"{s['gain']*100:+.2f}", s["changed"], s["W_to_R"], s["R_to_W"], s["net"]])
print("saved:", abl_csv)

# option distribution
dist = defaultdict(Counter)
for qid, x in data.items():
    y = get_int(x, "truth")
    b = get_int(x, "baseline_pred")
    p = get_int(x, "pred")
    dist["truth"][letter(y)] += 1
    dist["baseline"][letter(b)] += 1
    dist["mop"][letter(p)] += 1

opt_csv = os.path.join(out_dir, "intentqa_option_distribution.csv")
with open(opt_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "A", "B", "C", "D", "E"])
    for src in ["truth", "baseline", "mop"]:
        w.writerow([src] + [dist[src][l] for l in LET])
print("saved:", opt_csv)
PY

echo "===== all done ====="
echo "Main summary:"
cat "$MOP_OUT/mop_intentqa_seac_compare_summary.json"

echo
echo "Analysis files:"
ls -lh "$ANALYSIS_OUT"
