import json
import csv
import os
from collections import Counter, defaultdict

PRED_PATH = "output/mop_nextqa_full/mop_nextqa.json"
OUT_DIR = "output/mop_nextqa_full/case_analysis_d_vs_f"
os.makedirs(OUT_DIR, exist_ok=True)

LET = "ABCDE"

def load_json(path):
    obj = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj

def get_int(x, key, default=-1):
    try:
        return int(x.get(key, default))
    except Exception:
        s = str(x.get(key, "")).strip().upper()
        if s and s[0] in LET:
            return LET.index(s[0])
        return default

def letter(i):
    return LET[i] if isinstance(i, int) and 0 <= i < len(LET) else str(i)

def get_type(x):
    for k in ["type", "q_type", "qtype", "question_type", "category"]:
        if k in x and str(x[k]).strip():
            return str(x[k]).strip()
    return "UNK"

def get_options(x):
    # 兼容不同格式
    if isinstance(x.get("options"), list):
        return x["options"]
    opts = []
    for k in ["a0", "a1", "a2", "a3", "a4"]:
        if k in x:
            opts.append(x.get(k, ""))
    for k in ["optionA", "optionB", "optionC", "optionD", "optionE"]:
        if k in x:
            opts.append(x.get(k, ""))
    return opts

def get_prompt_pred(x, mode):
    # 兼容 v4 输出格式 1: mop_prompt_responses
    d = x.get("mop_prompt_responses", {})
    if isinstance(d, dict) and mode in d:
        try:
            return int(d[mode].get("pred", -1))
        except Exception:
            pass

    # 兼容 v4 输出格式 2: modes + prompt_preds
    modes = x.get("mop_prompt_modes", [])
    preds = x.get("prompt_preds", [])
    if isinstance(modes, list) and isinstance(preds, list) and mode in modes:
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
        return b, cnt

    top, tv = cnt.most_common(1)[0]
    bv = cnt.get(b, 0)

    if top != b and tv >= min_override_votes and (tv - bv) >= min_vote_margin:
        return top, cnt

    return b, cnt

def make_row(qid, x, b, d, f, y, vote_counter):
    opts = get_options(x)
    prompt_preds = {
        "direct": letter(get_prompt_pred(x, "direct")),
        "verify": letter(get_prompt_pred(x, "verify")),
        "eliminate": letter(get_prompt_pred(x, "eliminate")),
        "temporal": letter(get_prompt_pred(x, "temporal")),
        "contrastive": letter(get_prompt_pred(x, "contrastive")),
    }
    return {
        "qid": qid,
        "video": x.get("video", x.get("video_id", "")),
        "type": get_type(x),
        "question": x.get("question", ""),
        "truth": letter(y),
        "baseline": letter(b),
        "mop_d_direct": letter(d),
        "mop_f_full": letter(f),
        "baseline_correct": int(b == y),
        "direct_correct": int(d == y),
        "full_correct": int(f == y),
        "options": json.dumps(opts, ensure_ascii=False),
        "prompt_preds": json.dumps(prompt_preds, ensure_ascii=False),
        "vote_counter": json.dumps({letter(k): v for k, v in vote_counter.items()}, ensure_ascii=False),
    }

data = load_json(PRED_PATH)

summary = {
    "total": 0,
    "baseline_correct": 0,
    "direct_correct": 0,
    "full_correct": 0,

    "direct_changed": 0,
    "direct_W_to_R": 0,
    "direct_R_to_W": 0,

    "full_changed": 0,
    "full_W_to_R": 0,
    "full_R_to_W": 0,

    "D_right_F_wrong": 0,
    "F_right_D_wrong": 0,
    "D_and_F_both_right": 0,
    "D_and_F_both_wrong": 0,
    "D_eq_F": 0,
    "D_ne_F": 0,
}

direct_w2r = []
direct_r2w = []
d_right_f_wrong = []
f_right_d_wrong = []
both_right = []
both_wrong = []
d_f_diff_all = []

for qid, x in data.items():
    y = get_int(x, "truth")
    b = get_int(x, "baseline_pred")
    d = get_prompt_pred(x, "direct")

    # 如果 direct 无法解析，则回退 baseline
    if d < 0:
        d = b

    # MOP-F 就是当前 full 输出 pred
    f = get_int(x, "pred")

    # 如果 pred 不存在，也按 five-prompt gate 重算
    if f < 0:
        f, vote_counter = vote_pred(
            x, b,
            ["direct", "verify", "eliminate", "temporal", "contrastive"],
            baseline_weight=1,
            min_override_votes=2,
            min_vote_margin=1
        )
    else:
        _, vote_counter = vote_pred(
            x, b,
            ["direct", "verify", "eliminate", "temporal", "contrastive"],
            baseline_weight=1,
            min_override_votes=2,
            min_vote_margin=1
        )

    row = make_row(qid, x, b, d, f, y, vote_counter)

    summary["total"] += 1
    summary["baseline_correct"] += int(b == y)
    summary["direct_correct"] += int(d == y)
    summary["full_correct"] += int(f == y)

    summary["direct_changed"] += int(d != b)
    summary["direct_W_to_R"] += int(b != y and d == y)
    summary["direct_R_to_W"] += int(b == y and d != y)

    summary["full_changed"] += int(f != b)
    summary["full_W_to_R"] += int(b != y and f == y)
    summary["full_R_to_W"] += int(b == y and f != y)

    if d == f:
        summary["D_eq_F"] += 1
    else:
        summary["D_ne_F"] += 1
        d_f_diff_all.append(row)

    # Direct-only W2R / R2W
    if b != y and d == y:
        direct_w2r.append(row)
    if b == y and d != y:
        direct_r2w.append(row)

    # D vs F 对比
    if d == y and f != y:
        summary["D_right_F_wrong"] += 1
        d_right_f_wrong.append(row)

    if f == y and d != y:
        summary["F_right_D_wrong"] += 1
        f_right_d_wrong.append(row)

    if d == y and f == y:
        summary["D_and_F_both_right"] += 1
        both_right.append(row)

    if d != y and f != y:
        summary["D_and_F_both_wrong"] += 1
        both_wrong.append(row)

def write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    fieldnames = [
        "qid", "video", "type", "question",
        "truth", "baseline", "mop_d_direct", "mop_f_full",
        "baseline_correct", "direct_correct", "full_correct",
        "options", "prompt_preds", "vote_counter"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path} ({len(rows)} rows)")

write_csv("direct_only_w2r_cases.csv", direct_w2r)
write_csv("direct_only_r2w_cases.csv", direct_r2w)
write_csv("D_right_F_wrong_cases.csv", d_right_f_wrong)
write_csv("F_right_D_wrong_cases.csv", f_right_d_wrong)
write_csv("D_F_different_all_cases.csv", d_f_diff_all)
write_csv("D_and_F_both_right_cases.csv", both_right[:200])
write_csv("D_and_F_both_wrong_cases.csv", both_wrong[:200])

# 汇总指标
total = summary["total"]
summary["baseline_acc"] = summary["baseline_correct"] / total if total else 0
summary["direct_acc"] = summary["direct_correct"] / total if total else 0
summary["full_acc"] = summary["full_correct"] / total if total else 0
summary["direct_gain"] = summary["direct_acc"] - summary["baseline_acc"]
summary["full_gain"] = summary["full_acc"] - summary["baseline_acc"]
summary["direct_net"] = summary["direct_W_to_R"] - summary["direct_R_to_W"]
summary["full_net"] = summary["full_W_to_R"] - summary["full_R_to_W"]
summary["direct_override_precision"] = summary["direct_W_to_R"] / summary["direct_changed"] if summary["direct_changed"] else 0
summary["full_override_precision"] = summary["full_W_to_R"] / summary["full_changed"] if summary["full_changed"] else 0

summary_path = os.path.join(OUT_DIR, "D_vs_F_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n===== SUMMARY =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nsaved summary:", summary_path)

# 生成一个可直接复制进论文的 markdown 表
md_path = os.path.join(OUT_DIR, "D_vs_F_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# MOP-VQA-D vs MOP-VQA-F Summary\n\n")
    f.write("| Method | Acc | Gain | Changed | W→R | R→W | Net | Override Precision |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write(f"| Baseline | {summary['baseline_acc']*100:.2f} | +0.00 | 0 | 0 | 0 | 0 | - |\n")
    f.write(f"| MOP-VQA-D | {summary['direct_acc']*100:.2f} | {summary['direct_gain']*100:+.2f} | {summary['direct_changed']} | {summary['direct_W_to_R']} | {summary['direct_R_to_W']} | {summary['direct_net']} | {summary['direct_override_precision']*100:.2f} |\n")
    f.write(f"| MOP-VQA-F | {summary['full_acc']*100:.2f} | {summary['full_gain']*100:+.2f} | {summary['full_changed']} | {summary['full_W_to_R']} | {summary['full_R_to_W']} | {summary['full_net']} | {summary['full_override_precision']*100:.2f} |\n\n")

    f.write("## D vs F Difference\n\n")
    f.write(f"- D right, F wrong: {summary['D_right_F_wrong']}\n")
    f.write(f"- F right, D wrong: {summary['F_right_D_wrong']}\n")
    f.write(f"- D = F: {summary['D_eq_F']}\n")
    f.write(f"- D != F: {summary['D_ne_F']}\n")

print("saved markdown:", md_path)
