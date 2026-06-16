"""MOP-VQA: Multi-Oracle Prompt Ensemble for caption-only long-video QA.

This v4 script is meant for the situation where conservative evidence-gating gives
only tiny gains. It creates stronger independent candidate predictions by asking the
same local LLM several *different* evidence-oriented prompts, then keeps the strong
LLoVi baseline unless multiple prompts agree on a different answer.

No video files are needed: it uses the existing LLoVi captions stored in the dataset
items (item['narration']).
"""
from __future__ import annotations
import argparse, json, os, re
from collections import Counter
from typing import Dict, Any, List
from tqdm import tqdm

from dataset import get_dataset
from eval import eval_qa_nextqa, eval_qa_egoschema
from model import get_model
from prompts import identity
from util import load_json, save_json, makedir

LETTERS = "ABCDE"


def parse_args():
    p = argparse.ArgumentParser("MOP-VQA prompt ensemble")
    p.add_argument("--dataset", default="nextqa", choices=["nextqa", "intentqa", "nextgqa", "egoschema", "videomme"])
    p.add_argument("--data_path", default="data/nextqa/llava1.5_fps1.json")
    p.add_argument("--anno_path", default="data/nextqa/val.csv")
    p.add_argument("--duration_path", default="data/nextqa/durations.json")
    p.add_argument("--fps", default=0.5, type=float)
    p.add_argument("--caption_every", default=2, type=int)
    p.add_argument("--num_examples_to_run", default=500, type=int)
    p.add_argument("--output_base_path", required=True)
    p.add_argument("--output_filename", default="mop_nextqa.json")
    p.add_argument("--start_from_scratch", action="store_true")
    p.add_argument("--save_every", default=10, type=int)
    p.add_argument("--disable_eval", action="store_true")

    p.add_argument("--baseline_pred_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api_key", default="")
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--max_new_tokens", default=192, type=int)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    p.add_argument("--prompt_modes", default="direct,verify,eliminate,temporal", help="Comma separated modes: direct,verify,eliminate,temporal,contrastive")
    p.add_argument("--max_narration_chars", default=12000, type=int)
    p.add_argument("--baseline_weight", default=1, type=int)
    p.add_argument("--min_override_votes", default=2, type=int)
    p.add_argument("--min_vote_margin", default=1, type=int, help="non-baseline votes must exceed baseline votes by this margin after baseline_weight")
    p.add_argument("--risk_baseline_letters", default="", help="If non-empty, only allow overrides when baseline letter is in this set, e.g. E")
    p.add_argument("--max_calls", default=-1, type=int)
    return p.parse_args()


def load_pred(path):
    d = load_json(path)
    if isinstance(d, dict) and "data" in d:
        d = d["data"]
    if not isinstance(d, dict):
        raise ValueError(f"Expected dict prediction file: {path}")
    return d


def norm_truth(x):
    try:
        return int(x)
    except Exception:
        s = str(x).strip().upper()
        return LETTERS.index(s[0]) if s and s[0] in LETTERS else -1


def parse_answer(text: str) -> int:
    s = str(text or "").strip()
    up = s.upper()
    # Prefer explicit final answers.
    pats = [
        r"FINAL\s*ANSWER\s*[:：]\s*([A-E])\b",
        r"ANSWER\s*[:：]\s*([A-E])\b",
        r"THE\s+ANSWER\s+IS\s*([A-E])\b",
        r"OPTION\s*([A-E])\b",
    ]
    for pat in pats:
        m = re.search(pat, up)
        if m:
            return LETTERS.index(m.group(1))
    # Then use the last standalone letter, because explanations may mention A-E earlier.
    ms = list(re.finditer(r"\b([A-E])\b", up))
    if ms:
        return LETTERS.index(ms[-1].group(1))
    return -1


def options_block(item: Dict[str, Any]) -> str:
    lines=[]
    for i, l in enumerate(LETTERS):
        v = item.get(f"option{l}", item.get("options", [""]*5)[i] if isinstance(item.get("options"), list) and len(item.get("options"))>i else "")
        lines.append(f"{l}. {v}")
    return "\n".join(lines)


def trim_narration(narr: str, max_chars: int) -> str:
    narr = str(narr or "")
    if len(narr) <= max_chars:
        return narr
    # Keep beginning + end; many long-video questions ask before/after/end.
    half = max_chars // 2
    return narr[:half] + "\n... [middle captions truncated] ...\n" + narr[-half:]


def build_prompt(mode: str, item: Dict[str, Any], max_chars: int) -> str:
    q = str(item.get("question", "")).strip().rstrip("?") + "?"
    narr = trim_narration(item.get("narration", ""), max_chars)
    opts = options_block(item)
    common = f"Video captions in temporal order:\n{narr}\n\nQuestion: {q}\nChoices:\n{opts}\n"
    mode = mode.strip().lower()
    if mode == "direct":
        return common + "\nChoose the best answer using the video captions. Return exactly one line: ANSWER: <A/B/C/D/E>."
    if mode == "verify":
        return common + "\nFor each choice, silently check whether the captions SUPPORT it, CONTRADICT it, or do not mention it. Prefer direct caption evidence over commonsense. Then return exactly one line: ANSWER: <A/B/C/D/E>."
    if mode == "eliminate":
        return common + "\nEliminate choices contradicted or unsupported by the captions. Do not guess from choice plausibility. Select the remaining choice best supported by explicit captions. Return exactly one line: ANSWER: <A/B/C/D/E>."
    if mode == "temporal":
        return common + "\nPay special attention to temporal order: before, after, then, finally, beginning, end, repeat, and changes over time. Select the choice consistent with the caption sequence. Return exactly one line: ANSWER: <A/B/C/D/E>."
    if mode == "contrastive":
        return common + "\nCompare the two or three most plausible choices against the exact caption evidence. Choose the one with the strongest direct evidence and weakest contradiction. Return exactly one line: ANSWER: <A/B/C/D/E>."
    return common + "\nReturn exactly one line: ANSWER: <A/B/C/D/E>."


def decide_from_votes(prompt_preds: List[int], baseline_pred: int, args) -> Dict[str, Any]:
    cnt = Counter(p for p in prompt_preds if 0 <= p < 5)
    base_letter = LETTERS[baseline_pred] if 0 <= baseline_pred < 5 else "INVALID"
    if 0 <= baseline_pred < 5 and args.baseline_weight > 0:
        cnt[baseline_pred] += args.baseline_weight
    if not cnt:
        return {"pred": baseline_pred, "changed": False, "reason": "no_valid_prompt_pred", "vote_counts": {}}
    top_pred, top_votes = cnt.most_common(1)[0]
    base_votes = cnt.get(baseline_pred, 0)
    risk_ok = (not args.risk_baseline_letters) or (base_letter in set(str(args.risk_baseline_letters).upper()))
    can_override = (
        top_pred != baseline_pred and
        risk_ok and
        top_votes >= args.min_override_votes and
        (top_votes - base_votes) >= args.min_vote_margin
    )
    pred = top_pred if can_override else baseline_pred
    return {
        "pred": int(pred),
        "changed": bool(pred != baseline_pred),
        "reason": "mop_prompt_majority_override" if pred != baseline_pred else "keep_baseline_vote_gate",
        "top_pred": int(top_pred),
        "top_votes": int(top_votes),
        "baseline_votes_after_weight": int(base_votes),
        "vote_counts": {LETTERS[k]: int(v) for k, v in cnt.items() if 0 <= k < 5},
        "risk_gate_ok": bool(risk_ok),
    }


def launch():
    args = parse_args()
    makedir(args.output_base_path)
    out_path = os.path.join(args.output_base_path, args.output_filename)
    baseline = load_pred(args.baseline_pred_path)
    processed = {}
    if not args.start_from_scratch and os.path.exists(out_path):
        processed = load_json(out_path)
        if isinstance(processed, dict) and "data" in processed:
            processed = processed["data"]
    dataset = get_dataset(args, quids_to_exclude=set(processed.keys()), num_examples_to_run=args.num_examples_to_run)
    model = get_model(args)
    model.set_post_process_fn(identity)
    modes = [m.strip() for m in str(args.prompt_modes).split(',') if m.strip()]
    calls = 0
    pbar = tqdm(total=len(dataset))
    for i, item0 in enumerate(dataset):
        item = dict(item0)
        ukey = item[dataset.ukey]
        if ukey not in baseline:
            raise KeyError(f"Missing baseline pred for {ukey}")
        b = int(baseline[ukey].get("pred", -1))
        prompt_preds=[]; prompt_responses={}
        for mode in modes:
            if args.max_calls >= 0 and calls >= args.max_calls:
                break
            prompt = build_prompt(mode, item, args.max_narration_chars)
            head = "You are a careful long-video question answering assistant. Answer only from the given captions."
            _, info = model.forward(head, [prompt])
            calls += 1
            resp = info.get("response", "")
            pred = parse_answer(resp)
            prompt_preds.append(pred)
            prompt_responses[mode] = {"response": resp, "pred": pred, "letter": LETTERS[pred] if 0 <= pred < 5 else "INVALID"}
        dec = decide_from_votes(prompt_preds, b, args)
        pred = int(dec["pred"])
        item.update({
            "pred": pred,
            "truth": norm_truth(item.get("truth", -1)),
            "response": LETTERS[pred] if 0 <= pred < 5 else "INVALID",
            "baseline_pred": b,
            "baseline_response": baseline[ukey].get("response", ""),
            "baseline_mode": "file",
            "mop_prompt_modes": modes,
            "mop_prompt_responses": prompt_responses,
            "prompt_preds": prompt_preds,
            "seac_decision": dec,
        })
        processed[ukey]=item
        if i % args.save_every == 0:
            save_json(processed, out_path)
        pbar.update(1)
    save_json(processed, out_path)
    if not args.disable_eval:
        if args.dataset == "egoschema":
            out = eval_qa_egoschema(processed)
        elif args.dataset in {"nextqa","intentqa","nextgqa"}:
            out = eval_qa_nextqa(args.anno_path, processed)
        else:
            out = {"data": processed}
        save_json(out, out_path)

if __name__ == "__main__":
    launch()
