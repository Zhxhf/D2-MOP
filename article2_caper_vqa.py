"""CAPER-VQA: Candidate-Pool Evidence Reranking for long video QA.

This script is a safer v3 replacement for the overly aggressive multi-choice
reranker in article2_seac_vqa.py.

Key idea:
- Keep the strong LLoVi-direct baseline by default.
- Do NOT let the LLM freely re-rank all five options for every suspicious sample.
- Only build a small candidate pool from independent alternative runs
  (Choice-blind, Choice-blind+Logic, TrustQA, optional SEAC outputs).
- Ask the LLM to override the baseline only when a non-baseline candidate has
  explicit caption evidence and the baseline lacks/contradicts evidence.

This is designed to target the large oracle gap between LLoVi and existing
auxiliary runs while avoiding the R->W explosion seen in v2 aggressive mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from dataset import get_dataset
from eval import eval_qa_nextqa, eval_qa_egoschema
from model import get_model
from prompts import identity
from util import load_json, makedir, save_json
from seac_utils import (
    LETTERS,
    build_option_ledgers,
    format_ledger_for_prompt,
    normalize_truth,
    output_summary_row,
)


def parse_args():
    p = argparse.ArgumentParser("CAPER-VQA: candidate-pool evidence reranking")
    p.add_argument("--dataset", default="nextqa", choices=["nextqa", "intentqa", "nextgqa", "egoschema", "videomme"])
    p.add_argument("--data_path", default="data/nextqa/llava1.5_fps1.json")
    p.add_argument("--anno_path", default="data/nextqa/val.csv")
    p.add_argument("--duration_path", default="data/nextqa/durations.json")
    p.add_argument("--fps", default=0.5, type=float)
    p.add_argument("--caption_every", default=2, type=int)
    p.add_argument("--num_examples_to_run", default=500, type=int)

    p.add_argument("--output_base_path", required=True)
    p.add_argument("--output_filename", default="caper_nextqa.json")
    p.add_argument("--start_from_scratch", action="store_true")
    p.add_argument("--save_every", default=20, type=int)
    p.add_argument("--disable_eval", action="store_true")

    p.add_argument("--baseline_pred_path", required=True)
    p.add_argument("--candidate_pred_paths", default="", help="Comma-separated auxiliary prediction JSON files.")
    p.add_argument("--candidate_names", default="", help="Optional comma-separated names matching candidate_pred_paths.")
    p.add_argument("--include_rule_best", action="store_true", help="Also add SEAC rule-best as a candidate source.")

    p.add_argument("--model", required=True, help="Local HF LLM path/name, or 'rule' for dry-run keep-baseline mode.")
    p.add_argument("--api_key", default="")
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--max_new_tokens", default=192, type=int)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # Evidence and candidate-pool controls.
    p.add_argument("--retrieval_top_k", default=10, type=int)
    p.add_argument("--evidence_window", default=2, type=int)
    p.add_argument("--retrieval_alpha", default=0.35, type=float)
    p.add_argument("--min_candidate_votes", default=2, type=int, help="Non-baseline candidate must be proposed by at least this many auxiliary sources.")
    p.add_argument("--risk_baseline_letters", default="E", help="Only trigger for these baseline letters unless --scope all_disagreement. Use '' to disable.")
    p.add_argument("--scope", default="risk_or_consensus", choices=["risk_or_consensus", "risk_only", "consensus_only", "all_disagreement"],
                   help="Which samples to send to the LLM reranker.")
    p.add_argument("--max_candidates", default=3, type=int, help="Max non-baseline candidates in the LLM prompt.")
    p.add_argument("--max_calls", default=-1, type=int)

    # Guardrails. Default is intentionally strict.
    p.add_argument("--require_high", action="store_true", default=True)
    p.add_argument("--allow_medium", action="store_true")
    p.add_argument("--min_support", default=0.22, type=float)
    p.add_argument("--max_missing", default=0.78, type=float)
    p.add_argument("--max_contradiction", default=0.70, type=float)
    p.add_argument("--min_margin_vs_base", default=-0.15, type=float,
                   help="Candidate calibrated score can be slightly lower than baseline because auxiliary methods provide extra signal.")
    p.add_argument("--keep_on_uncertain", action="store_true", default=True)
    return p.parse_args()


def load_pred_file(path: str) -> Dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, dict):
        raise ValueError(f"Prediction file is not a dict: {path}")
    return data


def parse_paths(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def candidate_prompt(item: Dict[str, Any], baseline_pred: int, ledgers, cand_indices: List[int], sources: Dict[int, List[str]]) -> str:
    base = ledgers[baseline_pred]
    blocks = ["BASELINE OPTION (keep this unless another option has clearly stronger video evidence):", format_ledger_for_prompt(base, max_evidence=5)]
    blocks.append("\nALTERNATIVE CANDIDATES proposed by independent auxiliary runs:")
    for idx in cand_indices:
        src = ", ".join(sources.get(idx, []))
        blocks.append(f"\nCandidate source votes: {src}")
        blocks.append(format_ledger_for_prompt(ledgers[idx], max_evidence=5))

    opts = []
    for lg in ledgers:
        if str(lg.choice).strip():
            opts.append(f"{lg.letter}. {lg.choice}")

    return f"""You are a VERY CONSERVATIVE video-evidence judge.
Your task is not to answer from commonsense. Your task is to decide whether the BASELINE answer should be replaced.

Rules:
1. Use ONLY the caption evidence shown under each option.
2. KEEP the baseline unless an alternative has clear, direct caption support.
3. Do NOT change the baseline for vague, implicit, or commonsense-only reasons.
4. Do NOT choose an alternative just because it sounds plausible.
5. If evidence is mixed or uncertain, answer KEEP.

Question: {item.get('question','').rstrip('?')}?
All choices:
{chr(10).join(opts)}

{chr(10).join(blocks)}

Return exactly this format:
ANSWER: KEEP or one letter from A/B/C/D/E
CONFIDENCE: HIGH / MEDIUM / LOW
REASON: one short sentence citing caption evidence.
"""


def parse_caper_response(text: str) -> Dict[str, str]:
    s = str(text or "").strip()
    up = s.upper()
    ans = "KEEP"
    m = re.search(r"ANSWER\s*[:：]\s*(KEEP|[A-E])\b", up)
    if m:
        ans = m.group(1)
    else:
        first = re.search(r"\b(KEEP|[A-E])\b", up)
        if first:
            ans = first.group(1)
    conf = "LOW"
    m = re.search(r"CONFIDENCE\s*[:：]\s*(HIGH|MEDIUM|LOW)\b", up)
    if m:
        conf = m.group(1)
    reason = ""
    m = re.search(r"REASON\s*[:：]\s*(.*)", s, re.I | re.S)
    if m:
        reason = m.group(1).strip()[:500]
    return {"answer": ans, "confidence": conf, "reason": reason, "raw": s}


def make_candidate_pool(ukey: str, baseline_pred: int, aux_data: List[Tuple[str, Dict[str, Any]]], min_votes: int) -> Tuple[List[int], Dict[int, List[str]], Counter]:
    votes = Counter()
    sources: Dict[int, List[str]] = defaultdict(list)
    for name, data in aux_data:
        if ukey not in data:
            continue
        try:
            p = int(data[ukey].get("pred", -1))
        except Exception:
            continue
        if 0 <= p < 5 and p != baseline_pred:
            votes[p] += 1
            sources[p].append(name)
    cands = [p for p, v in votes.most_common() if v >= min_votes and p != baseline_pred]
    return cands, sources, votes


def should_trigger(baseline_pred: int, cand_indices: List[int], votes: Counter, args) -> Tuple[bool, str]:
    if not cand_indices:
        return False, "no_candidate"
    base_letter = LETTERS[baseline_pred] if 0 <= baseline_pred < len(LETTERS) else "INVALID"
    risk = bool(args.risk_baseline_letters and base_letter in set(str(args.risk_baseline_letters).upper()))
    consensus = bool(cand_indices)
    if args.scope == "all_disagreement":
        return True, "all_disagreement"
    if args.scope == "risk_only":
        return (risk, f"risk_{base_letter}" if risk else "not_risk")
    if args.scope == "consensus_only":
        return (consensus, "consensus" if consensus else "no_consensus")
    if args.scope == "risk_or_consensus":
        return (risk or consensus, (f"risk_{base_letter}+consensus" if risk and consensus else f"risk_{base_letter}" if risk else "consensus"))
    return False, "unknown_scope"


def launch():
    args = parse_args()
    makedir(args.output_base_path)
    out_path = os.path.join(args.output_base_path, args.output_filename)

    processed: Dict[str, Any] = {}
    if not args.start_from_scratch and os.path.exists(out_path):
        processed = load_json(out_path)
        if isinstance(processed, dict) and "data" in processed:
            processed = processed["data"]

    baseline_data = load_pred_file(args.baseline_pred_path)
    cand_paths = parse_paths(args.candidate_pred_paths)
    cand_names = parse_paths(args.candidate_names)
    if cand_names and len(cand_names) != len(cand_paths):
        raise ValueError("candidate_names length must match candidate_pred_paths")
    if not cand_names:
        cand_names = [f"cand{i+1}" for i in range(len(cand_paths))]
    aux_data = [(n, load_pred_file(p)) for n, p in zip(cand_names, cand_paths)]

    dataset = get_dataset(args, quids_to_exclude=set(processed.keys()), num_examples_to_run=args.num_examples_to_run)
    model = None if str(args.model).lower() == "rule" else get_model(args)
    if model is not None:
        model.set_post_process_fn(identity)

    calls = 0
    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        item = dict(item)
        ukey = item[dataset.ukey]
        if ukey not in baseline_data:
            raise KeyError(f"Missing baseline prediction for {ukey}")
        baseline_pred = int(baseline_data[ukey].get("pred", -1))
        pred = baseline_pred
        ledgers = build_option_ledgers(item, top_k=args.retrieval_top_k, window=args.evidence_window, alpha=args.retrieval_alpha)

        cand_indices, sources, votes = make_candidate_pool(ukey, baseline_pred, aux_data, args.min_candidate_votes)
        # Sort candidates by vote count then evidence score; keep top max_candidates.
        cand_indices = sorted(cand_indices, key=lambda x: (votes[x], ledgers[x].calibrated_score), reverse=True)[: args.max_candidates]
        trig, trig_reason = should_trigger(baseline_pred, cand_indices, votes, args)

        decision = {
            "final_pred": baseline_pred,
            "baseline_pred": baseline_pred,
            "changed": False,
            "reason": "keep_baseline_no_trigger" if not trig else "triggered_but_kept",
            "trigger_reason": trig_reason,
            "candidate_indices": cand_indices,
            "candidate_votes": {LETTERS[k]: int(v) for k, v in votes.items() if 0 <= k < len(LETTERS)},
            "candidate_sources": {LETTERS[k]: v for k, v in sources.items() if 0 <= k < len(LETTERS)},
        }
        verifier_meta = {"used": False, "response": "", "answer": "KEEP", "confidence": "LOW", "reason": ""}

        if trig and cand_indices and (args.max_calls < 0 or calls < args.max_calls):
            if model is None:
                # dry run: keep baseline but still output candidate pool
                verifier_meta["reason"] = "rule_dry_run_keep"
            else:
                prompt = candidate_prompt(item, baseline_pred, ledgers, cand_indices, sources)
                head = "You are a conservative candidate-pool evidence verifier for video QA."
                _, info = model.forward(head, [prompt])
                calls += 1
                parsed = parse_caper_response(info.get("response", ""))
                verifier_meta.update({"used": True, **parsed})
                ans = parsed["answer"]
                allowed_conf = {"HIGH"} | ({"MEDIUM"} if args.allow_medium else set())
                if ans in LETTERS and parsed["confidence"] in allowed_conf:
                    cand = LETTERS.index(ans)
                    if cand in cand_indices and 0 <= baseline_pred < len(ledgers):
                        lg = ledgers[cand]
                        base_lg = ledgers[baseline_pred]
                        margin = float(lg.calibrated_score - base_lg.calibrated_score)
                        guard_ok = (
                            lg.support >= args.min_support and
                            lg.missing <= args.max_missing and
                            lg.contradiction <= args.max_contradiction and
                            margin >= args.min_margin_vs_base
                        )
                        if guard_ok:
                            pred = cand
                            decision.update({
                                "final_pred": pred,
                                "changed": pred != baseline_pred,
                                "reason": "caper_high_conf_candidate_override" if pred != baseline_pred else "caper_keep_baseline",
                                "margin_vs_base": round(margin, 4),
                                "guard_ok": True,
                            })
                        else:
                            decision.update({"reason": "llm_candidate_failed_guardrail", "guard_ok": False, "attempted": ans})
                elif ans == "KEEP":
                    decision["reason"] = "llm_keep_baseline"
                else:
                    decision["reason"] = "llm_low_conf_or_invalid_keep"

        item.update({
            "pred": int(pred),
            "truth": normalize_truth(item.get("truth", -1)),
            "response": LETTERS[pred] if 0 <= pred < len(LETTERS) else "INVALID",
            "baseline_pred": baseline_pred,
            "baseline_response": baseline_data[ukey].get("response", ""),
            "caper_decision": decision,
            "caper_verifier": verifier_meta,
            "option_ledgers": [x.as_dict() for x in ledgers],
            "seac_summary": output_summary_row({**item, "pred": pred, "baseline_pred": baseline_pred}),
        })
        processed[ukey] = item
        if i % args.save_every == 0:
            save_json(processed, out_path)
        pbar.update(1)

    save_json(processed, out_path)
    if not args.disable_eval:
        if args.dataset == "egoschema":
            out = eval_qa_egoschema(processed)
        elif args.dataset in {"nextqa", "intentqa", "nextgqa"}:
            out = eval_qa_nextqa(args.anno_path, processed)
        else:
            out = {"data": processed, "note": "raw output saved"}
        save_json(out, out_path)


if __name__ == "__main__":
    launch()
