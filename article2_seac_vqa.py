"""Article 2 SEAC-VQA runner.

SEAC-VQA: Selective Evidence Arbitration with Counterfactual calibration.

Main design goals:
1) low-storage: works on caption JSON + annotation CSV/JSON, no raw videos required;
2) baseline-preserving: use LLoVi direct baseline as default and only override when calibrated evidence is stronger;
3) dataset-flexible: reuses LLoVi dataset.py for NextQA/IntentQA/NextGQA/EgoSchema/VideoMME caption formats;
4) simple operation: rule-only mode for CPU/debug; optional LLM mode for stronger verification.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from dataset import get_dataset
from eval import eval_qa_nextqa, eval_qa_egoschema
from model import get_model
from prompts import first_char_as_answer, identity
from util import load_json, makedir, save_json
from seac_utils import (
    LETTERS,
    arbitration_decision,
    build_option_ledgers,
    format_ledger_for_prompt,
    get_choices,
    normalize_truth,
    output_summary_row,
    parse_pred_letter,
    select_best_ledger,
)


def parse_args():
    p = argparse.ArgumentParser("SEAC-VQA: Selective Evidence Arbitration with Counterfactual Calibration")

    # Data. Defaults match the uploaded D2p package.
    p.add_argument("--dataset", default="nextqa", choices=["nextqa", "intentqa", "nextgqa", "egoschema", "videomme"])
    p.add_argument("--data_path", default="data/nextqa/llava1.5_fps1.json")
    p.add_argument("--anno_path", default="data/nextqa/val.csv")
    p.add_argument("--duration_path", default="data/nextqa/durations.json")
    p.add_argument("--fps", default=0.5, type=float)
    p.add_argument("--caption_every", default=2, type=int)
    p.add_argument("--num_examples_to_run", default=50, type=int)

    # Output.
    p.add_argument("--output_base_path", required=True)
    p.add_argument("--output_filename", required=True)
    p.add_argument("--start_from_scratch", action="store_true")
    p.add_argument("--save_every", default=20, type=int)
    p.add_argument("--save_info", action="store_true")
    p.add_argument("--disable_eval", action="store_true")

    # Baseline.
    p.add_argument("--baseline_pred_path", default="", help="Optional LLoVi-direct output JSON. If provided, SEAC keeps this baseline by default.")
    p.add_argument("--baseline_mode", default="auto", choices=["auto", "file", "llm", "rule"], help="auto=file if baseline_pred_path exists else llm when model is not rule/debug else rule.")

    # Model / LLM settings.
    p.add_argument("--model", default="rule", help="Use 'rule' for CPU/debug; use local HF model path/name for LLM mode.")
    p.add_argument("--api_key", default="")
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--max_new_tokens", default=160, type=int)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # SEAC retrieval / scoring.
    p.add_argument("--retrieval_top_k", default=5, type=int)
    p.add_argument("--evidence_window", default=1, type=int)
    p.add_argument("--retrieval_alpha", default=0.40, type=float, help="Question weight in question-option evidence retrieval.")
    p.add_argument("--margin_threshold", default=0.65, type=float)
    p.add_argument("--min_support", default=0.40, type=float)
    p.add_argument("--max_missing", default=0.35, type=float)
    p.add_argument("--max_contradiction", default=0.55, type=float)
    p.add_argument("--force_seac", action="store_true", help="Ignore baseline-preserving gate and always output the best calibrated SEAC option. Useful for ablation only.")

    # Optional LLM verification / arbitration.
    p.add_argument("--verifier_mode", default="rule", choices=["rule", "llm", "hybrid"], help="rule is fastest; hybrid uses rule scores then optional LLM top-2 arbitration.")
    p.add_argument("--enable_pairwise_llm", action="store_true", help="Use LLM only when SEAC best differs from baseline or top-2 are close.")
    p.add_argument("--pairwise_margin", default=0.06, type=float, help="Run pairwise arbitration when top-2 calibrated score gap is below this value.")

    # Aggressive LLM multi-choice verifier. This is designed for strong baselines with option bias.
    p.add_argument("--enable_multichoice_llm", action="store_true", help="Use an LLM to re-rank all options for suspicious samples, not only the rule top-1 alternative.")
    p.add_argument("--multichoice_scope", default="conflict", choices=["conflict", "baseline_e", "suspicious", "all"],
                   help="conflict: only when rule SEAC conflicts with baseline; baseline_e: also verify baseline E; suspicious: baseline E/invalid or high rule alternative; all: every sample.")
    p.add_argument("--suspicious_letters", default="E", help="Baseline letters to treat as potentially biased, e.g. 'E' or 'DE'. Empty string disables letter suspicion.")
    p.add_argument("--multichoice_min_margin", default=0.12, type=float, help="Run multi-choice verifier if best rule alternative beats baseline by this margin.")
    p.add_argument("--multichoice_allow_medium", action="store_true", help="Allow MEDIUM confidence LLM re-rank outputs to override baseline. Default requires HIGH.")
    p.add_argument("--multichoice_max_calls", default=-1, type=int, help="Safety cap on multi-choice LLM calls. -1 means no cap.")

    # Counterfactual bias settings.
    p.add_argument("--disable_counterfactual", action="store_true")
    p.add_argument("--beta_question", default=0.30, type=float)
    p.add_argument("--beta_option", default=0.15, type=float)
    p.add_argument("--beta_position", default=0.10, type=float)

    # Reproducibility.
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def load_baseline_predictions(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"baseline_pred_path not found: {path}")
    data = load_json(path)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data if isinstance(data, dict) else {}


def direct_baseline_prompt(item: Dict[str, Any], fps: float, caption_every: int) -> str:
    choices = get_choices(item)
    return f"""Please provide a single-letter answer (A, B, C, D, E) to the following multiple-choice question.
Your answer must be one of A, B, C, D, or E. Return only the letter as the first character.
You are given language descriptions of a video. The video captions are sequential and each caption starts with a frame/time index.

Video descriptions:
{item.get('narration', '')}

Question: {item.get('question', '').rstrip('?')}?
Choices:
A: {choices[0]}
B: {choices[1]}
C: {choices[2]}
D: {choices[3]}
E: {choices[4] if len(choices) > 4 else ''}

Answer:"""


def rule_baseline_pred(item: Dict[str, Any]) -> int:
    """A no-LLM fallback baseline. Not intended as paper baseline, only for pipeline tests."""
    from seac_utils import build_option_ledgers
    ledgers = build_option_ledgers(item, top_k=5, window=1)
    return select_best_ledger(ledgers).index


def get_baseline_pred(
    item: Dict[str, Any],
    ukey: str,
    baseline_data: Dict[str, Any],
    args,
    model=None,
) -> Dict[str, Any]:
    mode = args.baseline_mode
    if mode == "auto":
        if baseline_data:
            mode = "file"
        elif str(args.model).lower() in {"rule", "debug", "offline_rule"}:
            mode = "rule"
        else:
            mode = "llm"

    if mode == "file":
        if ukey in baseline_data:
            pred = int(baseline_data[ukey].get("pred", -1))
            return {"pred": pred, "mode": "file", "response": baseline_data[ukey].get("response", "")}
        # fallback if partial baseline file.
        mode = "rule" if str(args.model).lower() in {"rule", "debug", "offline_rule"} else "llm"

    if mode == "llm":
        if model is None:
            raise RuntimeError("LLM baseline requested but model is None")
        model.set_post_process_fn(first_char_as_answer)
        head = "You are a careful video question answering assistant."
        pred, info = model.forward(head, [direct_baseline_prompt(item, args.fps, args.caption_every)])
        return {"pred": int(pred), "mode": "llm", "response": info.get("response", ""), "info": info}

    pred = rule_baseline_pred(item)
    return {"pred": int(pred), "mode": "rule", "response": "rule_baseline_by_option_evidence"}


def pairwise_prompt(item: Dict[str, Any], base_ledger, alt_ledger) -> str:
    return f"""You are comparing two candidate answers for a video question using ONLY the listed caption evidence.
Do not use outside commonsense. Prefer the option with stronger direct video support and weaker contradiction.
If evidence is insufficient to replace the baseline option, choose BASELINE.

Question: {item.get('question', '').rstrip('?')}?

BASELINE CANDIDATE:
{format_ledger_for_prompt(base_ledger, max_evidence=5)}

ALTERNATIVE CANDIDATE:
{format_ledger_for_prompt(alt_ledger, max_evidence=5)}

Return exactly one token: BASELINE or ALTERNATIVE."""



def multichoice_prompt(item: Dict[str, Any], baseline_pred: int, ledgers, max_evidence: int = 4) -> str:
    """Prompt for aggressive option-wise evidence verification.

    Unlike pairwise arbitration, this prompt asks the LLM to compare all candidate
    hypotheses against their own retrieved evidence. It is useful when the direct
    baseline has a known option bias, e.g., over-selecting the last choice.
    """
    baseline_letter = LETTERS[baseline_pred] if 0 <= baseline_pred < len(LETTERS) else "INVALID"
    blocks = []
    for lg in ledgers:
        blocks.append(format_ledger_for_prompt(lg, max_evidence=max_evidence))
    return f"""You are a strict video evidence verifier for a multiple-choice video QA task.
You must judge which candidate answer is best supported by the listed video caption evidence.
Do NOT choose an option only because it is linguistically plausible. Prefer direct support, correct temporal order, and low contradiction.
If the baseline option is not clearly contradicted and the alternative evidence is weak, return KEEP.

Question: {item.get('question', '').rstrip('?')}?
Baseline answer: {baseline_letter}

Candidate evidence ledgers:
{chr(10).join(blocks)}

Return exactly this format:
ANSWER: <A/B/C/D/E/KEEP>
CONFIDENCE: <HIGH/MEDIUM/LOW>
REASON: <one short evidence-based phrase>
"""


def parse_multichoice_response(resp: str) -> Dict[str, Any]:
    text = str(resp).strip().upper()
    ans = None
    m = re.search(r"ANSWER\s*[:：]\s*(KEEP|[A-E])", text)
    if m:
        ans = m.group(1)
    elif text.startswith("KEEP"):
        ans = "KEEP"
    elif text[:1] in LETTERS:
        ans = text[:1]
    conf = "LOW"
    m = re.search(r"CONFIDENCE\s*[:：]\s*(HIGH|MEDIUM|MED|LOW)", text)
    if m:
        conf = m.group(1)
    if conf == "MED":
        conf = "MEDIUM"
    return {"answer": ans or "KEEP", "confidence": conf, "raw": resp}


def run_multichoice_if_needed(item: Dict[str, Any], baseline_pred: int, ledgers, decision: Dict[str, Any], args, model=None, call_state: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    meta = {"enabled": False, "used": False, "response": "", "answer": "KEEP", "confidence": "LOW", "reason": "not_enabled"}
    if not args.enable_multichoice_llm or model is None:
        return meta
    if call_state is None:
        call_state = {"calls": 0}
    if args.multichoice_max_calls >= 0 and call_state.get("calls", 0) >= args.multichoice_max_calls:
        meta["reason"] = "max_calls_reached"
        return meta

    best = select_best_ledger(ledgers)
    base = ledgers[baseline_pred] if 0 <= baseline_pred < len(ledgers) else None
    baseline_letter = LETTERS[baseline_pred] if 0 <= baseline_pred < len(LETTERS) else "INVALID"
    suspicious_letters = set(str(args.suspicious_letters or "").upper())
    margin = float(best.calibrated_score - base.calibrated_score) if base is not None else 999.0

    scope = args.multichoice_scope
    should_run = False
    why = []
    if scope == "all":
        should_run = True; why.append("scope_all")
    if scope in {"conflict", "suspicious", "baseline_e"} and best.index != baseline_pred and (decision.get("changed") or margin >= args.multichoice_min_margin):
        should_run = True; why.append("rule_conflict_or_margin")
    if scope in {"baseline_e", "suspicious"} and baseline_letter in suspicious_letters:
        should_run = True; why.append(f"baseline_{baseline_letter}_suspicious")
    if scope == "suspicious" and (baseline_pred < 0 or baseline_pred >= len(ledgers)):
        should_run = True; why.append("baseline_invalid")

    meta.update({"enabled": True, "trigger": "+".join(why) if why else "none", "margin": round(margin, 4), "baseline_letter": baseline_letter})
    if not should_run:
        meta["reason"] = "not_triggered"
        return meta

    model.set_post_process_fn(identity)
    head = "You are a conservative but bias-aware video evidence verifier."
    _, info = model.forward(head, [multichoice_prompt(item, baseline_pred, ledgers)])
    call_state["calls"] = call_state.get("calls", 0) + 1
    parsed = parse_multichoice_response(info.get("response", ""))
    meta.update({"used": True, "response": info.get("response", ""), "answer": parsed["answer"], "confidence": parsed["confidence"], "reason": "llm_verified"})

    ans = parsed["answer"]
    if ans == "KEEP" or ans not in LETTERS:
        return meta
    pred = LETTERS.index(ans)
    if not (0 <= pred < len(ledgers)):
        return meta
    # Guardrails: a selected option must not be empty and should have at least some retrieved evidence.
    lg = ledgers[pred]
    allow_conf = {"HIGH"} | ({"MEDIUM"} if args.multichoice_allow_medium else set())
    if parsed["confidence"] not in allow_conf:
        meta["reason"] = "low_confidence_keep"
        return meta
    if not str(lg.choice).strip() or not lg.evidence:
        meta["reason"] = "empty_or_no_evidence_keep"
        return meta
    # Bias-aware aggressive mode: allow the LLM to override even if rule gate did not pass,
    # but still block obviously missing/contradictory options.
    if lg.missing <= max(args.max_missing, 0.70) and lg.contradiction <= max(args.max_contradiction, 0.70):
        decision["final_pred"] = pred
        decision["changed"] = pred != baseline_pred
        decision["alternative_pred"] = pred
        decision["reason"] = "multichoice_llm_bias_aware_override" if pred != baseline_pred else "multichoice_llm_keep_baseline"
    else:
        meta["reason"] = "failed_guardrail_keep"
    return meta

def run_pairwise_if_needed(item: Dict[str, Any], baseline_pred: int, ledgers, decision: Dict[str, Any], args, model=None) -> Dict[str, Any]:
    """Optional LLM top-2/pairwise arbitration.

    The pairwise step is conservative: it can prevent an override, but it does not
    force an override unless the gate already selected the alternative.
    """
    meta = {"enabled": False, "used": False, "response": "", "kept_or_changed": decision.get("final_pred")}
    if not args.enable_pairwise_llm or model is None:
        return meta
    if not (0 <= baseline_pred < len(ledgers)):
        return meta
    alt_idx = int(decision.get("alternative_pred", baseline_pred))
    if alt_idx == baseline_pred:
        return meta
    alt = ledgers[alt_idx]
    base = ledgers[baseline_pred]
    gap = abs(float(alt.calibrated_score - base.calibrated_score))
    should_run = bool(decision.get("changed")) or gap <= args.pairwise_margin
    meta["enabled"] = True
    meta["gap"] = round(gap, 4)
    if not should_run:
        return meta
    model.set_post_process_fn(identity)
    head = "You are a conservative video evidence arbitrator."
    _, info = model.forward(head, [pairwise_prompt(item, base, alt)])
    resp = str(info.get("response", "")).strip().upper()
    meta.update({"used": True, "response": resp})
    if "BASELINE" in resp and decision.get("changed"):
        # Cancel unsafe override.
        decision["final_pred"] = baseline_pred
        decision["changed"] = False
        decision["reason"] = "pairwise_llm_cancelled_override_keep_baseline"
    elif "ALTERNATIVE" in resp and not decision.get("changed"):
        # Only allow pairwise to override if the margin is non-negative and thresholds passed.
        if alt.calibrated_score >= base.calibrated_score and alt.support >= args.min_support and alt.missing <= args.max_missing:
            decision["final_pred"] = alt_idx
            decision["changed"] = True
            decision["reason"] = "pairwise_llm_confirmed_alternative"
    meta["kept_or_changed"] = decision.get("final_pred")
    return meta


def needs_model(args, baseline_data: Dict[str, Any]) -> bool:
    mode = args.baseline_mode
    if mode == "auto":
        mode = "file" if baseline_data else ("rule" if str(args.model).lower() in {"rule", "debug", "offline_rule"} else "llm")
    return mode == "llm" or args.verifier_mode in {"llm", "hybrid"} or args.enable_pairwise_llm or args.enable_multichoice_llm


def launch():
    args = parse_args()
    pprint(args)
    random.seed(args.seed)
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)

    processed: Dict[str, Any] = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if isinstance(processed, dict) and "data" in processed:
            processed = processed["data"]

    baseline_data = load_baseline_predictions(args.baseline_pred_path) if args.baseline_pred_path else {}
    dataset = get_dataset(args, quids_to_exclude=set(processed.keys()), num_examples_to_run=args.num_examples_to_run)
    model = get_model(args) if needs_model(args, baseline_data) else None
    llm_call_state = {"calls": 0}

    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        item = dict(item)
        ukey = item[dataset.ukey]

        baseline = get_baseline_pred(item, ukey, baseline_data, args, model=model)
        baseline_pred = int(baseline["pred"])

        ledgers = build_option_ledgers(
            item,
            top_k=args.retrieval_top_k,
            window=args.evidence_window,
            alpha=args.retrieval_alpha,
            beta_question=0.0 if args.disable_counterfactual else args.beta_question,
            beta_option=0.0 if args.disable_counterfactual else args.beta_option,
            beta_position=0.0 if args.disable_counterfactual else args.beta_position,
        )
        decision = arbitration_decision(
            baseline_pred,
            ledgers,
            margin_threshold=args.margin_threshold,
            min_support=args.min_support,
            max_missing=args.max_missing,
            max_contradiction=args.max_contradiction,
            force_seac=args.force_seac,
        )
        pairwise_meta = run_pairwise_if_needed(item, baseline_pred, ledgers, decision, args, model=model)
        multichoice_meta = run_multichoice_if_needed(item, baseline_pred, ledgers, decision, args, model=model, call_state=llm_call_state)

        pred = int(decision["final_pred"])
        best = select_best_ledger(ledgers)
        item.update({
            "pred": pred,
            "truth": normalize_truth(item.get("truth", -1)),
            "response": LETTERS[pred] if 0 <= pred < len(LETTERS) else "INVALID",
            "baseline_pred": baseline_pred,
            "baseline_mode": baseline.get("mode"),
            "baseline_response": baseline.get("response", ""),
            "seac_best_pred": int(best.index),
            "seac_best_letter": best.letter,
            "seac_decision": decision,
            "pairwise_meta": pairwise_meta,
            "multichoice_meta": multichoice_meta,
            "option_ledgers": [x.as_dict() for x in ledgers],
            "seac_summary": output_summary_row({**item, "pred": pred, "baseline_pred": baseline_pred}),
        })
        if args.save_info and "info" in baseline:
            item["baseline_info"] = baseline["info"]
        processed[ukey] = item

        if i % args.save_every == 0:
            save_json(processed, output_path)
        pbar.update(1)

    save_json(processed, output_path)

    if not args.disable_eval:
        if args.dataset == "egoschema":
            out = eval_qa_egoschema(processed)
        elif args.dataset in {"nextqa", "intentqa", "nextgqa"}:
            out = eval_qa_nextqa(args.anno_path, processed)
        else:
            # For VideoMME, keep raw data; use its official conversion/eval outside if needed.
            out = {"data": processed, "note": "VideoMME raw SEAC output saved; use eval_videomme.py if your annotation format matches."}
        save_json(out, output_path)


if __name__ == "__main__":
    launch()
