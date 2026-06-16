"""SEAC-VQA utilities.

SEAC-VQA = Selective Evidence Arbitration with Counterfactual calibration
for caption-only long-range VideoQA.

This module is deliberately lightweight:
- no extra training;
- no video files required after captions are prepared;
- CPU-friendly lexical retrieval by default;
- optional LLM calls are handled in article2_seac_vqa.py.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple, Any, Optional

from eco_common import split_narration, simple_similarity, safe_lower, tokenize

LETTERS = ["A", "B", "C", "D", "E"]

# Common functional words that should not dominate evidence retrieval.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "at", "for", "from", "with",
    "without", "by", "about", "as", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "after", "before", "during", "next",
    "following", "first", "last", "finally", "beginning", "end", "video", "clip", "scene", "question", "answer", "option",
    "he", "she", "it", "they", "them", "his", "her", "their", "him", "this", "that", "these", "those", "there", "here",
    "person", "someone", "something", "thing", "one", "two", "three", "four", "five",
}

NEGATION_WORDS = {"not", "no", "never", "without", "none", "cannot", "can't", "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't"}

# Small antonym/action-conflict table. This is intentionally conservative.
CONFLICT_PAIRS = [
    ("enter", "leave"), ("inside", "outside"), ("open", "close"), ("opened", "closed"),
    ("sit", "stand"), ("sitting", "standing"), ("left", "right"), ("up", "down"),
    ("start", "finish"), ("begin", "end"), ("arrive", "depart"), ("approach", "away"),
    ("eat", "drink"), ("walk", "run"), ("before", "after"),
]

TEMPORAL_PATTERNS = {
    "after": [r"\bafter\b", r"\bthen\b", r"\bnext\b", r"\bfollowing\b"],
    "before": [r"\bbefore\b", r"\bprior\b", r"\bearlier\b"],
    "begin": [r"\bbeginning\b", r"\bstart\b", r"\bfirst\b"],
    "end": [r"\bend\b", r"\bfinally\b", r"\blast\b", r"\bat the end\b"],
    "repeat": [r"\bagain\b", r"\brepeat\b", r"\btwice\b", r"\bmultiple\b", r"\bmore than once\b"],
    "return": [r"\breturn\b", r"\bcomes back\b", r"\bcome back\b", r"\bback again\b"],
}


def content_tokens(text: str) -> List[str]:
    """Lowercase lexical tokens with simple stopword filtering."""
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9']*", safe_lower(text))
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def token_set(text: str) -> set:
    return set(content_tokens(text))


def compact_text(text: str, max_words: int = 80) -> str:
    words = str(text).replace("\n", " ").split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def get_choices(item: Dict[str, Any]) -> List[str]:
    choices = [item.get("optionA", ""), item.get("optionB", ""), item.get("optionC", ""), item.get("optionD", ""), item.get("optionE", "")]
    # Keep empty optionE for 4-choice datasets, but downstream filters score it very low.
    return [str(c) if c is not None else "" for c in choices]


def normalize_truth(x: Any) -> int:
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    if s[:1].upper() in LETTERS:
        return LETTERS.index(s[:1].upper())
    return -1


def parse_pred_letter(text: str) -> int:
    s = str(text).strip().upper()
    if not s:
        return -1
    # Prefer first non-space character.
    if s[0] in LETTERS:
        return LETTERS.index(s[0])
    m = re.search(r"\b([A-E])\b", s)
    return LETTERS.index(m.group(1)) if m else -1


def infer_temporal_type(question: str) -> str:
    q = safe_lower(question)
    for name, pats in TEMPORAL_PATTERNS.items():
        for pat in pats:
            if re.search(pat, q):
                return name
    return "general"


def extract_anchor_phrase(question: str, temporal_type: str) -> str:
    """Extract a rough anchor phrase around temporal trigger words.

    Example: "What does he do after opening the door" -> "opening the door".
    This is intentionally simple and robust, not a full parser.
    """
    q = str(question).strip().rstrip("?")
    q_low = q.lower()
    if temporal_type == "after" and "after" in q_low:
        return q[q_low.find("after") + len("after"):].strip(" ,.;:?")
    if temporal_type == "before" and "before" in q_low:
        return q[q_low.find("before") + len("before"):].strip(" ,.;:?")
    if temporal_type == "return":
        return "leave return come back"
    if temporal_type == "repeat":
        return "again repeat twice multiple"
    if temporal_type in {"begin", "end"}:
        return temporal_type
    return q


def compose_hypothesis(question: str, choice: str) -> str:
    q = str(question).strip()
    c = str(choice).strip()
    if not c:
        return ""
    # Lightweight templates for common question forms.
    q_clean = q.rstrip("?").strip()
    q_low = q_clean.lower()
    if q_low.startswith("why"):
        return f"The answer to '{q_clean}' is because {c}."
    if q_low.startswith("where"):
        return f"The correct location for '{q_clean}' is {c}."
    if q_low.startswith("when"):
        return f"The correct time or temporal condition for '{q_clean}' is {c}."
    if q_low.startswith("who") or q_low.startswith("which"):
        return f"The correct entity for '{q_clean}' is {c}."
    if "after" in q_low:
        return f"For the event asked in the question, after the anchor event, {c}."
    if "before" in q_low:
        return f"For the event asked in the question, before the anchor event, {c}."
    return f"For the question '{q_clean}', the video evidence supports this answer: {c}."


def narration_items(narration: Any) -> List[Dict[str, Any]]:
    items = split_narration(narration)
    # Guarantee monotonic local positions even if caption indices are sparse frame numbers.
    for pos, it in enumerate(items):
        it["pos"] = pos
    return items


def lexical_score(text_a: str, text_b: str) -> float:
    """A blend of cosine-like set overlap and coverage.

    simple_similarity alone can over-reward short options. Coverage helps ensure
    the evidence covers the option's content words.
    """
    a, b = token_set(text_a), token_set(text_b)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    cosine = inter / math.sqrt(len(a) * len(b))
    coverage_a = inter / max(1, len(a))
    coverage_b = inter / max(1, len(b))
    return 0.50 * cosine + 0.25 * coverage_a + 0.25 * coverage_b


def query_caption_score(question: str, hypothesis: str, choice: str, caption: str, alpha: float = 0.40) -> float:
    q_score = lexical_score(question, caption)
    h_score = lexical_score(hypothesis, caption)
    c_score = lexical_score(choice, caption)
    return alpha * q_score + 0.45 * h_score + 0.15 * c_score


def retrieve_option_evidence(
    narration: Any,
    question: str,
    hypothesis: str,
    choice: str,
    top_k: int = 5,
    window: int = 1,
    alpha: float = 0.40,
) -> List[Dict[str, Any]]:
    """Retrieve top caption windows for a candidate option hypothesis."""
    items = narration_items(narration)
    if not items or not str(choice).strip():
        return []
    scored = []
    for it in items:
        s = query_caption_score(question, hypothesis, choice, it["caption"], alpha=alpha)
        # Mild boost if option content tokens appear exactly in the caption.
        if token_set(choice) & token_set(it["caption"]):
            s += 0.05
        scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected_pos = []
    for s, it in scored[: max(1, top_k)]:
        if s <= 0 and len(selected_pos) >= 1:
            continue
        for p in range(max(0, it["pos"] - window), min(len(items), it["pos"] + window + 1)):
            selected_pos.append(p)
    # Deduplicate while preserving temporal order.
    selected = []
    seen = set()
    for p in sorted(set(selected_pos)):
        if p in seen:
            continue
        ev = dict(items[p])
        ev["retrieval_score"] = round(query_caption_score(question, hypothesis, choice, ev["caption"], alpha=alpha), 4)
        selected.append(ev)
        seen.add(p)
    return selected


def temporal_consistency(question: str, hypothesis: str, evidence: List[Dict[str, Any]], all_items: Optional[List[Dict[str, Any]]] = None) -> float:
    ttype = infer_temporal_type(question)
    if not evidence:
        return 0.0
    if all_items is None:
        all_items = evidence
    n = max(1, len(all_items))
    ev_positions = [int(e.get("pos", 0)) for e in evidence]

    if ttype == "general":
        return 0.50
    if ttype == "begin":
        return 1.0 - min(ev_positions) / max(1, n - 1)
    if ttype == "end":
        return max(ev_positions) / max(1, n - 1)
    if ttype == "repeat":
        # Repeated answer evidence should appear in multiple non-adjacent positions or repeatedly mention key tokens.
        if len(set(ev_positions)) >= 2 and (max(ev_positions) - min(ev_positions)) >= 2:
            return 0.75
        return 0.35
    if ttype == "return":
        joined = " ".join(e.get("caption", "") for e in all_items).lower()
        has_leave = any(w in joined for w in ["leave", "leaves", "left", "exit", "walks away", "goes away", "disappears"])
        has_enter = any(w in joined for w in ["return", "comes back", "come back", "enter", "appears", "arrives"])
        return 0.85 if has_leave and has_enter else 0.35

    anchor = extract_anchor_phrase(question, ttype)
    if not anchor or len(token_set(anchor)) == 0:
        return 0.50
    anchor_scores = [(lexical_score(anchor, it.get("caption", "")), int(it.get("pos", 0))) for it in all_items]
    anchor_scores.sort(key=lambda x: x[0], reverse=True)
    anchor_score, anchor_pos = anchor_scores[0]
    if anchor_score <= 0:
        return 0.50
    if ttype == "after":
        after_ratio = sum(1 for p in ev_positions if p >= anchor_pos) / max(1, len(ev_positions))
        return 0.25 + 0.75 * after_ratio
    if ttype == "before":
        before_ratio = sum(1 for p in ev_positions if p <= anchor_pos) / max(1, len(ev_positions))
        return 0.25 + 0.75 * before_ratio
    return 0.50


def contradiction_score(hypothesis: str, choice: str, evidence: List[Dict[str, Any]]) -> float:
    if not evidence or not choice:
        return 0.0
    h = safe_lower(hypothesis + " " + choice)
    ev_text = safe_lower(" ".join(e.get("caption", "") for e in evidence))
    score = 0.0
    # Negation conflict: choice has an action/entity while evidence has negation around similar words.
    if token_set(choice) & token_set(ev_text) and (token_set(ev_text) & NEGATION_WORDS):
        score += 0.25
    for a, b in CONFLICT_PAIRS:
        if a in h and b in ev_text:
            score += 0.20
        if b in h and a in ev_text:
            score += 0.20
    return min(score, 1.0)


def missing_score(hypothesis: str, choice: str, evidence: List[Dict[str, Any]]) -> float:
    terms = token_set(hypothesis) | token_set(choice)
    if not terms:
        return 1.0
    ev_terms = token_set(" ".join(e.get("caption", "") for e in evidence))
    # Focus mostly on answer terms. It is okay if generic question words are absent.
    choice_terms = token_set(choice)
    if choice_terms:
        coverage = len(choice_terms & ev_terms) / max(1, len(choice_terms))
    else:
        coverage = len(terms & ev_terms) / max(1, len(terms))
    return max(0.0, min(1.0, 1.0 - coverage))


def support_score(question: str, hypothesis: str, choice: str, evidence: List[Dict[str, Any]]) -> float:
    if not evidence or not choice.strip():
        return 0.0
    caps = [e.get("caption", "") for e in evidence]
    best_h = max(lexical_score(hypothesis, c) for c in caps)
    best_c = max(lexical_score(choice, c) for c in caps)
    avg_q = sum(lexical_score(question, c) for c in caps) / max(1, len(caps))
    return min(1.0, 0.55 * best_h + 0.30 * best_c + 0.15 * avg_q)


def question_only_bias(question: str, choice: str) -> float:
    # Language prior: how naturally the option overlaps with the question without video evidence.
    return lexical_score(question, choice)


def option_only_bias(choice: str) -> float:
    # Generic/vague options often look plausible without evidence. Penalize them lightly.
    toks = content_tokens(choice)
    if not toks:
        return 0.0
    generic = {"yes", "no", "person", "man", "woman", "people", "object", "thing", "something", "someone", "same", "different"}
    gen = len(set(toks) & generic) / max(1, len(set(toks)))
    length_prior = min(0.20, len(toks) / 80.0)
    return min(1.0, 0.35 * gen + length_prior)


def position_bias(index: int, num_choices: int = 5) -> float:
    # Conservative fixed prior. It is intentionally tiny; LLM shuffle can override this in the main script.
    base = [0.040, 0.025, 0.015, 0.010, 0.005]
    return base[index] if 0 <= index < len(base) else 0.0


@dataclass
class OptionLedger:
    index: int
    letter: str
    choice: str
    hypothesis: str
    support: float
    contradiction: float
    missing: float
    temporal: float
    raw_score: float
    q_bias: float
    o_bias: float
    p_bias: float
    calibrated_score: float
    evidence: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Round floats for smaller JSON and cleaner inspection.
        for k in ["support", "contradiction", "missing", "temporal", "raw_score", "q_bias", "o_bias", "p_bias", "calibrated_score"]:
            d[k] = round(float(d[k]), 4)
        return d


def build_option_ledgers(
    item: Dict[str, Any],
    top_k: int = 5,
    window: int = 1,
    alpha: float = 0.40,
    lambda_support: float = 1.00,
    lambda_contradict: float = 0.70,
    lambda_missing: float = 0.45,
    lambda_temporal: float = 0.25,
    beta_question: float = 0.30,
    beta_option: float = 0.15,
    beta_position: float = 0.10,
    shuffle_position_bias: Optional[List[float]] = None,
) -> List[OptionLedger]:
    question = item.get("question", "")
    narration = item.get("narration", "")
    all_items = narration_items(narration)
    choices = get_choices(item)
    ledgers: List[OptionLedger] = []
    for i, choice in enumerate(choices):
        hyp = compose_hypothesis(question, choice)
        ev = retrieve_option_evidence(narration, question, hyp, choice, top_k=top_k, window=window, alpha=alpha)
        sup = support_score(question, hyp, choice, ev)
        con = contradiction_score(hyp, choice, ev)
        mis = missing_score(hyp, choice, ev)
        tmp = temporal_consistency(question, hyp, ev, all_items=all_items)
        raw = lambda_support * sup - lambda_contradict * con - lambda_missing * mis + lambda_temporal * tmp
        qb = question_only_bias(question, choice)
        ob = option_only_bias(choice)
        pb = shuffle_position_bias[i] if shuffle_position_bias and i < len(shuffle_position_bias) else position_bias(i, len(choices))
        cal = raw - beta_question * qb - beta_option * ob - beta_position * pb
        if not choice.strip():
            raw = cal = -999.0
            mis = 1.0
        ledgers.append(OptionLedger(
            index=i, letter=LETTERS[i], choice=choice, hypothesis=hyp, support=sup, contradiction=con, missing=mis,
            temporal=tmp, raw_score=raw, q_bias=qb, o_bias=ob, p_bias=pb, calibrated_score=cal, evidence=ev,
        ))
    return ledgers


def select_best_ledger(ledgers: Sequence[OptionLedger]) -> OptionLedger:
    return max(ledgers, key=lambda x: x.calibrated_score)


def margin_between(ledgers: Sequence[OptionLedger], i: int, j: int) -> float:
    return float(ledgers[i].calibrated_score - ledgers[j].calibrated_score)


def arbitration_decision(
    baseline_pred: int,
    ledgers: Sequence[OptionLedger],
    margin_threshold: float = 0.12,
    min_support: float = 0.12,
    max_missing: float = 0.92,
    max_contradiction: float = 0.55,
    force_seac: bool = False,
) -> Dict[str, Any]:
    """Baseline-preserving selective arbitration.

    The method keeps the baseline unless the alternative option has clear calibrated
    evidence advantage and passes safety thresholds.
    """
    best = select_best_ledger(ledgers)
    baseline_valid = 0 <= baseline_pred < len(ledgers)
    if not baseline_valid:
        return {
            "final_pred": best.index,
            "alternative_pred": best.index,
            "changed": True,
            "reason": "baseline_invalid_use_best_seac",
            "margin_vs_baseline": None,
            "passed_thresholds": True,
        }
    base = ledgers[baseline_pred]
    margin = best.calibrated_score - base.calibrated_score
    passed = (
        best.support >= min_support and
        best.missing <= max_missing and
        best.contradiction <= max_contradiction and
        margin > margin_threshold
    )
    if force_seac:
        return {
            "final_pred": best.index,
            "alternative_pred": best.index,
            "changed": best.index != baseline_pred,
            "reason": "force_seac_best_calibrated",
            "margin_vs_baseline": round(float(margin), 4),
            "passed_thresholds": passed,
        }
    if best.index != baseline_pred and passed:
        return {
            "final_pred": best.index,
            "alternative_pred": best.index,
            "changed": True,
            "reason": "override_baseline_with_stronger_calibrated_evidence",
            "margin_vs_baseline": round(float(margin), 4),
            "passed_thresholds": True,
        }
    return {
        "final_pred": baseline_pred,
        "alternative_pred": best.index,
        "changed": False,
        "reason": "keep_baseline_no_safe_override",
        "margin_vs_baseline": round(float(margin), 4),
        "passed_thresholds": passed,
    }


def format_ledger_for_prompt(ledger: OptionLedger, max_evidence: int = 5) -> str:
    lines = [
        f"Option {ledger.letter}: {ledger.choice}",
        f"Hypothesis: {ledger.hypothesis}",
        f"Rule scores: support={ledger.support:.3f}, contradiction={ledger.contradiction:.3f}, missing={ledger.missing:.3f}, temporal={ledger.temporal:.3f}, calibrated={ledger.calibrated_score:.3f}",
        "Evidence:",
    ]
    for e in ledger.evidence[:max_evidence]:
        lines.append(f"  - [t={e.get('index', e.get('pos'))}] {compact_text(e.get('caption',''), 50)}")
    if not ledger.evidence:
        lines.append("  - No evidence retrieved.")
    return "\n".join(lines)


def output_summary_row(data: Dict[str, Any]) -> Dict[str, Any]:
    """Quick summary for one processed item."""
    truth = normalize_truth(data.get("truth", -1))
    pred = int(data.get("pred", -1))
    base = int(data.get("baseline_pred", -1))
    return {
        "truth": truth,
        "pred": pred,
        "baseline_pred": base,
        "correct": pred == truth,
        "baseline_correct": base == truth,
        "changed": pred != base,
    }
