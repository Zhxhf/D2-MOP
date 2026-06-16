
import os
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

from util import makedir, save_json, load_json
from dataset import get_dataset
from model import get_model
from prompts import identity, first_char_as_answer
from eval import eval_qa_nextqa, eval_qa_egoschema
from eco_common import extract_events, format_events, parse_letter, word_count
from trust_logic import parse_logic, execute_logic, repair_answer, choice_score


def parse_args():
    p = argparse.ArgumentParser("Article 2: Choice-blind temporal logic and hallucination repair")
    p.add_argument('--dataset', default='nextqa')
    p.add_argument('--data_path', default='data/nextqa/llava1.5_fps1.json')
    p.add_argument('--anno_path', default='data/nextqa/val.csv')
    p.add_argument('--duration_path', default='data/nextqa/durations.json')
    p.add_argument('--fps', default=0.5, type=float)
    p.add_argument('--caption_every', default=2, type=int)
    p.add_argument('--num_examples_to_run', default=200, type=int)
    p.add_argument('--output_base_path', required=True)
    p.add_argument('--output_filename', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--api_key', default='')
    p.add_argument('--temperature', default=0.0, type=float)
    p.add_argument('--max_new_tokens', default=96, type=int)
    p.add_argument('--load_in_4bit', action='store_true')
    p.add_argument('--trust_remote_code', action='store_true')
    p.add_argument('--torch_dtype', default='float16', choices=['float16','bfloat16','float32'])
    p.add_argument('--free_answer_mode', default='llm', choices=['llm','evidence'])
    p.add_argument('--matcher_mode', default='llm', choices=['llm','rule'])
    p.add_argument('--question_only', action='store_true', help='Do not use video evidence; for bias test.')
    p.add_argument('--shuffle_choices', action='store_true', help='Shuffle choices and remap truth; for stability test.')
    p.add_argument('--disable_logic', action='store_true', help='Ablation: skip temporal logic execution.')
    p.add_argument('--disable_repair', action='store_true', help='Ablation: skip claim-level hallucination repair.')
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--start_from_scratch', action='store_true')
    p.add_argument('--disable_eval', action='store_true')
    p.add_argument('--save_every', default=20, type=int)
    p.add_argument('--save_info', action='store_true')
    return p.parse_args()


def get_choices(item):
    return [item['optionA'], item['optionB'], item['optionC'], item['optionD'], item.get('optionE','')]


def maybe_shuffle(item, rng):
    item = dict(item)
    choices = get_choices(item)
    idxs = list(range(len(choices)))
    rng.shuffle(idxs)
    shuffled = [choices[i] for i in idxs]
    for key, val in zip(['optionA','optionB','optionC','optionD','optionE'], shuffled):
        item[key] = val
    old_truth = int(item['truth']) if str(item['truth']).isdigit() else item['truth']
    if isinstance(old_truth, int) and 0 <= old_truth < len(idxs):
        item['truth'] = idxs.index(old_truth)
    return item, idxs


def free_answer_prompt(item, event_text):
    return f"""You are an ecological video analysis assistant.
Candidate answer choices are intentionally hidden. First answer the question freely using ONLY the video event evidence.
If the evidence is insufficient, say that it is not clearly observed.

Video event evidence:
{event_text}

Question: {item['question']}?

Free answer with one concise sentence:"""


def match_prompt(item, repaired, logic_answer, path_text):
    return f"""Select the option that best matches the repaired evidence-based answer.
Do not use outside commonsense if it conflicts with evidence.

Question: {item['question']}?
Repaired answer: {repaired}
Temporal logic answer: {logic_answer}
Evidence path: {path_text}

Options:
A: {item['optionA']}
B: {item['optionB']}
C: {item['optionC']}
D: {item['optionD']}
E: {item.get('optionE','')}

Return only one letter from A, B, C, D, E as the first character."""


def launch():
    args = parse_args()
    pprint(args)
    rng = random.Random(args.seed)
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)
    processed = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if 'data' in processed: processed = processed['data']
    dataset = get_dataset(args, quids_to_exclude=set(processed.keys()), num_examples_to_run=args.num_examples_to_run)
    model = get_model(args)
    head = 'You are a careful and conservative ecological video question answering assistant.'
    pbar = tqdm(total=len(dataset))
    for i, raw_item in enumerate(dataset):
        item, shuffle_map = maybe_shuffle(raw_item, rng) if args.shuffle_choices else (dict(raw_item), None)
        events = [] if args.question_only else extract_events(item['narration'])
        event_text = 'No video evidence is provided in this question-only bias setting.' if args.question_only else format_events(events, max_items=18)
        if args.free_answer_mode == 'llm':
            model.set_post_process_fn(identity)
            _, free_info = model.forward(head, [free_answer_prompt(item, event_text)])
            free_answer = free_info.get('response','').strip()
        else:
            free_answer = 'Evidence shows: ' + '; '.join(e['caption'] for e in events[:5]) if events else 'No video evidence is available.'
            free_info = {'response': free_answer}
        if args.disable_logic:
            program = {'op': 'disabled', 'raw': item['question']}
            logic_answer, path = 'Temporal logic is disabled in this ablation.', []
        else:
            program = parse_logic(item['question'])
            logic_answer, path = execute_logic(program, events)
        # conservative fusion: keep both; optionally repair unsupported text
        initial = free_answer if args.disable_logic else (free_answer + ' ' + logic_answer)
        if args.disable_repair or not events:
            repaired, repair_meta = initial, {'claims': [], 'unsupported': [], 'unsupported_rate': None, 'disabled': args.disable_repair}
        else:
            repaired, repair_meta = repair_answer(initial, events)
        path_text = format_events(path, max_items=6)
        if args.matcher_mode == 'llm':
            model.set_post_process_fn(first_char_as_answer)
            pred, match_info = model.forward(head, [match_prompt(item, repaired, logic_answer, path_text)])
            response = match_info.get('response','')
        else:
            choices = get_choices(item)
            scores = [choice_score(repaired, logic_answer, path, c) for c in choices]
            pred = max(range(len(scores)), key=lambda j: scores[j])
            response = f'rule_scores={scores}'
            match_info = {'response': response}
        ukey = raw_item[dataset.ukey]
        processed[ukey] = dict(item)
        processed[ukey].update({
            'pred': pred,
            'response': response,
            'free_answer': free_answer,
            'logic_program': program,
            'logic_answer': logic_answer,
            'evidence_path': path,
            'repaired_answer': repaired,
            'repair_meta': repair_meta,
            'events': events[:32],
            'question_only': args.question_only,
            'shuffle_choices': args.shuffle_choices,
            'shuffle_map': shuffle_map,
            'raw_caption_words': word_count(item.get('narration','')),
            'event_words': word_count(event_text),
        })
        if args.save_info:
            processed[ukey]['free_info'] = free_info
            processed[ukey]['match_info'] = match_info
        if i % args.save_every == 0:
            save_json(processed, output_path)
        pbar.update(1)
    save_json(processed, output_path)
    if not args.disable_eval:
        if args.dataset == 'egoschema':
            out = eval_qa_egoschema(processed)
        else:
            out = eval_qa_nextqa(args.anno_path, processed)
        save_json(out, output_path)

if __name__ == '__main__':
    launch()
