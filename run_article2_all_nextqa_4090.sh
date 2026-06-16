#!/usr/bin/env bash
set -euo pipefail
# Usage example:
#   MODEL=/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2.5-7B-Instruct N=300 bash run_article2_all_nextqa_4090.sh
MODEL=${MODEL:-/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2.5-7B-Instruct}
N=${N:-300}
OUT=${OUT:-output/article2_nextqa}
mkdir -p "$OUT"
COMMON="--dataset nextqa --data_path data/nextqa/llava1.5_fps1.json --caption_every 2 --anno_path data/nextqa/val.csv --duration_path data/nextqa/durations.json --num_examples_to_run ${N} --start_from_scratch"
LLM="--model ${MODEL} --load_in_4bit --trust_remote_code --torch_dtype float16 --max_new_tokens 96 --temperature 0.0"

# 0. Original LLoVi direct baseline. Use qa_next because NExT-QA is a multi-choice dataset.
python main.py $COMMON --prompt_type qa_next $LLM --output_base_path "$OUT" --output_filename 00_llovi_direct_${N}.json

# 1. Choice-blind only: choices hidden in free answer stage; no temporal logic and no repair.
python article2_trustqa.py $COMMON $LLM --output_base_path "$OUT" --output_filename 01_choice_blind_only_${N}.json --disable_logic --disable_repair

# 2. Choice-blind + temporal logic: no claim repair.
python article2_trustqa.py $COMMON $LLM --output_base_path "$OUT" --output_filename 02_choice_blind_logic_${N}.json --disable_repair

# 3. Full method: choice-blind + temporal logic + dynamic hallucination repair + LLM matcher.
python article2_trustqa.py $COMMON $LLM --output_base_path "$OUT" --output_filename 03_full_trustqa_llm_matcher_${N}.json

# 4. Full method with rule matcher: faster backup if LLM matching is too slow or unstable.
python article2_trustqa.py $COMMON $LLM --output_base_path "$OUT" --output_filename 04_full_trustqa_rule_matcher_${N}.json --matcher_mode rule

# 5. Question-only bias test: no video evidence.
python article2_trustqa.py $COMMON $LLM --output_base_path "$OUT" --output_filename 05_question_only_bias_${N}.json --question_only

# 6. Choice-shuffle stability test: shuffle options and remap ground truth.
python article2_trustqa.py $COMMON $LLM --output_base_path "$OUT" --output_filename 06_choice_shuffle_${N}.json --shuffle_choices

for f in "$OUT"/*_${N}.json; do
  b=$(basename "$f" .json)
  python article2_eval_extra.py --pred_path "$f" --out_path "$OUT/${b}_extra.json"
done
python collect_article2_results.py --input_dir "$OUT" --out_csv "$OUT/summary_${N}.csv" --out_md "$OUT/summary_${N}.md"
echo "Done. See $OUT/summary_${N}.md"
