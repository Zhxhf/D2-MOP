#!/usr/bin/env bash
set -euo pipefail
OUT=${OUT:-output/debug_article2_all}
mkdir -p "$OUT"
COMMON="--dataset nextqa --data_path examples/eco_captions.json --anno_path examples/eco_val.csv --duration_path examples/eco_durations.json --num_examples_to_run 2 --start_from_scratch --disable_eval"

python main.py $COMMON --prompt_type qa_next --model debug --output_base_path "$OUT" --output_filename 00_llovi_direct.json
python article2_trustqa.py $COMMON --model debug --output_base_path "$OUT" --output_filename 01_choice_blind_only.json --disable_logic --disable_repair
python article2_trustqa.py $COMMON --model debug --output_base_path "$OUT" --output_filename 02_choice_blind_logic.json --disable_repair
python article2_trustqa.py $COMMON --model debug --output_base_path "$OUT" --output_filename 03_full_trustqa_llm_matcher.json
python article2_trustqa.py $COMMON --model debug --output_base_path "$OUT" --output_filename 04_full_trustqa_rule_matcher.json --matcher_mode rule
python article2_trustqa.py $COMMON --model debug --output_base_path "$OUT" --output_filename 05_question_only_bias.json --question_only
python article2_trustqa.py $COMMON --model debug --output_base_path "$OUT" --output_filename 06_choice_shuffle.json --shuffle_choices

for f in "$OUT"/*.json; do
  b=$(basename "$f" .json)
  python article2_eval_extra.py --pred_path "$f" --out_path "$OUT/${b}_extra.json"
done
python collect_article2_results.py --input_dir "$OUT" --out_csv "$OUT/summary.csv" --out_md "$OUT/summary.md"
echo "Done. See $OUT/summary.md"
