#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# CPU/no-GPU smoke test. It uses the included small NextQA caption/annotation files.
python article2_seac_vqa.py \
  --dataset nextqa \
  --data_path data/nextqa/llava1.5_fps1.json \
  --anno_path data/nextqa/val.csv \
  --duration_path data/nextqa/durations.json \
  --caption_every 2 \
  --num_examples_to_run 20 \
  --model rule \
  --baseline_mode rule \
  --verifier_mode rule \
  --output_base_path output/seac_debug \
  --output_filename seac_debug20.json \
  --start_from_scratch

python eval_seac_compare.py \
  --pred_path output/seac_debug/seac_debug20.json \
  --out_dir output/seac_debug
