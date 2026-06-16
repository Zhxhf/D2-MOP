#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Recommended low-storage run when you already have LLoVi-direct output.
# The uploaded package includes: output/article2_nextqa_full/00_llovi_direct_-1.json
# Replace BASELINE with your new baseline file path for other datasets.
BASELINE=${BASELINE:-output/article2_nextqa_full/00_llovi_direct_-1.json}
OUTDIR=${OUTDIR:-output/seac_nextqa}
MODEL=${MODEL:-rule}   # For stronger LLM pairwise arbitration, set MODEL=/path/to/Qwen2.5-7B-Instruct and add ENABLE_PAIRWISE=1.
N=${N:--1}

EXTRA_ARGS=""
if [ "${ENABLE_PAIRWISE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable_pairwise_llm --verifier_mode hybrid --load_in_4bit"
fi

python article2_seac_vqa.py \
  --dataset nextqa \
  --data_path data/nextqa/llava1.5_fps1.json \
  --anno_path data/nextqa/val.csv \
  --duration_path data/nextqa/durations.json \
  --caption_every 2 \
  --num_examples_to_run "$N" \
  --model "$MODEL" \
  --baseline_pred_path "$BASELINE" \
  --baseline_mode file \
  --retrieval_top_k 5 \
  --evidence_window 1 \
  --margin_threshold 0.65 \
  --min_support 0.40 \
  --max_missing 0.35 \
  --max_contradiction 0.55 \
  --output_base_path "$OUTDIR" \
  --output_filename seac_nextqa.json \
  --start_from_scratch \
  $EXTRA_ARGS

python eval_seac_compare.py \
  --pred_path "$OUTDIR/seac_nextqa.json" \
  --out_dir "$OUTDIR"
