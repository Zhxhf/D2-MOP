#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MODEL=${MODEL:-/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2.5-7B-Instruct}
N=${N:-500}
OUTDIR=${OUTDIR:-output/mop_nextqa_500}
mkdir -p "$OUTDIR"
python article2_prompt_ensemble_vqa.py \
  --dataset nextqa \
  --data_path data/nextqa/llava1.5_fps1.json \
  --anno_path data/nextqa/val.csv \
  --duration_path data/nextqa/durations.json \
  --caption_every 2 \
  --num_examples_to_run "$N" \
  --model "$MODEL" \
  --load_in_4bit \
  --baseline_pred_path output/article2_nextqa_full/00_llovi_direct_-1.json \
  --prompt_modes direct,verify,eliminate,temporal,contrastive \
  --baseline_weight 1 \
  --min_override_votes 2 \
  --min_vote_margin 1 \
  --max_new_tokens 128 \
  --output_base_path "$OUTDIR" \
  --output_filename mop_nextqa.json \
  --start_from_scratch
python eval_seac_compare.py --pred_path "$OUTDIR/mop_nextqa.json" --out_dir "$OUTDIR"
cat "$OUTDIR/mop_nextqa_seac_compare_summary.json"
