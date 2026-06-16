#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MODEL=${MODEL:-/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2.5-7B-Instruct}
OUTDIR=${OUTDIR:-output/seac_nextqa_v2_full}
mkdir -p "$OUTDIR"
nohup python article2_seac_vqa.py \
  --dataset nextqa \
  --data_path data/nextqa/llava1.5_fps1.json \
  --anno_path data/nextqa/val.csv \
  --duration_path data/nextqa/durations.json \
  --caption_every 2 \
  --num_examples_to_run -1 \
  --model "$MODEL" \
  --baseline_pred_path output/article2_nextqa_full/00_llovi_direct_-1.json \
  --baseline_mode file \
  --retrieval_top_k 10 \
  --evidence_window 2 \
  --retrieval_alpha 0.35 \
  --margin_threshold 0.35 \
  --min_support 0.30 \
  --max_missing 0.60 \
  --max_contradiction 0.55 \
  --enable_pairwise_llm \
  --verifier_mode hybrid \
  --pairwise_margin 0.20 \
  --enable_multichoice_llm \
  --multichoice_scope suspicious \
  --suspicious_letters E \
  --multichoice_min_margin 0.08 \
  --multichoice_allow_medium \
  --load_in_4bit \
  --output_base_path "$OUTDIR" \
  --output_filename seac_nextqa.json \
  --start_from_scratch \
  > "$OUTDIR/run.log" 2>&1 &
echo "Started. Log: $OUTDIR/run.log"
echo "After it finishes run:"
echo "python eval_seac_compare.py --pred_path $OUTDIR/seac_nextqa.json --out_dir $OUTDIR && cat $OUTDIR/seac_nextqa_seac_compare_summary.json"
