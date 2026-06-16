#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MODEL=${MODEL:-/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2.5-7B-Instruct}
N=${N:-500}
OUTDIR=${OUTDIR:-output/caper_nextqa_500}
mkdir -p "$OUTDIR"
python article2_caper_vqa.py \
  --dataset nextqa \
  --data_path data/nextqa/llava1.5_fps1.json \
  --anno_path data/nextqa/val.csv \
  --duration_path data/nextqa/durations.json \
  --caption_every 2 \
  --num_examples_to_run "$N" \
  --model "$MODEL" \
  --load_in_4bit \
  --baseline_pred_path output/article2_nextqa_full/00_llovi_direct_-1.json \
  --candidate_pred_paths output/article2_nextqa_full/01_choice_blind_only_-1.json,output/article2_nextqa_full/02_choice_blind_logic_-1.json,output/article2_nextqa_full/03_full_trustqa_llm_matcher_-1.json \
  --candidate_names blind,logic,fulltrust \
  --min_candidate_votes 2 \
  --risk_baseline_letters E \
  --scope risk_or_consensus \
  --retrieval_top_k 10 \
  --evidence_window 2 \
  --retrieval_alpha 0.35 \
  --min_support 0.22 \
  --max_missing 0.78 \
  --max_contradiction 0.70 \
  --min_margin_vs_base -0.15 \
  --output_base_path "$OUTDIR" \
  --output_filename caper_nextqa.json \
  --start_from_scratch
python eval_seac_compare.py --pred_path "$OUTDIR/caper_nextqa.json" --out_dir "$OUTDIR"
cat "$OUTDIR/caper_nextqa_seac_compare_summary.json"
