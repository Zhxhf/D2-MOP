#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Generate a fresh LLoVi-direct baseline on GPU1.
# Change MODEL to your local 7B instruct model path, e.g. /home/ubuntu/.../Qwen2.5-7B-Instruct
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
OUTDIR=${OUTDIR:-output/llovi_direct_nextqa}
N=${N:--1}

python main.py \
  --dataset nextqa \
  --data_path data/nextqa/llava1.5_fps1.json \
  --anno_path data/nextqa/val.csv \
  --duration_path data/nextqa/durations.json \
  --fps 0.5 \
  --caption_every 2 \
  --prompt_type qa_next \
  --task qa \
  --model "$MODEL" \
  --load_in_4bit \
  --num_examples_to_run "$N" \
  --output_base_path "$OUTDIR" \
  --output_filename llovi_direct_nextqa.json \
  --start_from_scratch
