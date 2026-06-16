#!/usr/bin/env bash
set -e
python article2_trustqa.py --dataset nextqa --data_path examples/eco_captions.json --anno_path examples/eco_val.csv --duration_path examples/eco_durations.json --model debug --num_examples_to_run 2 --output_base_path output/debug_article2 --output_filename debug.json --start_from_scratch --disable_eval
python article2_eval_extra.py --pred_path output/debug_article2/debug.json --out_path output/debug_article2/extra.json
