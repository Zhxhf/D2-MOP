#!/usr/bin/env bash
set -euo pipefail

REPORT=extra_dataset_mop_check_report.txt
: > "$REPORT"

echo "===== MOP extra dataset check =====" | tee -a "$REPORT"
echo "time: $(date)" | tee -a "$REPORT"
echo | tee -a "$REPORT"

echo "===== 1. Check Video-MME annotations =====" | tee -a "$REPORT"
for p in \
/home/ubuntu/videomind/VideoMind/datasets/video_mme/videomme_ml50_scvm.json \
/home/ubuntu/videomind/VideoMind/datasets/video_mme/videomme_local_medium_long_50.json \
/home/ubuntu/videomind/VideoMind/datasets/video_mme/videomme_local_all.json
do
  if [ -f "$p" ]; then
    echo "[OK] $p" | tee -a "$REPORT"
    python - <<PY | tee -a "$REPORT"
import json
p="$p"
data=json.load(open(p,"r",encoding="utf-8"))
rows=list(data.values()) if isinstance(data,dict) else data
print("rows:", len(rows))
if rows:
    x=rows[0]
    print("keys:", list(x.keys()))
    print("has_caption:", any(k.lower() in ["caption","captions","video_caption","frame_captions","context","video_text","subtitles"] for k in x.keys()))
PY
  else
    echo "[MISS] $p" | tee -a "$REPORT"
  fi
done

echo | tee -a "$REPORT"
echo "===== 2. Check Video-MME local videos =====" | tee -a "$REPORT"
VMME_N=$(find /home/ubuntu/videomind/VideoMind/datasets/video_mme -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.webm" \) 2>/dev/null | wc -l)
echo "Video-MME video files: $VMME_N" | tee -a "$REPORT"

echo | tee -a "$REPORT"
echo "===== 3. Check MLVU annotations =====" | tee -a "$REPORT"
for p in \
/home/ubuntu/videomind/VideoMind/datasets/mlvu_dev/splits/custom_test_part8_50.json \
/home/ubuntu/videomind/VideoMind/datasets/mlvu_dev/splits/part1_chunks/chunk_000.json
do
  if [ -f "$p" ]; then
    echo "[OK] $p" | tee -a "$REPORT"
    python - <<PY | tee -a "$REPORT"
import json
p="$p"
data=json.load(open(p,"r",encoding="utf-8"))
rows=list(data.values()) if isinstance(data,dict) else data
print("rows:", len(rows))
if rows:
    x=rows[0]
    print("keys:", list(x.keys()))
    print("has_caption:", any(k.lower() in ["caption","captions","video_caption","frame_captions","context","video_text","subtitles"] for k in x.keys()))
PY
  else
    echo "[MISS] $p" | tee -a "$REPORT"
  fi
done

echo | tee -a "$REPORT"
echo "===== 4. Check MLVU local videos =====" | tee -a "$REPORT"
MLVU_N=$(find /home/ubuntu/videomind/VideoMind/datasets/mlvu_dev -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.webm" \) 2>/dev/null | wc -l)
echo "MLVU video files: $MLVU_N" | tee -a "$REPORT"

echo | tee -a "$REPORT"
echo "===== 5. Check old outputs =====" | tee -a "$REPORT"
for p in \
/home/ubuntu/videomind/VideoMind/VideoMind-xinG3/outputs_mlvu/baseline_final_part1_8_all2174/output.json \
/home/ubuntu/videomind/VideoMind/VideoMind-xinG3/analysis_mlvu/mlvu_final_part1_8_all2174_score3_compare.json \
/home/ubuntu/videomind/VideoMind/LANZHOUhuiyi/D2p/LLoVi_article2_trustqa_4090/output/videomme_full_merged/baseline_videomme_full_all2700_blip_f12_qwen25_7b.json
do
  if [ -f "$p" ]; then
    echo "[OK] $p" | tee -a "$REPORT"
  else
    echo "[MISS] $p" | tee -a "$REPORT"
  fi
done

echo | tee -a "$REPORT"
echo "===== Conclusion =====" | tee -a "$REPORT"
if [ "$VMME_N" -gt 0 ]; then
  echo "Video-MME can proceed to caption generation." | tee -a "$REPORT"
else
  echo "Video-MME cannot run valid MOP now: no local videos and no captions." | tee -a "$REPORT"
fi

if [ "$MLVU_N" -gt 0 ]; then
  echo "MLVU can proceed to caption generation." | tee -a "$REPORT"
else
  echo "MLVU cannot run valid MOP now: no local videos and no captions." | tee -a "$REPORT"
fi

echo | tee -a "$REPORT"
echo "Report saved to: $REPORT"
