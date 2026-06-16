import os
import re
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/ubuntu/videomind/VideoMind/LANZHOUhuiyi/D2p/LLoVi_article2_trustqa_4090")
OUT_DIR = ROOT / "figs_mop_vqa_cases"
VIDEO_DIR = OUT_DIR / "videos"
FRAME_DIR = OUT_DIR / "frames"
FIG_DIR = OUT_DIR / "figures"
LOG_PATH = OUT_DIR / "case_visual_report.txt"

for d in [OUT_DIR, VIDEO_DIR, FRAME_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 4 个论文案例
CASES = [
    {
        "fig": "fig1_multiview_consensus_success",
        "title": "Figure 1. Multi-view consensus corrects the baseline",
        "case_type": "Success: all prompt views agree",
        "qid": "6356067859_6",
        "video_id": "6356067859",
        "qtype": "TN",
        "question": "what does the girl in white do after bending down in the middle",
        "options": [
            "A. grab her",
            "B. feed horse with grass",
            "C. run towards the camera",
            "D. umbrella",
            "E. put her arms up",
        ],
        "gt": "B",
        "baseline": "E",
        "mop_d": "B",
        "mop_f": "B",
        "prompt_preds": {"Direct": "B", "Verify": "B", "Eliminate": "B", "Temporal": "B", "Contrastive": "B"},
        "vote": {"B": 5, "E": 1},
        "analysis": "All prompt views support the correct answer B. The full voting strategy safely overrides the baseline.",
    },
    {
        "fig": "fig2_intent_reasoning_success",
        "title": "Figure 2. Multi-view prompting improves intent reasoning",
        "case_type": "Success: intent/causal reasoning",
        "qid": "5902452647_0",
        "video_id": "5902452647",
        "qtype": "CW",
        "question": "why did the baby hold the ball and moving forward",
        "options": [
            "A. give ball to lady",
            "B. kick off the ground",
            "C. to throw it",
            "D. wants to play with girl",
            "E. person inside is walking",
        ],
        "gt": "D",
        "baseline": "B",
        "mop_d": "D",
        "mop_f": "D",
        "prompt_preds": {"Direct": "D", "Verify": "E", "Eliminate": "D", "Temporal": "D", "Contrastive": "D"},
        "vote": {"D": 4, "E": 1, "B": 1},
        "analysis": "The baseline focuses on a surface motion cue, while multiple MOP views support the intended interaction.",
    },
    {
        "fig": "fig3_d_correct_f_wrong",
        "title": "Figure 3. MOP-VQA-D is correct but MOP-VQA-F is suppressed",
        "case_type": "D correct, F wrong",
        "qid": "8783897632_7",
        "video_id": "8783897632",
        "qtype": "TC",
        "question": "how does the male cyclist react when he sees the steep path",
        "options": [
            "A. fells on the ground",
            "B. dismount from his bicycle",
            "C. hold the bicycle",
            "D. support the vegetables in basket",
            "E. cycle down",
        ],
        "gt": "B",
        "baseline": "E",
        "mop_d": "B",
        "mop_f": "E",
        "prompt_preds": {"Direct": "B", "Verify": "E", "Eliminate": "E", "Temporal": "E", "Contrastive": "E"},
        "vote": {"B": 1, "E": 5},
        "analysis": "The direct view captures the correct reaction, but auxiliary views are biased toward E, so the full vote suppresses D.",
    },
    {
        "fig": "fig4_f_correct_d_wrong",
        "title": "Figure 4. MOP-VQA-F corrects the direct-view error",
        "case_type": "F correct, D wrong",
        "qid": "5328616848_5",
        "video_id": "5328616848",
        "qtype": "TC",
        "question": "what was the boy doing while the woman was pouring the ingredients",
        "options": [
            "A. fidgeting with something",
            "B. pretend to feed toy dog",
            "C. watch tv",
            "D. pour water on dog",
            "E. drying clothes",
        ],
        "gt": "A",
        "baseline": "E",
        "mop_d": "B",
        "mop_f": "A",
        "prompt_preds": {"Direct": "B", "Verify": "A", "Eliminate": "A", "Temporal": "B", "Contrastive": "A"},
        "vote": {"B": 2, "A": 3, "E": 1},
        "analysis": "The full voting strategy aggregates verify, eliminate and contrastive views to correct the direct-view mistake.",
    },
]

SEARCH_ROOTS = [
    Path("/home/ubuntu/videomind/VideoMind/datasets"),
    Path("/home/ubuntu/videomind/VideoMind/LANZHOUhuiyi"),
    Path("/home/ubuntu/videomind"),
]

ANNOTATION_CANDIDATES = [
    ROOT / "data/nextqa/val.csv",
    ROOT / "data/nextqa/test.csv",
    ROOT / "data/nextqa/train.csv",
    Path("/home/ubuntu/videomind/VideoMind/datasets/nextqa/val.csv"),
    Path("/home/ubuntu/videomind/VideoMind/datasets/nextqa/train.csv"),
    Path("/home/ubuntu/videomind/VideoMind/datasets/intentqa/official_repo/datasets/IntentQA/val.csv"),
]

VIDEO_EXTS = [".mp4", ".mkv", ".webm", ".avi", ".mov"]

def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

def run_cmd(cmd, check=False):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 1, "", str(e)

def find_local_video(video_id):
    # 先快速按常见文件名查找
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for ext in VIDEO_EXTS:
            p = root / f"{video_id}{ext}"
            if p.exists():
                return p

    # 再 find 搜索
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for ext in VIDEO_EXTS:
            cmd = ["bash", "-lc", f"find '{root}' -type f -name '{video_id}{ext}' 2>/dev/null | head -1"]
            code, out, err = run_cmd(cmd)
            if out.strip():
                return Path(out.strip())
    return None

def find_url_from_annotations(video_id):
    # 有些标注可能包含 url 字段；NExT-QA 通常不一定有
    for p in ANNOTATION_CANDIDATES:
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vals = {k: str(v) for k, v in row.items()}
                    joined = " ".join(vals.values())
                    if video_id not in joined:
                        continue
                    for key in ["url", "youtube_url", "video_url", "link"]:
                        if key in row and str(row[key]).startswith("http"):
                            return row[key]
        except Exception:
            pass
    return None

def try_download(video_id):
    url = find_url_from_annotations(video_id)
    if not url:
        return None, "no_url_in_annotations"

    out_tmpl = str(VIDEO_DIR / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "mp4/best",
        "--no-playlist",
        "-o", out_tmpl,
        url
    ]
    code, out, err = run_cmd(cmd)
    if code != 0:
        return None, err[-1000:]

    for ext in VIDEO_EXTS:
        p = VIDEO_DIR / f"{video_id}{ext}"
        if p.exists():
            return p, "downloaded"
    return None, "download_finished_but_file_not_found"

def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    code, out, err = run_cmd(cmd)
    if code == 0:
        try:
            return float(out.strip())
        except Exception:
            return None
    return None

def extract_frames_cv2(video_path, case_name, n=6):
    try:
        import cv2
    except Exception:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

    out_dir = FRAME_DIR / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [int((i + 1) * frame_count / (n + 1)) for i in range(n)]
    paths = []
    for j, idx in enumerate(indices, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        out = out_dir / f"frame_{j:02d}.jpg"
        img.save(out, quality=92)
        paths.append(out)
    cap.release()
    return paths

def extract_frames_ffmpeg(video_path, case_name, n=6):
    duration = get_video_duration(video_path)
    if not duration:
        return []

    out_dir = FRAME_DIR / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i in range(n):
        t = (i + 1) * duration / (n + 1)
        out = out_dir / f"frame_{i+1:02d}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out)
        ]
        code, stdout, stderr = run_cmd(cmd)
        if out.exists():
            paths.append(out)
    return paths

def extract_frames(video_path, case_name):
    paths = extract_frames_cv2(video_path, case_name, n=6)
    if paths:
        return paths
    return extract_frames_ffmpeg(video_path, case_name, n=6)

def get_font(size=28, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def wrap_text(draw, text, font, max_width):
    words = str(text).split()
    lines = []
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines

def draw_wrapped(draw, xy, text, font, fill, max_width, line_gap=8):
    x, y = xy
    lines = wrap_text(draw, text, font, max_width)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y

def paste_frame_strip(canvas, frames, x, y, w, h):
    draw = ImageDraw.Draw(canvas)
    n = max(len(frames), 1)
    gap = 14
    fw = int((w - gap * (n - 1)) / n)
    fh = h
    for i in range(n):
        bx = x + i * (fw + gap)
        draw.rounded_rectangle([bx, y, bx + fw, y + fh], radius=12, outline=(180,180,180), width=2, fill=(245,245,245))
        if i < len(frames):
            try:
                img = Image.open(frames[i]).convert("RGB")
                img.thumbnail((fw, fh - 34))
                px = bx + (fw - img.width) // 2
                py = y + 8
                canvas.paste(img, (px, py))
            except Exception:
                pass
        draw.text((bx + 10, y + fh - 28), f"Frame {i+1}", font=get_font(18), fill=(80,80,80))

def draw_vote_box(draw, x, y, label, pred, gt):
    font = get_font(24, bold=True)
    small = get_font(20)
    color = (220, 245, 225) if pred == gt else (250, 225, 225)
    draw.rounded_rectangle([x, y, x + 230, y + 72], radius=12, fill=color, outline=(190,190,190), width=2)
    draw.text((x + 14, y + 10), label, font=small, fill=(50,50,50))
    draw.text((x + 14, y + 36), f"Pred: {pred}", font=font, fill=(20,20,20))

def draw_summary_box(draw, x, y, title, value, correct=None):
    font_t = get_font(22)
    font_v = get_font(30, bold=True)
    if correct is True:
        fill = (220, 245, 225)
    elif correct is False:
        fill = (250, 225, 225)
    else:
        fill = (235, 240, 250)
    draw.rounded_rectangle([x, y, x + 250, y + 88], radius=14, fill=fill, outline=(185,185,185), width=2)
    draw.text((x + 14, y + 12), title, font=font_t, fill=(70,70,70))
    draw.text((x + 14, y + 45), value, font=font_v, fill=(20,20,20))

def make_figure(case, frame_paths, video_status):
    W, H = 2400, 1700
    canvas = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(canvas)

    font_title = get_font(46, bold=True)
    font_subtitle = get_font(28)
    font_text = get_font(26)
    font_small = get_font(22)
    font_bold = get_font(28, bold=True)

    # Header
    draw.rectangle([0, 0, W, 110], fill=(32, 43, 64))
    draw.text((60, 28), case["title"], font=font_title, fill=(255,255,255))
    draw.text((60, 82), f"QID: {case['qid']} | Type: {case['qtype']} | {case['case_type']}", font=font_small, fill=(220,230,245))

    y = 145
    draw.text((60, y), "Question", font=font_bold, fill=(20,20,20))
    y += 42
    y = draw_wrapped(draw, (60, y), case["question"], font_text, (20,20,20), 2200)
    y += 18

    # Options
    draw.text((60, y), "Options", font=font_bold, fill=(20,20,20))
    y += 42
    for opt in case["options"]:
        draw.text((80, y), opt, font=font_small, fill=(40,40,40))
        y += 34
    y += 20

    # Prediction boxes
    box_y = y
    draw_summary_box(draw, 60, box_y, "Ground truth", case["gt"], None)
    draw_summary_box(draw, 340, box_y, "Baseline", case["baseline"], case["baseline"] == case["gt"])
    draw_summary_box(draw, 620, box_y, "MOP-VQA-D", case["mop_d"], case["mop_d"] == case["gt"])
    draw_summary_box(draw, 900, box_y, "MOP-VQA-F", case["mop_f"], case["mop_f"] == case["gt"])

    draw.text((1220, box_y + 10), "Vote count", font=font_bold, fill=(20,20,20))
    vx = 1220
    vy = box_y + 48
    for k, v in case["vote"].items():
        draw.rounded_rectangle([vx, vy, vx + 120, vy + 46], radius=10, fill=(238,238,245), outline=(190,190,200))
        draw.text((vx + 14, vy + 10), f"{k}: {v}", font=font_small, fill=(30,30,30))
        vx += 135

    y += 125

    # Frame strip
    draw.text((60, y), f"Sampled video frames  ({video_status})", font=font_bold, fill=(20,20,20))
    y += 45
    paste_frame_strip(canvas, frame_paths, 60, y, 2280, 430)
    y += 470

    # Prompt predictions
    draw.text((60, y), "Prompt-view predictions", font=font_bold, fill=(20,20,20))
    y += 46
    x = 60
    for label, pred in case["prompt_preds"].items():
        draw_vote_box(draw, x, y, label, pred, case["gt"])
        x += 250
    y += 105

    # Analysis
    draw.text((60, y), "Analysis", font=font_bold, fill=(20,20,20))
    y += 42
    y = draw_wrapped(draw, (60, y), case["analysis"], font_text, (30,30,30), 2200)

    # Footer
    draw.rectangle([0, H - 70, W, H], fill=(245,245,245))
    footer = "Green boxes indicate correct predictions; red boxes indicate wrong predictions. The figure visualizes how MOP-VQA-D and MOP-VQA-F update the baseline."
    draw.text((60, H - 48), footer, font=font_small, fill=(80,80,80))

    out_png = FIG_DIR / f"{case['fig']}.png"
    out_pdf = FIG_DIR / f"{case['fig']}.pdf"
    canvas.save(out_png)
    canvas.save(out_pdf)
    return out_png, out_pdf

def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log("===== MOP-VQA case visualization =====")
    log(f"output: {OUT_DIR}")

    manifest = []

    for case in CASES:
        log("\n" + "=" * 100)
        log(f"case: {case['qid']} video_id={case['video_id']}")

        local_video = find_local_video(case["video_id"])
        status = ""

        if local_video:
            status = f"local video found: {local_video}"
            log(status)
        else:
            log("local video not found, trying to download from annotation URL...")
            local_video, msg = try_download(case["video_id"])
            if local_video:
                status = f"downloaded: {local_video}"
                log(status)
            else:
                status = f"video unavailable: {msg}"
                log(status)

        frame_paths = []
        if local_video and local_video.exists():
            frame_paths = extract_frames(local_video, case["fig"])
            log(f"extracted frames: {len(frame_paths)}")
        else:
            log("no frames extracted because video is unavailable")

        out_png, out_pdf = make_figure(case, frame_paths, status)

        log(f"saved figure png: {out_png}")
        log(f"saved figure pdf: {out_pdf}")

        manifest.append({
            "qid": case["qid"],
            "video_id": case["video_id"],
            "video_status": status,
            "num_frames": len(frame_paths),
            "png": str(out_png),
            "pdf": str(out_pdf),
        })

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log("\n===== done =====")
    log(f"manifest: {OUT_DIR / 'manifest.json'}")
    log(f"figures: {FIG_DIR}")

if __name__ == "__main__":
    main()
