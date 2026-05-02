"""Gradio demo for the fine-tuned YOLO26-n Shahed detector.

Pulls weights from HF Hub on first run, then serves a webapp with image and
video inference tabs. Runs on whatever GPU is available.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

REPO = os.environ.get("HF_MODEL_REPO", "sapoepsilon/swarmsight-yolo26n")
WEIGHT_FILE = os.environ.get("WEIGHT_FILE", "weights/best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"[serve] device={DEVICE}")
print(f"[serve] downloading {REPO}/{WEIGHT_FILE} ...")
model_path = hf_hub_download(REPO, WEIGHT_FILE)
print(f"[serve] loading {model_path}")
model = YOLO(model_path)
model.to(DEVICE)
print(f"[serve] ready: {sum(p.numel() for p in model.model.parameters())/1e6:.2f}M params")


def predict_image(img: np.ndarray | None, conf: float, iou: float):
    if img is None:
        return None, "upload an image"
    t0 = time.perf_counter()
    results = model(img, conf=conf, iou=iou, imgsz=640, device=DEVICE, verbose=False)
    elapsed = (time.perf_counter() - t0) * 1000
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        info = f"no detections    inference={elapsed:.1f} ms"
    else:
        lines = [f"{len(boxes)} detection(s)    inference={elapsed:.1f} ms", "─" * 50]
        for i, box in enumerate(boxes, 1):
            c = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            lines.append(f"  {i}. shahed   conf={c:.3f}   bbox=({x1},{y1})→({x2},{y2})")
        info = "\n".join(lines)
    return annotated_rgb, info


def predict_video(video_path: str | None, conf: float, iou: float, max_seconds: float):
    if not video_path:
        return None, "upload a video"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * max_seconds) if max_seconds > 0 else 100000
    out_path = str(Path("/tmp") / f"swarmsight_out_{int(time.time())}.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    detected = 0
    inferences_ms: list[float] = []
    t_total = time.perf_counter()
    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        results = model(frame, conf=conf, iou=iou, imgsz=640, device=DEVICE, verbose=False)
        inferences_ms.append((time.perf_counter() - t0) * 1000)
        annotated = results[0].plot()
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            detected += 1
        out.write(annotated)
        n += 1

    cap.release()
    out.release()
    total_s = time.perf_counter() - t_total
    avg_ms = float(np.mean(inferences_ms)) if inferences_ms else 0.0
    info = (
        f"frames processed: {n} (capped at {max_seconds:.0f}s)\n"
        f"frames with detections: {detected}\n"
        f"avg inference: {avg_ms:.1f} ms / frame   ({1000/max(avg_ms, 1e-6):.0f} FPS)\n"
        f"total wall-time: {total_s:.1f}s"
    )
    return out_path, info


with gr.Blocks(title="SwarmSight — Shahed detector", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # SwarmSight — YOLO26-n Shahed detector
        Fine-tuned on ~13.7K labeled images (4 sources, RGB only).
        **Final val: mAP@0.5 = 0.973, mAP@0.5-0.95 = 0.934.**
        Running on **NVIDIA RTX 3090 Ti**.

        Upload an image or video that the model has never seen and watch it work.
        """
    )

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(type="numpy", label="Image", height=400)
                conf_i = gr.Slider(0.05, 0.95, 0.25, step=0.05, label="Confidence threshold")
                iou_i = gr.Slider(0.3, 0.9, 0.45, step=0.05, label="IoU NMS threshold (unused — YOLO26 is NMS-free)")
                btn_i = gr.Button("Detect", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Detections", height=400)
                out_info_i = gr.Textbox(label="Result", lines=8)
        btn_i.click(predict_image, [inp_img, conf_i, iou_i], [out_img, out_info_i])

    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_vid = gr.Video(label="Video", height=400)
                conf_v = gr.Slider(0.05, 0.95, 0.25, step=0.05, label="Confidence threshold")
                iou_v = gr.Slider(0.3, 0.9, 0.45, step=0.05, label="IoU NMS threshold")
                max_s = gr.Slider(1, 60, 15, step=1, label="Max seconds to process")
                btn_v = gr.Button("Process", variant="primary")
            with gr.Column(scale=1):
                out_vid = gr.Video(label="Annotated", height=400)
                out_info_v = gr.Textbox(label="Result", lines=6)
        btn_v.click(predict_video, [inp_vid, conf_v, iou_v, max_s], [out_vid, out_info_v])

    gr.Markdown(
        """
        **Tips**
        - Lower the confidence slider to see borderline detections (more false positives).
        - Try a clean Shahed photo first, then try noisy/partial images to gauge robustness.
        - Try a Mavic/Phantom image — the model was trained on a mixed-drone synthetic set,
          so it may flag those as shahed (a known training-data limitation).
        """
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
    )
