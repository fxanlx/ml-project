"""Host the image-classification Gradio app on Modal (https://modal.com).

The app code in image_classification/app/ is used unchanged: this file only builds a
container with that folder's requirements.txt and serves the Gradio `demo` from app.py.

One-time setup:   modal setup                  (log in through the browser)
Deploy:           modal deploy scripts/modal_app.py
The URL printed by `modal deploy` stays the same on every redeploy.

Cost control (the free plan without a card has $1 of credit):
- no min_containers  -> nothing runs while nobody uses the app (first visit after idle waits ~30-60 s)
- scaledown_window   -> the container shuts down 2 minutes after the last request
- max_containers=1   -> Gradio keeps session state in memory, so every request must reach the same container
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "image_classification" / "app"
REMOTE_APP_DIR = "/app"


def download_weights():
    """Bake the ImageNet ConvNeXt-Tiny weights into the image so a cold start does not download 110 MB."""
    from torchvision import models
    models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(APP_DIR / "requirements.txt"))   # same pinned versions as the app
    .run_function(download_weights)
    .add_local_dir(APP_DIR, remote_path=REMOTE_APP_DIR, ignore=["__pycache__", "*.pyc", ".gradio", "flagged"])
)

app = modal.App("pet-breed-classifier", image=image)


@app.function(cpu=1.0, memory=2048, max_containers=1, scaledown_window=120, timeout=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import os
    import sys

    import gradio as gr
    from fastapi import FastAPI

    sys.path.insert(0, REMOTE_APP_DIR)
    os.chdir(REMOTE_APP_DIR)
    from app import CSS, THEME, demo   # image_classification/app/app.py (loads model.joblib and the feature extractor)

    return gr.mount_gradio_app(FastAPI(), demo, path="/", theme=THEME, css=CSS)
