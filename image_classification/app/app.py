"""Gradio app: upload a photo of a cat or dog and get its breed (37 breeds, Oxford-IIIT Pet).

Uses the model trained in notebooks/image_classification.ipynb:
    image -> FeatureExtractor (ConvNeXt-Tiny, same preprocessing as training)
          -> model.joblib (StandardScaler + LogisticRegression) -> breed probabilities
"""

from pathlib import Path

import gradio as gr
import joblib

from feature_extractor import FeatureExtractor

APP_DIR = Path(__file__).resolve().parent
LOW_CONFIDENCE = 0.5          # below this the app warns that the answer may be wrong

bundle = joblib.load(APP_DIR / "model.joblib")
model = bundle["model"]
classes = bundle["classes"]
species = bundle["species"]
metrics = bundle["test_metrics"]
extractor = FeatureExtractor(bundle["backbone"])

SPECIES_NAME = {"cat": "cat", "dog": "dog"}
ICON = {"cat": "🐱", "dog": "🐶"}


def pretty(label: str) -> str:
    return label.replace("_", " ").title()


def predict(image):
    if image is None:
        return None, "⚠️ Please upload a photo first."

    probs = model.predict_proba(extractor.extract([image]))[0]
    order = probs.argsort()[::-1]
    top3 = {f"{ICON[species[classes[i]]]} {pretty(classes[i])}": float(probs[i]) for i in order[:3]}

    best = classes[order[0]]
    confidence = probs[order[0]]
    message = (f"### {ICON[species[best]]} {pretty(best)}\n"
               f"Looks like a **{SPECIES_NAME[species[best]]}** · confidence **{confidence:.0%}**")
    if confidence < LOW_CONFIDENCE:
        message += ("\n\n⚠️ **Low confidence** — this may be wrong. The pet may be a breed the model doesn't know, "
                    "a mixed breed, or not clearly visible. Check the 2nd and 3rd guesses, or try a clearer photo.")
    return top3, message


def breed_list(sp: str) -> str:
    names = sorted((pretty(c) for c in classes if species[c] == sp), key=str.lower)
    return ", ".join(names)


HERO = f"""
<div class="hero">
  <div class="hero-text">
    <div class="hero-title">Pet Breed Classifier</div>
    <div class="hero-sub">Show us your cat or dog and we'll guess the breed 🐾<br><b>37 breeds</b> (12 cats, 25 dogs) with the top 3 most likely matches</div>
    <div class="chips">
      <span class="chip">🎯 Macro-F1 {metrics['macro_f1']:.3f}</span>
      <span class="chip">✅ Accuracy {metrics['accuracy']:.1%}</span>
      <span class="chip">🏅 Top-3 {metrics['top3_accuracy']:.1%}</span>
      <span class="chip">🧪 Tested on 3,669 photos</span>
    </div>
  </div>
  <div class="hero-pets"><span class="pet dog">🐶</span><span class="pet cat">🐱</span></div>
</div>
"""

STEPS = """
<div class="steps">
  <div class="step"><span class="num">1</span>📸 Upload a photo, or pick an <b>example</b> below</div>
  <div class="step"><span class="num">2</span>🐾 Click <b>Predict breed</b></div>
  <div class="step"><span class="num">3</span>💖 See the breed, its confidence and the runner-ups</div>
</div>
"""

TIP = "💡 Works best with **one** pet whose face and body are clearly visible · JPG / PNG / WEBP"

FOOTER = """
<div class="footer">made with 💕🦴 · Training data: <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/" target="_blank">The Oxford-IIIT Pet Dataset</a>
(CC BY-SA 4.0) · Model: ConvNeXt-Tiny + Logistic Regression</div>
"""

# Passed to demo.launch() here and to gr.mount_gradio_app() in scripts/modal_app.py (Gradio 6 sets them at launch)
THEME = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="orange",
    neutral_hue="stone",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Fredoka"), "ui-rounded", "sans-serif"],
)

PAW = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='90' height='90' viewBox='0 0 90 90'>"
       "<g fill='%23f472b6' fill-opacity='0.09' transform='rotate(-15 45 45)'><ellipse cx='45' cy='55' rx='13' ry='11'/>"
       "<circle cx='29' cy='38' r='5.5'/><circle cx='40' cy='31' r='5.5'/><circle cx='51' cy='31' r='5.5'/><circle cx='62' cy='38' r='5.5'/></g></svg>")

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important;
                    background-image: url("PAW"); background-size: 110px; }
.hero { position: relative; display: flex; align-items: center; justify-content: space-between; gap: 16px;
        margin-top: 34px; padding: 26px 32px; border-radius: 28px; color: #4a1d3f;
        background: linear-gradient(120deg, #fce7f3 0%, #ffe4e6 40%, #ffedd5 75%, #fef3c7 100%);
        box-shadow: 0 14px 34px rgba(236, 72, 153, .25); }
.hero::before, .hero::after { content: ""; position: absolute; width: 0; height: 0;
        border-left: 30px solid transparent; border-right: 30px solid transparent; }
.hero::before { left: 44px; top: -30px; border-bottom: 40px solid #fbcfe8; }   /* pointy cat ear */
.hero::after { right: 48px; top: -24px; width: 46px; height: 58px; border: none; background: #fdba74;
               border-radius: 50% 50% 45% 45% / 65% 65% 35% 35%; transform: rotate(22deg); }   /* floppy dog ear */
.hero-title { font-size: 2.2rem; font-weight: 700; color: #be185d; line-height: 1.2; }
.hero-sub, .hero-sub * { color: #6b2150 !important; }
.hero-sub { margin-top: 6px; font-size: 1.05rem; line-height: 1.6; }
.hero-sub b { color: #9d174d !important; }
.hero-pets { display: flex; gap: 4px; font-size: 4.6rem; line-height: 1; }
.pet { display: inline-block; animation: bob 2.6s ease-in-out infinite; }
.pet.cat { animation-delay: -1.3s; }
@keyframes bob { 0%, 100% { transform: translateY(0) rotate(-4deg); } 50% { transform: translateY(-8px) rotate(4deg); } }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.chip { background: rgba(255, 255, 255, .7); border: 1.5px solid #f9a8d4; border-radius: 999px;
        padding: 5px 13px; font-size: .88rem; font-weight: 600; color: #9d174d; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.step { background: var(--block-background-fill); border: 2px dashed #f9a8d4; border-radius: 18px;
        padding: 12px 14px; font-size: .95rem; }
.step .num { display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
             margin-right: 8px; border-radius: 50%; background: #ec4899; color: #fff; font-size: .85rem; font-weight: 700; }
.panel { border-radius: 24px !important; border: 2px solid #fbcfe8 !important; padding: 14px !important; }
#result-md { min-height: 100px; padding: 4px 6px; }
#result-md h3 { font-size: 1.45rem; margin-bottom: 4px; color: #db2777; }
#predict-btn { font-size: 1.08rem; font-weight: 700; color: #fff !important; border: none !important;
               background: linear-gradient(90deg, #ec4899, #f97316) !important; box-shadow: 0 6px 18px rgba(236, 72, 153, .35); }
#predict-btn:hover { filter: brightness(1.08); transform: translateY(-1px); }
.footer { text-align: center; font-size: .85rem; opacity: .75; padding: 8px 0 4px; }
@media (max-width: 720px) {
  .steps { grid-template-columns: 1fr; } .hero { padding: 22px; } .hero-title { font-size: 1.6rem; } .hero-pets { font-size: 3rem; }
}
""".replace("PAW", PAW)

ABOUT = f"""
**Model:** ConvNeXt-Tiny (pretrained on ImageNet) extracts image features → Logistic Regression,
trained on 3,680 photos from the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset (CC BY-SA 4.0)

**Test set (3,669 photos):** macro-F1 = {metrics['macro_f1']:.3f}, accuracy = {metrics['accuracy']:.1%},
top-3 accuracy = {metrics['top3_accuracy']:.1%}

**Limitations:** it only knows the 37 breeds below. Other breeds, mixed breeds or other animals are always matched to the closest known breed.
Very similar breeds (e.g. American Pit Bull Terrier / Staffordshire Bull Terrier, Ragdoll / Birman) can be confused.

**🐱 Cats (12):** {breed_list('cat')}

**🐶 Dogs (25):** {breed_list('dog')}
"""

with gr.Blocks(title="Pet Breed Classifier") as demo:
    gr.HTML(HERO)
    gr.HTML(STEPS)
    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="panel", variant="panel"):
            image_in = gr.Image(type="pil", label="Your cat or dog", sources=["upload", "webcam", "clipboard"], height=360)
            gr.Markdown(TIP)
            with gr.Row():
                clear_btn = gr.ClearButton(value="🧹 Clear", variant="secondary")
                predict_btn = gr.Button("🐾 Predict breed", variant="primary", elem_id="predict-btn")
        with gr.Column(elem_classes="panel", variant="panel"):
            result_md = gr.Markdown("### 🐶🐱 Waiting for a furry friend~\nUpload a photo and click **Predict breed**", elem_id="result-md")
            label_out = gr.Label(num_top_classes=3, label="Top 3 matches")

    gr.Markdown("### 🖼️ Try an example (photos from the test set)")
    gr.Examples(
        examples=[str(p) for p in sorted((APP_DIR / "examples").glob("*.jpg"))],
        inputs=image_in,
        outputs=[label_out, result_md],
        fn=predict,
        run_on_click=True,
        cache_examples=False,
        examples_per_page=8,
    )
    with gr.Accordion("About the model, supported breeds and limitations", open=False):
        gr.Markdown(ABOUT)
    gr.HTML(FOOTER)

    predict_btn.click(predict, inputs=image_in, outputs=[label_out, result_md])
    clear_btn.add([image_in, label_out, result_md])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
