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

SPECIES_TH = {"cat": "แมว", "dog": "สุนัข"}
ICON = {"cat": "🐱", "dog": "🐶"}


def pretty(label: str) -> str:
    return label.replace("_", " ").title()


def predict(image):
    if image is None:
        return None, "⚠️ กรุณาอัปโหลดรูปภาพก่อน"

    probs = model.predict_proba(extractor.extract([image]))[0]
    order = probs.argsort()[::-1]
    top3 = {f"{ICON[species[classes[i]]]} {pretty(classes[i])}": float(probs[i]) for i in order[:3]}

    best = classes[order[0]]
    confidence = probs[order[0]]
    message = (f"### ผลทำนาย: {ICON[species[best]]} **{pretty(best)}** ({SPECIES_TH[species[best]]})\n"
               f"ความมั่นใจ **{confidence:.0%}**")
    if confidence < LOW_CONFIDENCE:
        message += ("\n\n⚠️ **ความมั่นใจต่ำ** — ผลอาจไม่ถูกต้อง ภาพอาจเป็นสายพันธุ์ที่แบบจำลองไม่รู้จัก, "
                    "พันธุ์ผสม, หรือเห็นตัวสัตว์ไม่ชัด ลองดูอันดับ 2–3 ประกอบ หรือใช้ภาพที่เห็นหน้าและลำตัวชัดเจน")
    return top3, message


def breed_list(sp: str) -> str:
    names = sorted((pretty(c) for c in classes if species[c] == sp), key=str.lower)
    return ", ".join(names)


DESCRIPTION = f"""
# 🐾 Pet Breed Classifier — จำแนกสายพันธุ์สุนัขและแมว
อัปโหลดรูปสุนัขหรือแมว แล้วแบบจำลองจะทายว่าเป็นสายพันธุ์ใดใน **37 สายพันธุ์** (แมว 12, สุนัข 25)
พร้อมแสดง 3 อันดับที่เป็นไปได้มากที่สุด

**วิธีใช้งาน**
1. อัปโหลดรูป (ลากไฟล์มาวาง / คลิกเลือกไฟล์ / ถ่ายจากกล้อง) หรือคลิกภาพในหัวข้อ **ตัวอย่างข้อมูลนำเข้า** ด้านล่าง
2. กดปุ่ม **ทำนายสายพันธุ์**
3. ดูผลทำนายพร้อมความมั่นใจ (%) และอันดับ 2–3

**เคล็ดลับ:** ใช้ภาพที่มีสัตว์ **1 ตัว** เห็นหน้าและลำตัวชัดเจน รองรับไฟล์ JPG / PNG / WEBP
"""

ABOUT = f"""
**แบบจำลอง:** ConvNeXt-Tiny (pretrained ImageNet) สกัดคุณลักษณะภาพ → Logistic Regression
ฝึกด้วยชุดข้อมูล [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (CC BY-SA 4.0) 3,680 ภาพ

**ผลบนชุดทดสอบ (3,669 ภาพ):** macro-F1 = {metrics['macro_f1']:.3f}, accuracy = {metrics['accuracy']:.1%},
top-3 accuracy = {metrics['top3_accuracy']:.1%}

**ข้อจำกัด:** รู้จักเฉพาะ 37 สายพันธุ์ด้านล่าง — ภาพสายพันธุ์อื่น พันธุ์ผสม หรือสัตว์ชนิดอื่น จะถูกทายเป็นสายพันธุ์ที่ใกล้เคียงที่สุดเสมอ
สายพันธุ์ที่หน้าตาคล้ายกันมาก (เช่น American Pit Bull Terrier / Staffordshire Bull Terrier, Ragdoll / Birman) อาจสับสนกัน

**🐱 แมว (12):** {breed_list('cat')}

**🐶 สุนัข (25):** {breed_list('dog')}
"""

with gr.Blocks(title="Pet Breed Classifier") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="รูปสุนัขหรือแมว", sources=["upload", "webcam", "clipboard"], height=360)
            with gr.Row():
                clear_btn = gr.ClearButton(value="ล้าง")
                predict_btn = gr.Button("ทำนายสายพันธุ์", variant="primary")
        with gr.Column():
            result_md = gr.Markdown("ผลทำนายจะแสดงที่นี่")
            label_out = gr.Label(num_top_classes=3, label="3 อันดับที่เป็นไปได้มากที่สุด")

    gr.Markdown("### ตัวอย่างข้อมูลนำเข้า (คลิกเพื่อทดลอง — ภาพจากชุดทดสอบ)")
    gr.Examples(
        examples=[str(p) for p in sorted((APP_DIR / "examples").glob("*.jpg"))],
        inputs=image_in,
        outputs=[label_out, result_md],
        fn=predict,
        run_on_click=True,
        cache_examples=False,
        examples_per_page=8,
    )
    with gr.Accordion("เกี่ยวกับแบบจำลอง, สายพันธุ์ที่รองรับ และข้อจำกัด", open=False):
        gr.Markdown(ABOUT)

    predict_btn.click(predict, inputs=image_in, outputs=[label_out, result_md])
    clear_btn.add([image_in, label_out, result_md])

if __name__ == "__main__":
    demo.launch()
