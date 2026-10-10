---
title: Pet Breed Classifier
emoji: 🐾
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-sa-4.0
---

# Pet Breed Classifier (Gradio app)

อัปโหลดรูปสุนัขหรือแมว แล้วแอปจะทายสายพันธุ์ (37 สายพันธุ์จาก Oxford-IIIT Pet) พร้อมแสดง 3 อันดับและความมั่นใจ

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | ซอร์สโค้ดแอป Gradio |
| `feature_extractor.py` | เตรียมภาพ + สกัดคุณลักษณะด้วย ConvNeXt-Tiny (โค้ดเดียวกับที่ใช้ตอนฝึกใน notebook) |
| `model.joblib` | แบบจำลองที่ฝึกแล้ว (StandardScaler + LogisticRegression, รายชื่อคลาส, ชื่อ backbone) |
| `examples/` | ภาพตัวอย่างข้อมูลนำเข้า (จากชุดทดสอบ, CC BY-SA 4.0) |
| `requirements.txt` | แพ็กเกจที่ต้องใช้ |

รันบนเครื่อง:
```bash
pip install -r requirements.txt
python app.py
```
แล้วเปิด http://localhost:7860 (ครั้งแรกจะดาวน์โหลด weights ของ ConvNeXt-Tiny ~110 MB อัตโนมัติ)

ข้อมูลฝึก: [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — Parkhi et al., "Cats and Dogs", CVPR 2012 (CC BY-SA 4.0)
