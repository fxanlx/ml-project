# โครงการ Machine Learning: Regression & Image Classification

โครงการพัฒนาแบบจำลอง Machine Learning 2 งาน ตั้งแต่เตรียมข้อมูล ฝึกและประเมินแบบจำลอง จนถึงเผยแพร่เป็นเว็บแอปพลิเคชันด้วย Gradio

## 👥 ผู้จัดทำ
<!-- เรียงตามรหัสนิสิตจากน้อยไปมาก -->
| รหัสนิสิต | ชื่อ–นามสกุล |
|---|---|
| 6630300866 | นาย สกลภัทร ภิญโญ |
| 6730300281 | นาย นันทพัฒน์ วรรณลักษณ์ |
| 6730300591 | นาย ศุภวิชญ์ ไชยานุศักดิ์ |
| 6730300779 | นาย ธรรศธรณ์ เจียมตน |
| 6730300850 | นาย พัฒนเดช มะหิเมือง |

## 🌐 เว็บแอปพลิเคชัน
| แอป | URL |
|---|---|
| Regression App | _TODO: ใส่ URL_ |
| Image Classification App — Pet Breed Classifier | _TODO: ใส่ URL ของ Hugging Face Space_ |

## 📁 โครงสร้างไฟล์และโฟลเดอร์
```
.
├── README.md                     ← ไฟล์นี้
├── requirements.txt              ← แพ็กเกจทั้งหมดสำหรับรัน notebook และแอปบนเครื่อง
├── regression/                   ← ส่วนที่ 1: Regression (TODO)
└── image_classification/         ← ส่วนที่ 2: Image Classification
    ├── data/
    │   ├── train/<สายพันธุ์>/*.jpg    ← ชุดฝึก 3,680 ภาพ
    │   ├── test/<สายพันธุ์>/*.jpg     ← ชุดทดสอบ 3,669 ภาพ
    │   ├── labels.csv                ← label ของทุกภาพ (filepath, label, class_id, species, split)
    │   ├── LICENSE.txt               ← เครดิตและสัญญาอนุญาตของชุดข้อมูล (CC BY-SA 4.0)
    │   └── README.md                 ← รายละเอียดชุดข้อมูล
    ├── scripts/
    │   └── prepare_data.py           ← ดาวน์โหลด/ย่อภาพ/แบ่งชุดข้อมูล (สร้าง data/ ซ้ำได้เหมือนเดิม)
    ├── notebooks/
    │   └── image_classification.ipynb  ← สำรวจข้อมูล ฝึก ประเมิน วิเคราะห์ข้อผิดพลาด และบันทึกแบบจำลอง
    └── app/                          ← เว็บแอป (โฟลเดอร์นี้อัปโหลดขึ้น Hugging Face Spaces ได้ทันที)
        ├── app.py                    ← ซอร์สโค้ด Gradio
        ├── feature_extractor.py      ← ขั้นตอนเตรียมภาพ + สกัดคุณลักษณะ (ใช้ร่วมกันทั้ง notebook และแอป)
        ├── model.joblib              ← แบบจำลองที่ฝึกและบันทึกไว้
        ├── examples/                 ← ภาพตัวอย่างข้อมูลนำเข้า
        ├── requirements.txt          ← แพ็กเกจที่แอปต้องใช้
        └── README.md                 ← ค่าตั้ง Hugging Face Space
```

---

## ส่วนที่ 1: Regression Model

> _TODO_

---

## ส่วนที่ 2: Image Classification — จำแนกสายพันธุ์สุนัขและแมว

### แนวคิด ปัญหา และประโยชน์
ผู้เลี้ยงสัตว์ ผู้ที่จะรับเลี้ยง หรือสถานพักพิงสัตว์ มักไม่แน่ใจว่าสุนัขหรือแมวเป็นสายพันธุ์อะไร ทั้งที่สายพันธุ์บอกข้อมูลสำคัญ
เช่น ขนาดตัวเมื่อโต นิสัย การดูแลขน และโรคที่พบบ่อย หลายสายพันธุ์หน้าตาใกล้เคียงกันจนแยกด้วยตาเปล่าได้ยาก

**แอป Pet Breed Classifier** รับรูปสุนัขหรือแมว 1 รูป แล้วทายว่าเป็นสายพันธุ์ใดใน **37 สายพันธุ์** พร้อมแสดง 3 อันดับและความมั่นใจ
ช่วยให้ผู้ใช้ระบุสายพันธุ์เบื้องต้นเพื่อหาข้อมูลการดูแลต่อ และช่วยแพลตฟอร์มรับเลี้ยงสัตว์จัดหมวดหมู่ภาพอัตโนมัติ

### ชุดข้อมูล
| หัวข้อ | รายละเอียด |
|---|---|
| ชื่อ | **The Oxford-IIIT Pet Dataset** |
| แหล่งที่มา | https://www.robots.ox.ac.uk/~vgg/data/pets/ — Visual Geometry Group, University of Oxford |
| สัญญาอนุญาต | **CC BY-SA 4.0** — อนุญาตให้ใช้และเผยแพร่ต่อโดยให้เครดิต (ดู [`LICENSE.txt`](image_classification/data/LICENSE.txt)) |
| อ้างอิง | O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *"Cats and Dogs"*, IEEE CVPR 2012 |
| จำนวน | **7,349 ภาพ, 37 คลาส** — แมว 12 สายพันธุ์ (2,371 ภาพ), สุนัข 25 สายพันธุ์ (4,978 ภาพ) |
| การแบ่งข้อมูล | ใช้ **split ทางการ** ของชุดข้อมูล: **train 3,680 ภาพ / test 3,669 ภาพ** (~100 ภาพต่อคลาสในแต่ละชุด) |
| Label | ชื่อโฟลเดอร์ `data/<split>/<label>/` และไฟล์ [`labels.csv`](image_classification/data/labels.csv) ระบุ label ของทุกภาพ |
| การเตรียมไฟล์ | แปลงเป็น RGB และย่อด้านยาวสุดเหลือ 320 px เพื่อเก็บบน GitHub (~160 MB) ด้วย [`prepare_data.py`](image_classification/scripts/prepare_data.py) |

**คลาสที่จำแนก:**
- 🐱 **แมว (12):** Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx
- 🐶 **สุนัข (25):** American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle, Boxer, Chihuahua, English Cocker Spaniel, English Setter,
  German Shorthaired, Great Pyrenees, Havanese, Japanese Chin, Keeshond, Leonberger, Miniature Pinscher, Newfoundland, Pomeranian, Pug,
  Saint Bernard, Samoyed, Scottish Terrier, Shiba Inu, Staffordshire Bull Terrier, Wheaten Terrier, Yorkshire Terrier

<details>
<summary><b>จำนวนภาพแต่ละคลาส (คลิกเพื่อดู)</b></summary>

| ชนิด | สายพันธุ์ | train | test | รวม |
|---|---|---:|---:|---:|
| แมว | Abyssinian | 100 | 98 | 198 |
| แมว | Bengal | 100 | 100 | 200 |
| แมว | Birman | 100 | 100 | 200 |
| แมว | Bombay | 96 | 88 | 184 |
| แมว | British_Shorthair | 100 | 100 | 200 |
| แมว | Egyptian_Mau | 93 | 97 | 190 |
| แมว | Maine_Coon | 100 | 100 | 200 |
| แมว | Persian | 100 | 100 | 200 |
| แมว | Ragdoll | 100 | 100 | 200 |
| แมว | Russian_Blue | 100 | 100 | 200 |
| แมว | Siamese | 99 | 100 | 199 |
| แมว | Sphynx | 100 | 100 | 200 |
| สุนัข | american_bulldog | 100 | 100 | 200 |
| สุนัข | american_pit_bull_terrier | 100 | 100 | 200 |
| สุนัข | basset_hound | 100 | 100 | 200 |
| สุนัข | beagle | 100 | 100 | 200 |
| สุนัข | boxer | 100 | 99 | 199 |
| สุนัข | chihuahua | 100 | 100 | 200 |
| สุนัข | english_cocker_spaniel | 96 | 100 | 196 |
| สุนัข | english_setter | 100 | 100 | 200 |
| สุนัข | german_shorthaired | 100 | 100 | 200 |
| สุนัข | great_pyrenees | 100 | 100 | 200 |
| สุนัข | havanese | 100 | 100 | 200 |
| สุนัข | japanese_chin | 100 | 100 | 200 |
| สุนัข | keeshond | 100 | 99 | 199 |
| สุนัข | leonberger | 100 | 100 | 200 |
| สุนัข | miniature_pinscher | 100 | 100 | 200 |
| สุนัข | newfoundland | 96 | 100 | 196 |
| สุนัข | pomeranian | 100 | 100 | 200 |
| สุนัข | pug | 100 | 100 | 200 |
| สุนัข | saint_bernard | 100 | 100 | 200 |
| สุนัข | samoyed | 100 | 100 | 200 |
| สุนัข | scottish_terrier | 100 | 99 | 199 |
| สุนัข | shiba_inu | 100 | 100 | 200 |
| สุนัข | staffordshire_bull_terrier | 100 | 89 | 189 |
| สุนัข | wheaten_terrier | 100 | 100 | 200 |
| สุนัข | yorkshire_terrier | 100 | 100 | 200 |
| | **รวม** | **3,680** | **3,669** | **7,349** |
</details>

### การเตรียมภาพและแบบจำลอง
ชุดฝึกมีเพียง ~100 ภาพต่อคลาส จึงใช้ **transfer learning แบบ feature extraction** แทนการฝึก CNN ตั้งแต่ศูนย์:

1. **เตรียมภาพ:** แปลงเป็น RGB → ย่อขนาด (236 px) → ครอปกลาง 224×224 → normalize ด้วยค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานของ ImageNet
2. **สกัดคุณลักษณะ:** ส่งภาพเข้า **ConvNeXt-Tiny** ที่ฝึกบน ImageNet แล้ว (ไม่ปรับ weights) ตัดชั้นจำแนกออก ได้เวกเตอร์ 768 มิติต่อภาพ
3. **จำแนก:** `StandardScaler` → **Logistic Regression** (multinomial, C = 0.1)

ขั้นตอนที่ 1–2 อยู่ใน [`feature_extractor.py`](image_classification/app/feature_extractor.py) ซึ่ง notebook และแอปใช้**โค้ดชุดเดียวกัน** แอปจึงเตรียมภาพเหมือนตอนฝึกทุกประการ

**การเลือกแบบจำลอง:** เปรียบเทียบ 2 backbone (EfficientNet-B0, ConvNeXt-Tiny) × 3 ตัวจำแนก (Logistic Regression, SVM-RBF, kNN)
ด้วย **5-fold stratified cross-validation บนชุดฝึกเท่านั้น** (เมตริก macro-F1) — ชุดทดสอบใช้ครั้งเดียวตอนประเมินผลสุดท้าย

| Backbone | Logistic Regression | SVM (RBF) | kNN |
|---|---:|---:|---:|
| **ConvNeXt-Tiny** | **0.954** ✅ | 0.951 | 0.916 |
| EfficientNet-B0 | 0.938 | 0.934 | 0.903 |

_ค่าในตาราง = CV macro-F1 ที่ดีที่สุดของแต่ละคู่_ — Logistic Regression ดีที่สุด ฝึกเร็ว ให้ค่าความน่าจะเป็นโดยตรง และไฟล์โมเดลเล็ก (0.13 MB)

### ผลประเมินหลัก (ชุดทดสอบ 3,669 ภาพ)
| เมตริก | ค่า | ความหมาย |
|---|---:|---|
| **Macro-F1 (เมตริกหลัก)** | **0.934** | F1 ของแต่ละสายพันธุ์เฉลี่ยแบบ **macro** (ทุกคลาสน้ำหนักเท่ากัน) — ทุกสายพันธุ์สำคัญเท่ากัน |
| Accuracy | 0.934 | ทายสายพันธุ์ถูก ~93 จาก 100 ภาพ (เดาสุ่มจะถูกเพียง ~2.7%) |
| Macro precision / recall | 0.935 / 0.933 | |
| Weighted-F1 | 0.934 | ใกล้ macro-F1 เพราะจำนวนภาพแต่ละคลาสใกล้เคียงกัน |
| Top-3 accuracy | 0.993 | คำตอบที่ถูกอยู่ใน 3 อันดับแรกที่แอปแสดง |
| แยกแมว/สุนัขถูก | 0.999 | ข้อผิดพลาดเกือบทั้งหมดเกิดภายในสายพันธุ์ที่คล้ายกัน |

- **จำแนกได้ดีที่สุด:** leonberger (F1 1.00), newfoundland, yorkshire_terrier, scottish_terrier (0.995), pomeranian, pug, samoyed (≥ 0.985)
- **ยังสับสน:** american_pit_bull_terrier (0.74) ↔ staffordshire_bull_terrier (0.78), Ragdoll (0.75) ↔ Birman (0.80), Egyptian_Mau / Abyssinian → Bengal
  ซึ่งเป็นสายพันธุ์ที่หน้าตา/ลายขนคล้ายกันจริง
- โดยรวม **แมวยากกว่าสุนัข** (F1 เฉลี่ย 0.89 vs 0.96)
- ผลรายคลาส, confusion matrix และการวิเคราะห์ข้อผิดพลาดครบถ้วนอยู่ใน [notebook](image_classification/notebooks/image_classification.ipynb)

### ข้อจำกัด
- รู้จักเพียง **37 สายพันธุ์** — ภาพสายพันธุ์อื่น พันธุ์ผสม หรือสัตว์ชนิดอื่น จะถูกทายเป็นสายพันธุ์ที่ใกล้เคียงที่สุดเสมอ (ไม่มีคลาส "ไม่รู้จัก")
  แอปจึงแสดงคำเตือนเมื่อความมั่นใจต่ำกว่า 50%
- สายพันธุ์ที่คล้ายกันมากยังสับสนกัน และแบบจำลองบางครั้งมั่นใจเกินจริงแม้ทายผิด
- ทำงานได้ดีที่สุดกับภาพที่มีสัตว์ 1 ตัวเห็นชัด — ภาพลูกสัตว์ มุมกล้องแปลก หรือหลายตัวในภาพ ความแม่นยำลดลง
- ใช้ backbone แบบไม่ fine-tune เพื่อให้ฝึกบน CPU ได้ — การ fine-tune พร้อม data augmentation น่าจะเพิ่มความแม่นยำได้อีก

### วิธีใช้งานแอป
1. เปิด URL ของแอป (ตารางด้านบน)
2. อัปโหลดรูปสุนัขหรือแมว (ลากไฟล์มาวาง / คลิกเลือกไฟล์ / ถ่ายจากกล้อง) **หรือคลิกภาพในหัวข้อ "ตัวอย่างข้อมูลนำเข้า"**
3. กด **ทำนายสายพันธุ์** → แอปแสดงสายพันธุ์ ชนิดสัตว์ (แมว/สุนัข) ความมั่นใจ และ 3 อันดับที่เป็นไปได้มากที่สุด

**ตัวอย่างข้อมูลนำเข้า:** ไฟล์ภาพ JPG/PNG/WEBP ที่มีสุนัขหรือแมว 1 ตัว เช่น ภาพใน [`image_classification/app/examples/`](image_classification/app/examples/)

| ข้อมูลนำเข้า | ผลลัพธ์ที่ได้ |
|---|---|
| [`examples/Persian.jpg`](image_classification/app/examples/Persian.jpg) | 🐱 **Persian** (แมว) — ความมั่นใจ 99% |
| [`examples/beagle.jpg`](image_classification/app/examples/beagle.jpg) | 🐶 **Beagle** (สุนัข) — ความมั่นใจ 99% |
| ภาพที่ไม่ใช่สุนัข/แมว | ความมั่นใจต่ำ + ⚠️ คำเตือนว่าผลอาจไม่ถูกต้อง |

---

## 💻 การติดตั้งและรันบนเครื่อง

ต้องมี **Python 3.12 ขึ้นไป** (ทดสอบกับ Python 3.14 บน Windows 11)

### 1. Clone และติดตั้งแพ็กเกจ
```bash
git clone https://github.com/fxanlx/ml-project.git
cd ml-project
python -m venv .venv
```
เปิดใช้ virtual environment:
- Windows: `.venv\Scripts\activate`
- macOS / Linux: `source .venv/bin/activate`

แล้วติดตั้งแพ็กเกจ (PyTorch เวอร์ชัน CPU — ไม่ต้องมี GPU):
```bash
pip install -r requirements.txt
```

### 2. รันแอป Image Classification
```bash
python image_classification/app/app.py
```
เปิดเบราว์เซอร์ที่ **http://localhost:7860** (ครั้งแรกจะดาวน์โหลด weights ของ ConvNeXt-Tiny ~110 MB อัตโนมัติ)

### 3. รัน notebook
```bash
jupyter notebook image_classification/notebooks/image_classification.ipynb
```
กด **Run All** — ชุดข้อมูลอยู่ใน repo แล้ว ไม่ต้องดาวน์โหลดเพิ่ม
การรันครั้งแรกใช้เวลา ~25 นาทีบน CPU (ส่วนใหญ่เป็นการสกัดคุณลักษณะภาพ ซึ่งจะถูก cache ไว้ใน `image_classification/features_cache/` ทำให้รันครั้งถัดไปเร็วขึ้นมาก)
เมื่อรันจบจะบันทึก `app/model.joblib` และ `app/examples/` ใหม่

### (ไม่บังคับ) สร้างชุดข้อมูลใหม่จากต้นฉบับ
```bash
python image_classification/scripts/prepare_data.py --raw-dir <โฟลเดอร์สำหรับไฟล์ดาวน์โหลด ~800 MB>
```
ดาวน์โหลดไฟล์ต้นฉบับ ย่อภาพ และแบ่ง train/test ตาม split ทางการ — ได้ผลเหมือนข้อมูลใน `image_classification/data/` ทุกครั้ง

---

## 🚀 การเผยแพร่เว็บแอป
แอป Image Classification เผยแพร่บน **Hugging Face Spaces** (Gradio SDK, CPU basic):
อัปโหลดไฟล์ทั้งหมดใน `image_classification/app/` ขึ้น Space — ไฟล์ `README.md` ในโฟลเดอร์นั้นมีค่าตั้ง Space (SDK, เวอร์ชัน Python) แล้ว

## 📚 อ้างอิง
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs.* IEEE CVPR, 2012. — The Oxford-IIIT Pet Dataset (CC BY-SA 4.0)
- Z. Liu et al. *A ConvNet for the 2020s.* CVPR, 2022. — ConvNeXt (weights จาก torchvision)
- Gradio — https://www.gradio.app
