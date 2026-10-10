# Dataset: The Oxford-IIIT Pet Dataset

- **แหล่งที่มา:** https://www.robots.ox.ac.uk/~vgg/data/pets/ (Visual Geometry Group, University of Oxford)
- **License:** CC BY-SA 4.0 (ดู [LICENSE.txt](LICENSE.txt))
- **Citation:** O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs", IEEE CVPR 2012

## เนื้อหา
- ภาพสัตว์เลี้ยง **7,349 ภาพ** จำนวน **37 สายพันธุ์** (แมว 12, สุนัข 25)
- การแบ่งข้อมูลใช้ **split ทางการของชุดข้อมูล** (`trainval.txt` / `test.txt`)
  - `train/` — 3,680 ภาพ (ประมาณ 100 ภาพต่อสายพันธุ์)
  - `test/` — 3,669 ภาพ (ประมาณ 100 ภาพต่อสายพันธุ์) — **ใช้ประเมินผลครั้งสุดท้ายเท่านั้น** ห้ามใช้ฝึกหรือปรับเลือกแบบจำลอง
- ชื่อสายพันธุ์แมวขึ้นต้นด้วยตัวพิมพ์ใหญ่ (เช่น `Persian`) ส่วนสุนัขเป็นตัวพิมพ์เล็ก (เช่น `pug`) ตามชุดข้อมูลต้นฉบับ

## โครงสร้าง
```
data/
├── train/<label>/<image>.jpg
├── test/<label>/<image>.jpg
├── labels.csv      # ระบุ label ของทุกภาพ
└── LICENSE.txt
```

`labels.csv` มีคอลัมน์:

| คอลัมน์ | ความหมาย |
|---|---|
| `filepath` | path ของภาพเทียบกับโฟลเดอร์ `data/` |
| `label` | ชื่อสายพันธุ์ (37 คลาส) |
| `class_id` | รหัสคลาส 1–37 ตามชุดข้อมูลต้นฉบับ |
| `species` | `cat` หรือ `dog` |
| `split` | `train` หรือ `test` |

## การเตรียมข้อมูล
ภาพถูกแปลงเป็น RGB และย่อให้ด้านยาวสุดไม่เกิน 320 px เพื่อให้เก็บบน GitHub ได้ (รวมประมาณ 160 MB)
สร้างซ้ำได้ด้วยคำสั่ง:

```bash
python image_classification/scripts/prepare_data.py --raw-dir <โฟลเดอร์สำหรับไฟล์ดาวน์โหลด>
```
