# Dementia Detection Using CNN Model

نسخه نهایی پروژه برای طبقه‌بندی مراحل دمانس/آلزایمر از روی تصاویر MRI.

## فایل نهایی
- `final_dementia_cnn.py` ← نسخه قابل تحویل (production-style)
- `DementiaDetection.ipynb` ← نسخه نوت‌بوکی

## قابلیت‌های نسخه نهایی
- Pipeline کامل train/validation/test
- Data augmentation + normalization
- CNN بهینه‌تر با BatchNorm, Dropout, L2
- Callbackهای حرفه‌ای: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- گزارش کامل: learning curves, confusion matrix, classification report
- inference روی تصویر جدید
- ذخیره خروجی‌ها در پوشه `artifacts/`

## ساختار دیتاست
```text
data/train/
  Non Demented/
  Mild Dementia/
  Moderate Dementia/
  Very mild Dementia/
```

## اجرا
```bash
python final_dementia_cnn.py --data-dir data/train --epochs 30
```

پیش‌بینی روی یک تصویر:
```bash
python final_dementia_cnn.py \
  --data-dir data/train \
  --weights artifacts/best_dementia_cnn.keras \
  --predict data/new_cases/sample_1.jpg
```
