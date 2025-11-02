# Cyber Security Intelligent Threat Mitigation - POC

## ⚠️ تحذير قانوني صارم

**هذا المشروع للأغراض التعليمية والبحثية فقط.**

- **لا تستخدم** هذه الأدوات على أي نظام أو شبكة لا تملك إذنًا صريحًا كتابيًا لاختبارها.
- الاستخدام غير المصرح به لأدوات الاختراق الأمني يُعتبر جريمة في معظم الدول.
- المستهدف الافتراضي المسموح: `localhost`, `127.0.0.1` فقط.
- المطورون والمستخدمون مسؤولون قانونيًا عن أي إساءة استخدام.

## نظرة عامة

هذا المشروع يوفر:
1. **Reconnaissance Tools**: مسح الأمان الأساسي (nmap, ZAP, sqlmap)
2. **Synthetic Dataset Generator**: توليد بيانات ضخمة (1M+ records) لتدريب نماذج ML
3. **Rule-based Detection Backend**: API سريع للكشف عن هجمات SQL Injection
4. **Logging & Analytics**: تسجيل شامل لكل الطلبات والقرارات

## المتطلبات

- Python 3.8+
- (اختياري) nmap
- (اختياري) OWASP ZAP
- (اختياري) sqlmap

## التثبيت والإعداد

```bash
# 1. إنشاء البيئة وتثبيت المتطلبات
make setup

# 2. توليد Dataset ضخم (افتراضي: 700k benign + 300k malicious)
make gen-data

# 3. تشغيل عمليات الـ reconnaissance (اختياري)
make recon TARGET=http://localhost:8000

# 4. تشغيل السيرفر
make run-server

# 5. اختبار النظام
make test

# 6. تنظيف الملفات المؤقتة
make clean
```

## متغيرات التكوين

يمكنك تخصيص المتغيرات التالية:

```bash
TARGET=http://localhost:8000       # الهدف للمسح الأمني
NUM_BENIGN=700000                  # عدد السجلات البريئة
NUM_MALICIOUS=300000               # عدد السجلات الخبيثة
AUGMENT=5                          # مضاعف التنويع
CHUNK_SIZE=10000                   # حجم الدفعة للكتابة
BOT_IP_RATIO=0.1                   # نسبة IPs الآلية
```

## بنية المشروع

```
cyber-poc/
├── recon/                         # أدوات الاستطلاع الأمني
│   ├── run_nmap.py               # ماسح المنافذ
│   ├── run_zap.sh                # ماسح الثغرات
│   ├── run_sqlmap.sh             # كاشف SQL Injection
│   ├── generate_payloads.py      # مولد الـ payloads
│   └── payloads.json             # قاعدة بيانات الـ payloads
├── data/                          # البيانات والمولدات
│   ├── generate_synthetic_dataset.py
│   ├── benign_queries.txt
│   ├── dataset.csv               # Dataset الرئيسي
│   ├── dataset.jl                # نفس البيانات بصيغة JSONLines
│   └── generation_report.json    # تقرير التوليد
├── backend/                       # السيرفر الخلفي
│   ├── model_server.py           # FastAPI server
│   └── requirements-backend.txt
├── logs/                          # سجلات التشغيل
│   └── requests.jl               # سجل كل الطلبات
├── deliverables/                  # المخرجات النهائية
│   └── summary.txt               # ملخص شامل
├── Makefile                       # أوامر التشغيل الآلية
└── README.md                      # هذا الملف
```

## وصف Dataset

كل سجل في Dataset يحتوي على:

| Field | Description |
|-------|-------------|
| `timestamp` | وقت الطلب (UTC) |
| `source_ip` | IP المصدر |
| `method` | HTTP method (GET/POST/etc) |
| `url` | المسار المطلوب |
| `params` | Query parameters |
| `body` | Request body |
| `headers` | HTTP headers |
| `raw_query` | التمثيل الخام للطلب |
| `is_malicious` | 0=benign, 1=malicious |
| `attack_type` | نوع الهجوم (إن وُجد) |
| `insertion_point` | موقع الإدراج |
| `mutation_type` | نوع التشويش |
| `orig_payload_id` | معرف الـ payload الأصلي |
| `difficulty_score` | درجة صعوبة الكشف (0.0-1.0) |

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Inference
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "method": "GET",
    "url": "/search",
    "params": {"q": "test OR 1=1"},
    "body": null,
    "raw_query": "GET /search?q=test+OR+1=1",
    "source_ip": "192.168.1.100"
  }'
```

Response:
```json
{
  "score": 0.95,
  "action": "block",
  "reason": "rule",
  "matched_rules": ["OR 1=1"],
  "features": {
    "len_raw": 28,
    "count_susp_chars": 2,
    "num_sql_keywords": 1
  }
}
```

## الخطوات القادمة (ML Integration)

المشروع الحالي يوفر:
- ✅ Infrastructure كامل
- ✅ Dataset عالي الجودة (1M+ records)
- ✅ Rule-based detection يعمل
- ✅ Logging شامل

لإضافة ML لاحقًا:
1. إنشاء `ml/feature_extraction.py`
2. إنشاء `ml/train_model.py` (استخدام RandomForest/XGBoost/Neural Networks)
3. إنشاء `ml/eval_model.py`
4. تحديث `backend/model_server.py` لاستخدام النموذج المدرب

## المساهمة

هذا مشروع بحثي. الاستخدام الأخلاقي والمسؤول مطلوب.

## License

MIT License - للأغراض التعليمية فقط
