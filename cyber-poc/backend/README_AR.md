# Backend API - SQL Injection Detection

## 📁 الملفات المطلوبة / Required Files

```
backend/
├── model_server.py              # FastAPI Server
├── requirements-backend.txt     # Dependencies
└── models/
    ├── Bestmodel.py            # Model Wrapper Class
    ├── best_xgboost_*.json     # Trained Model (JSON format)
    ├── tfidf_vectorizer.joblib # TF-IDF Vectorizer
    ├── numeric_scaler.joblib   # Feature Scaler
    └── model_metadata.json     # Model Metadata
```

**حجم الملفات: ~350 KB فقط**

---

## 🚀 التشغيل السريع / Quick Start

### 1. تثبيت المكتبات / Install Dependencies

```bash
pip install -r backend/requirements-backend.txt
```

أو:

```bash
pip install fastapi uvicorn xgboost scikit-learn pandas numpy joblib scipy pydantic
```

### 2. تشغيل السيرفر / Start Server

```bash
cd backend
uvicorn model_server:app --host 0.0.0.0 --port 8000
```

أو للتطوير مع auto-reload:

```bash
uvicorn model_server:app --host 0.0.0.0 --port 8000 --reload
```

### 3. التحقق من التشغيل / Verify

افتح المتصفح على: http://localhost:8000

API Documentation: http://localhost:8000/docs

---

## 📡 API Endpoints

### 1. كشف SQL Injection بالـ ML Model

**Endpoint:** `POST /infer-ml`

**Request:**
```json
{
  "method": "GET",
  "url": "/users?id=1",
  "params": {"id": "1"},
  "body": null,
  "headers": {}
}
```

**Response:**
```json
{
  "action": "allow",
  "score": 0.008,
  "reason": "Low threat score (0.008). Request appears benign.",
  "confidence": "high",
  "model_version": "20251122_224844",
  "threshold_mode": "balanced",
  "features": {
    "len_raw": 15,
    "num_sql_keywords": 0,
    "has_union": 0,
    ...
  }
}
```

### 2. كشف SQL Injection بالقواعد

**Endpoint:** `POST /infer`

**Request:** نفس الشكل

**Response:** نفس الشكل (بدون ML model metrics)

### 3. معلومات الموديل

**Endpoint:** `GET /ml-model-info`

**Response:**
```json
{
  "model_type": "XGBoost",
  "version": "20251122_224844",
  "metrics": {
    "f1": 0.999,
    "roc_auc": 1.0,
    "precision": 1.0,
    "recall": 0.998,
    "train_time": 6.96
  },
  "threshold_mode": "balanced",
  "thresholds": {
    "challenge": 0.3,
    "block": 0.7
  }
}
```

### 4. فحص صحة السيرفر

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "ml_model_available": true
}
```

---

## 💻 أمثلة الاستخدام / Usage Examples

### Python

```python
import requests

# فحص طلب
response = requests.post(
    "http://localhost:8000/infer-ml",
    json={
        "method": "GET",
        "url": "/users?id=1 OR 1=1--",
        "params": {"id": "1 OR 1=1--"}
    }
)

result = response.json()
print(f"Action: {result['action']}")  # 'block'
print(f"Score: {result['score']}")    # 0.998
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/infer-ml', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        method: 'GET',
        url: '/users?id=1',
        params: {id: '1'}
    })
});

const result = await response.json();
console.log(`Action: ${result.action}`);
```

### cURL

```bash
curl -X POST "http://localhost:8000/infer-ml" \
     -H "Content-Type: application/json" \
     -d '{"method":"GET","url":"/users?id=1","params":{"id":"1"}}'
```

---

## 🔧 الإعدادات / Configuration

### تغيير الـ Threshold Mode

في ملف `Bestmodel.py`:

```python
model = BestModel(threshold_mode="high_security")
# Options: "balanced", "high_security", "high_availability"
```

- **balanced**: `challenge=0.3, block=0.7` (افتراضي)
- **high_security**: `challenge=0.2, block=0.5` (حماية عالية)
- **high_availability**: `challenge=0.5, block=0.9` (توفر عالي)

### تغيير CORS Origins

في ملف `model_server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # تحديد الدومينات المسموحة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 الأداء / Performance

- **دقة الكشف**: 99.9% F1 Score
- **سرعة الاستجابة**: < 50ms
- **حجم الموديل**: ~350 KB
- **استهلاك الذاكرة**: ~100 MB

---

## 🐛 استكشاف الأخطاء / Troubleshooting

### خطأ: "Cannot connect to backend server"

**الحل:**
```bash
cd backend
uvicorn model_server:app --host 0.0.0.0 --port 8000
```

### خطأ: "No module named 'xgboost'"

**الحل:**
```bash
pip install -r backend/requirements-backend.txt
```

### خطأ: "No XGBoost model found"

**الحل:** تأكد من وجود الملفات في `backend/models/`

---

## 🔐 الأمان / Security

- ✅ التحقق من جميع المدخلات
- ✅ معالجة الأخطاء بشكل آمن
- ✅ تسجيل جميع الطلبات في `logs/requests.jl`
- ✅ CORS configuration قابل للتخصيص
- ⚠️ للإنتاج: أضف Authentication و Rate Limiting

---

## 📝 التوثيق الكامل / Full Documentation

افتح API Documentation: http://localhost:8000/docs

أو Redoc: http://localhost:8000/redoc

---

## 🤝 التكامل مع Flask/Django

راجع ملف `integration_example.py` لأمثلة كاملة على:
- استخدام Middleware
- Batch checking
- Error handling

---

## 📞 الدعم / Support

للمزيد من المعلومات، راجع:
- `FINAL_FULL_PROJECT_REPORT.md` - تقرير المشروع الكامل
- `integration_example.py` - أمثلة الاستخدام
- `deliverables/` - جميع التقارير والوثائق

---

**المشروع جاهز للاستخدام! 🚀**
