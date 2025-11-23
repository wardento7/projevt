# 📦 Package للباك اند - SQL Injection Detection API

## 🎯 الهدف
حزمة كاملة لإضافة نظام كشف SQL Injection إلى أي Backend

---

## 📁 الملفات المطلوبة (7 ملفات فقط - ~400 KB)

```
backend_package/
├── model_server.py              # FastAPI Server (15 KB)
├── requirements.txt             # Dependencies list
├── README.md                    # هذا الملف
├── integration_examples.py      # أمثلة الاستخدام
└── models/
    ├── Bestmodel.py            # Model Wrapper (16 KB)
    ├── best_xgboost_20251122_224844_f1_0.999.json  # ML Model (160 KB)
    ├── tfidf_vectorizer.joblib  # TF-IDF (172 KB)
    ├── numeric_scaler.joblib    # Scaler (4 KB)
    └── model_metadata.json      # Metadata (88 KB)
```

**الحجم الإجمالي: ~455 KB**

---

## ⚡ التشغيل السريع (3 خطوات)

### 1. التثبيت
```bash
pip install -r requirements.txt
```

### 2. التشغيل
```bash
uvicorn model_server:app --host 0.0.0.0 --port 8000
```

### 3. الاختبار
افتح: http://localhost:8000/docs

---

## 📡 API Endpoints

### 1️⃣ كشف بالـ ML Model (موصى به)
```http
POST /infer-ml
Content-Type: application/json

{
  "method": "GET",
  "url": "/users?id=1",
  "params": {"id": "1"}
}
```

**Response:**
```json
{
  "action": "allow|challenge|block",
  "score": 0.008,
  "reason": "Low threat score...",
  "confidence": "high",
  "model_version": "20251122_224844"
}
```

### 2️⃣ كشف بالقواعد (أسرع)
```http
POST /infer
```
نفس الـ Request والـ Response

### 3️⃣ معلومات الموديل
```http
GET /ml-model-info
```

### 4️⃣ فحص صحة السيرفر
```http
GET /health
```

---

## 💻 أمثلة التكامل

### Python (Requests)
```python
import requests

response = requests.post(
    "http://localhost:8000/infer-ml",
    json={
        "method": "GET",
        "url": "/users?id=1 OR 1=1--",
        "params": {"id": "1 OR 1=1--"}
    }
)

result = response.json()
if result['action'] == 'block':
    return "Access Denied", 403
```

### Python (Flask Middleware)
```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
API_URL = "http://localhost:8000/infer-ml"

@app.before_request
def check_sql_injection():
    result = requests.post(API_URL, json={
        "method": request.method,
        "url": request.path,
        "params": dict(request.args),
        "body": request.get_data(as_text=True)
    }).json()
    
    if result['action'] == 'block':
        return jsonify({"error": "Malicious request detected"}), 403
```

### Python (Django Middleware)
```python
import requests

class SQLInjectionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.api_url = "http://localhost:8000/infer-ml"
    
    def __call__(self, request):
        result = requests.post(self.api_url, json={
            "method": request.method,
            "url": request.path,
            "params": dict(request.GET),
        }).json()
        
        if result['action'] == 'block':
            return HttpResponse("Access Denied", status=403)
        
        return self.get_response(request)
```

### Node.js (Express)
```javascript
const express = require('express');
const axios = require('axios');

const app = express();
const API_URL = 'http://localhost:8000/infer-ml';

// Middleware
app.use(async (req, res, next) => {
    try {
        const response = await axios.post(API_URL, {
            method: req.method,
            url: req.path,
            params: req.query,
            body: JSON.stringify(req.body)
        });
        
        if (response.data.action === 'block') {
            return res.status(403).json({error: 'Malicious request detected'});
        }
        
        next();
    } catch (error) {
        next(); // في حالة خطأ، السماح بالمرور
    }
});
```

### PHP
```php
<?php
function checkSQLInjection($method, $url, $params) {
    $data = json_encode([
        'method' => $method,
        'url' => $url,
        'params' => $params
    ]);
    
    $ch = curl_init('http://localhost:8000/infer-ml');
    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = json_decode(curl_exec($ch), true);
    curl_close($ch);
    
    return $response['action'];
}

// Usage
$action = checkSQLInjection($_SERVER['REQUEST_METHOD'], $_SERVER['REQUEST_URI'], $_GET);
if ($action === 'block') {
    http_response_code(403);
    die('Access Denied');
}
?>
```

---

## 🔧 التخصيص

### تغيير مستوى الحساسية

في ملف `model_server.py` أو `Bestmodel.py`:

```python
# Option 1: High Security (حماية عالية - أقل تساهل)
model = BestModel(threshold_mode="high_security")

# Option 2: Balanced (متوازن - افتراضي)
model = BestModel(threshold_mode="balanced")

# Option 3: High Availability (توفر عالي - أكثر تساهل)
model = BestModel(threshold_mode="high_availability")
```

| Mode | Challenge Threshold | Block Threshold | Use Case |
|------|-------------------|----------------|----------|
| high_security | 0.2 | 0.5 | Banking, Admin Panels |
| balanced | 0.3 | 0.7 | E-commerce, APIs |
| high_availability | 0.5 | 0.9 | Public Websites |

### تغيير CORS

في `model_server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # حدد الدومينات
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## 📊 الأداء والمواصفات

| Metric | Value |
|--------|-------|
| **دقة الكشف** | 99.9% F1 Score |
| **Precision** | 100% |
| **Recall** | 99.8% |
| **سرعة الاستجابة** | < 50ms |
| **استهلاك الذاكرة** | ~100 MB |
| **حجم الملفات** | ~455 KB |

---

## 🐛 استكشاف الأخطاء

### المشكلة: Cannot connect to server
**الحل:**
```bash
uvicorn model_server:app --host 0.0.0.0 --port 8000
```

### المشكلة: ModuleNotFoundError
**الحل:**
```bash
pip install -r requirements.txt
```

### المشكلة: Model not found
**الحل:** تأكد من وجود مجلد `models/` مع جميع الملفات

### المشكلة: CORS error
**الحل:** عدل `allow_origins` في `model_server.py`

---

## 🔐 ملاحظات الأمان

### ✅ ما يوفره النظام:
- كشف SQL Injection بدقة 99.9%
- تسجيل جميع الطلبات في `logs/requests.jl`
- معالجة آمنة للأخطاء
- Validation للمدخلات

### ⚠️ ما يجب إضافته للـ Production:
- **Authentication/Authorization** - أضف API Keys أو JWT
- **Rate Limiting** - استخدم `slowapi` أو `redis`
- **HTTPS** - استخدم شهادة SSL
- **Monitoring** - أضف Prometheus/Grafana
- **Load Balancing** - للتوزيع على عدة خوادم

---

## 📝 مثال Deployment على Production

### باستخدام Docker

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
docker build -t sql-injection-api .
docker run -p 8000:8000 sql-injection-api
```

### باستخدام Gunicorn (Production Server)

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker model_server:app --bind 0.0.0.0:8000
```

### باستخدام Nginx (Reverse Proxy)

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📞 الدعم والوثائق

- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Alternative Docs:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health
- **Model Info:** http://localhost:8000/ml-model-info

---

## 🧪 اختبار النظام

### اختبار سريع
```bash
# Test 1: Benign request
curl -X POST "http://localhost:8000/infer-ml" \
  -H "Content-Type: application/json" \
  -d '{"method":"GET","url":"/users?id=1","params":{"id":"1"}}'

# Test 2: SQL Injection
curl -X POST "http://localhost:8000/infer-ml" \
  -H "Content-Type: application/json" \
  -d '{"method":"GET","url":"/users?id=1 OR 1=1--","params":{"id":"1 OR 1=1--"}}'
```

### اختبار شامل
```bash
python integration_examples.py
```

---

## 📦 البدائل والتوسعات

### إذا أردت استخدامه Standalone (بدون API)
```python
from models.Bestmodel import BestModel

model = BestModel()
result = model.predict("SELECT * FROM users WHERE id=1 OR 1=1--")
print(result['action'])  # 'block'
```

### إذا أردت إضافة أنواع هجمات أخرى
النظام حالياً متدرب على:
- ✅ SQL Injection
- ⚠️ XSS (دقة متوسطة)
- ❌ Path Traversal (غير مدرب)
- ❌ Command Injection (غير مدرب)

لإضافة أنواع جديدة، ستحتاج إعادة تدريب الموديل.

---

## 📄 الملفات الإضافية (اختيارية)

الحزمة تحتوي أيضاً على:
- `integration_examples.py` - أمثلة كاملة للاستخدام
- `FIXES_APPLIED.md` - سجل التحديثات والإصلاحات
- `backend/README_AR.md` - دليل تفصيلي بالعربي

---

## ✅ Checklist قبل الإنتاج

- [ ] اختبار جميع الـ Endpoints
- [ ] إضافة Authentication
- [ ] إضافة Rate Limiting
- [ ] تفعيل HTTPS
- [ ] إعداد Monitoring
- [ ] إعداد Backup للـ logs
- [ ] إعداد Auto-restart (systemd/supervisor)
- [ ] اختبار الأداء تحت الضغط (Load Testing)

---

**النظام جاهز للاستخدام الفوري! 🚀**

**للاستفسارات:** راجع ملف `integration_examples.py` للأمثلة الكاملة
