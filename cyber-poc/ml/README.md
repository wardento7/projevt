# ML Training & Model Deployment

This directory contains all machine learning artifacts for the SQL Injection Detection system.

## 📁 Directory Structure

```
ml/
├── notebooks/
│   ├── train_compare_models.ipynb    # Jupyter notebook (Colab-ready)
│   └── train_compare_models.py       # Python script version
├── data/
│   ├── merged.csv                     # Cleaned & merged training data
│   └── merged.jl                      # JSONLines format
├── models/
│   ├── best_xgboost_*.joblib         # Best trained model
│   ├── tfidf_vectorizer.joblib       # TF-IDF feature extractor
│   ├── numeric_scaler.joblib         # Numeric feature scaler
│   └── model_metadata.json           # Model configuration & metrics
├── reports/
│   ├── model_card.md                 # Comprehensive model documentation
│   ├── model_comparison.csv          # All models performance comparison
│   ├── dataset_report.json           # Dataset statistics
│   ├── thresholds.json               # Recommended decision thresholds
│   └── label_distribution.png        # Visualization
└── scripts/
    └── train_compare_models.py       # Standalone training script
```

## 🚀 Quick Start

### Option 1: Run Training Script

```bash
# Activate virtual environment
source .venv/bin/activate

# Quick training (reduced grid search)
python ml/scripts/train_compare_models.py --quick

# Full training (comprehensive grid search)
python ml/scripts/train_compare_models.py
```

### Option 2: Use Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook ml/notebooks/train_compare_models.ipynb

# Or upload to Google Colab
# File → Upload notebook → ml/notebooks/train_compare_models.ipynb
```

## 📊 Model Performance

| Model | F1 Score | Precision | Recall | ROC AUC | Training Time |
|-------|----------|-----------|--------|---------|---------------|
| **XGBoost** ⭐ | **0.9980** | **1.0000** | **0.9960** | **1.0000** | 11.78s |
| Logistic Regression | 0.9959 | 1.0000 | 0.9919 | 1.0000 | 1.55s |
| Random Forest | 0.9908 | 0.9979 | 0.9838 | 0.9999 | 2.76s |

**Test Set Performance (Final):**
- Precision: 100%
- Recall: 99.8%
- F1 Score: 99.9%
- ROC AUC: 100%

## 🎯 Decision Thresholds

### Balanced (Recommended)
- **ALLOW**: score < 0.3
- **CHALLENGE**: 0.3 ≤ score < 0.7  (Show CAPTCHA, log for review)
- **BLOCK**: score ≥ 0.7

### High Security
- **ALLOW**: score < 0.2
- **CHALLENGE**: 0.2 ≤ score < 0.5
- **BLOCK**: score ≥ 0.5

### High Availability
- **ALLOW**: score < 0.5
- **CHALLENGE**: 0.5 ≤ score < 0.9
- **BLOCK**: score ≥ 0.9

See `reports/thresholds.json` for detailed metrics.

## 🔧 Model Features

### TF-IDF Features (5,000 features)
- Character-level n-grams (3-6 chars)
- Captures attack patterns at character level
- Robust to typos and variations

### Numeric Features (12 features)
- `len_raw`: Query length
- `count_single_quote`: Single quote count
- `count_double_quote`: Double quote count
- `count_dashes`: SQL comment indicator
- `count_semicolon`: Stacked query indicator
- `count_comment`: Comment patterns (/*,*/,#)
- `num_sql_keywords`: SQL keyword count
- `has_union`: UNION keyword present
- `has_or_equals`: OR-equals pattern
- `has_sleep`: Time-based attack indicator
- `has_comments`: Comment patterns present
- `url_encoded`: URL encoding detected

## 📦 Model Integration

### Backend Integration

Copy model files to backend:

```bash
# Create backend models directory
mkdir -p backend/models

# Copy model artifacts
cp ml/models/best_xgboost_*.joblib backend/models/
cp ml/models/tfidf_vectorizer.joblib backend/models/
cp ml/models/numeric_scaler.joblib backend/models/
cp ml/models/model_metadata.json backend/models/
```

### Usage Example

```python
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Load model artifacts
model = joblib.load('backend/models/best_xgboost_*.joblib')
tfidf = joblib.load('backend/models/tfidf_vectorizer.joblib')
scaler = joblib.load('backend/models/numeric_scaler.joblib')

# Feature extraction function
def extract_numeric_features(query):
    sql_keywords = ['UNION', 'SELECT', 'INSERT', ...]
    q_upper = query.upper()
    
    return {
        'len_raw': len(q_upper),
        'count_single_quote': q_upper.count("'"),
        # ... other features
    }

# Prediction function
def predict(raw_query):
    # Extract TF-IDF features
    tfidf_features = tfidf.transform([raw_query])
    
    # Extract numeric features
    numeric_features = pd.DataFrame([extract_numeric_features(raw_query)])
    numeric_scaled = scaler.transform(numeric_features)
    
    # Combine features
    X = hstack([tfidf_features, csr_matrix(numeric_scaled)])
    
    # Predict
    score = model.predict_proba(X)[0, 1]
    
    # Determine action
    if score < 0.3:
        action = 'allow'
    elif score < 0.7:
        action = 'challenge'
    else:
        action = 'block'
    
    return {'score': score, 'action': action}

# Example usage
result = predict("1' OR '1'='1")
print(result)  # {'score': 0.98, 'action': 'block'}
```

## 📚 Documentation

- **Model Card:** `reports/model_card.md` - Comprehensive model documentation
- **Dataset Report:** `reports/dataset_report.json` - Training data statistics
- **Thresholds:** `reports/thresholds.json` - Decision threshold recommendations
- **Comparison:** `reports/model_comparison.csv` - All models metrics

## 🔄 Retraining

### When to Retrain
- **Quarterly:** Scheduled retraining with new attack samples
- **On-Demand:** After security incidents or new attack patterns discovered
- **Performance Drop:** If metrics degrade >5%

### Retraining Steps

1. **Collect new data:**
   ```bash
   # Add new attack samples to data/ directory
   cp new_attacks.csv data/
   ```

2. **Run training:**
   ```bash
   python ml/scripts/train_compare_models.py
   ```

3. **Evaluate new model:**
   ```bash
   # Compare with previous version
   diff ml/reports/model_comparison.csv ml/reports/model_comparison_old.csv
   ```

4. **Deploy if better:**
   ```bash
   # Copy to backend
   cp ml/models/best_xgboost_*.joblib backend/models/
   ```

## 🐛 Troubleshooting

### Memory Issues
If training fails due to memory:

```bash
# Reduce TF-IDF features
# Edit CONFIG in train_compare_models.py:
'tfidf': {
    'max_features': 2000,  # Reduced from 5000
}
```

### Slow Training
Use quick mode for testing:

```bash
python ml/scripts/train_compare_models.py --quick
```

### Missing Dependencies
Install all ML dependencies:

```bash
pip install scikit-learn xgboost pandas numpy pyarrow shap joblib matplotlib seaborn
```

## 📈 Monitoring

Monitor these metrics in production:

1. **Daily:**
   - Prediction distribution (allow/challenge/block)
   - Average scores
   - Challenge rate (should be <5%)

2. **Weekly:**
   - False positive reports
   - Feature drift
   - Score distribution changes

3. **Monthly:**
   - Precision/Recall on labeled incidents
   - Model vs rules comparison
   - Retrain if needed

## 🔐 Security Notes

- All training data anonymized (IPs redacted)
- Model files contain no sensitive data
- Only attack patterns learned, not actual user data
- Logs minimal information for debugging

## 📞 Support

For issues or questions:
- Check `deliverables/summary.txt` for training logs
- Review `ml/reports/model_card.md` for detailed docs
- File issues in project repository

---

**Last Updated:** November 3, 2025  
**Model Version:** 1.0  
**Status:** Production Ready ✅
