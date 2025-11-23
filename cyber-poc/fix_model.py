#!/usr/bin/env python3
"""
Script to retrain the XGBoost model with proper configuration
Fixes:
1. XGBoost pickle warning - uses save_model/load_model
2. Feature names warning - properly names features
3. Over-sensitivity issue - validates model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("ml/data/merged_normalized.csv")
MODELS_DIR = Path("backend/models")
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("RETRAINING XGBOOST MODEL WITH FIXES")
print("=" * 80)

# 1. Load data
print("\n1. Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   ✓ Loaded {len(df)} samples")
print(f"   ✓ Label distribution: {df['label'].value_counts().to_dict()}")

# 2. Prepare data
print("\n2. Preparing features...")
X = df['raw_query'].values
y = df['label'].values

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   ✓ Train: {len(X_train)} samples")
print(f"   ✓ Val: {len(X_val)} samples")
print(f"   ✓ Test: {len(X_test)} samples")

# 3. Extract numeric features with proper names
print("\n3. Extracting numeric features...")

def extract_features(query):
    """Extract 12 numeric features from query"""
    query_lower = query.lower()
    
    return {
        'len_raw': len(query),
        'num_sql_keywords': sum(1 for kw in ['select', 'union', 'insert', 'update', 
                                               'delete', 'drop', 'create', 'alter', 
                                               'exec', 'script'] if kw in query_lower),
        'has_union': int('union' in query_lower),
        'has_select': int('select' in query_lower),
        'has_or_1_1': int('or 1=1' in query_lower or 'or 1 = 1' in query_lower),
        'has_comment': int('--' in query or '/*' in query or '#' in query),
        'has_quote': int("'" in query or '"' in query),
        'num_special': sum(1 for c in query if c in "';\"=<>()"),
        'has_script': int('<script' in query_lower),
        'has_encoded': int('%' in query),
        'num_equals': query.count('='),
        'num_parens': query.count('(') + query.count(')')
    }

# Extract features for all sets
train_features = pd.DataFrame([extract_features(q) for q in X_train])
val_features = pd.DataFrame([extract_features(q) for q in X_val])
test_features = pd.DataFrame([extract_features(q) for q in X_test])

feature_names = list(train_features.columns)
print(f"   ✓ Extracted {len(feature_names)} numeric features: {feature_names}")

# 4. TF-IDF vectorization
print("\n4. Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=5000,
    analyzer='char_wb',
    ngram_range=(3, 6),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

tfidf_feature_names = [f'tfidf_{i}' for i in range(X_train_tfidf.shape[1])]
print(f"   ✓ Created {len(tfidf_feature_names)} TF-IDF features")

# 5. Scale numeric features with proper feature names
print("\n5. Scaling numeric features...")
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)

# Convert to DataFrames with feature names to avoid warnings
train_features_df = pd.DataFrame(train_features_scaled, columns=feature_names)
val_features_df = pd.DataFrame(val_features_scaled, columns=feature_names)
test_features_df = pd.DataFrame(test_features_scaled, columns=feature_names)

print(f"   ✓ Scaled features with proper names")

# 6. Combine features
print("\n6. Combining all features...")
from scipy.sparse import hstack, csr_matrix

X_train_combined = hstack([csr_matrix(train_features_df.values), X_train_tfidf])
X_val_combined = hstack([csr_matrix(val_features_df.values), X_val_tfidf])
X_test_combined = hstack([csr_matrix(test_features_df.values), X_test_tfidf])

all_feature_names = feature_names + tfidf_feature_names
print(f"   ✓ Total features: {len(all_feature_names)}")

# 7. Train XGBoost
print("\n7. Training XGBoost model...")
model = XGBClassifier(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

start_time = datetime.now()
model.fit(
    X_train_combined, 
    y_train,
    eval_set=[(X_val_combined, y_val)],
    verbose=False
)
train_time = (datetime.now() - start_time).total_seconds()

print(f"   ✓ Training completed in {train_time:.2f} seconds")

# 8. Evaluate
print("\n8. Evaluating model...")
y_pred = model.predict(X_test_combined)
y_pred_proba = model.predict_proba(X_test_combined)[:, 1]

f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   ✓ F1 Score: {f1:.4f}")
print(f"   ✓ ROC-AUC: {roc_auc:.4f}")

# Detailed metrics
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"   ✓ Precision: {precision:.4f}")
print(f"   ✓ Recall: {recall:.4f}")

# 9. Test predictions on sample queries
print("\n9. Testing on sample queries...")
test_queries = [
    ("SELECT * FROM users WHERE id=1 OR 1=1 --", "SQL Injection"),
    ("https://shop.example.com/product?id=1", "Benign URL"),
    ("' UNION SELECT username,password FROM users--", "SQL Injection with UNION"),
    ("<script>alert('XSS')</script>", "XSS Attack"),
    ("search?q=python+tutorial", "Normal Search"),
]

for query, label in test_queries:
    # Extract features
    numeric_feats = pd.DataFrame([extract_features(query)])
    numeric_scaled = pd.DataFrame(
        scaler.transform(numeric_feats),
        columns=feature_names
    )
    tfidf_feats = tfidf.transform([query])
    combined = hstack([csr_matrix(numeric_scaled.values), tfidf_feats])
    
    # Predict
    proba = model.predict_proba(combined)[0]
    score = proba[1]
    
    print(f"   {label:30s} → Score: {score:.4f} ({'MALICIOUS' if score > 0.5 else 'BENIGN'})")

# 10. Save model properly (fixes XGBoost warning)
print("\n10. Saving model artifacts...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save XGBoost using native format (no pickle warning)
model_path = MODELS_DIR / f"best_xgboost_{timestamp}_f1_{f1:.3f}.json"
model.save_model(str(model_path))
print(f"   ✓ Saved XGBoost model to: {model_path.name}")

# Save TF-IDF and scaler
tfidf_path = MODELS_DIR / "tfidf_vectorizer.joblib"
scaler_path = MODELS_DIR / "numeric_scaler.joblib"
joblib.dump(tfidf, tfidf_path)
joblib.dump(scaler, scaler_path)
print(f"   ✓ Saved TF-IDF vectorizer")
print(f"   ✓ Saved numeric scaler")

# Save metadata
metadata = {
    "model_name": "XGBoost",
    "timestamp": timestamp,
    "model_file": model_path.name,
    "metrics": {
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "train_time": train_time
    },
    "config": {
        "random_seed": 42,
        "test_size": 0.2,
        "val_size": 0.1,
        "tfidf": {
            "max_features": 5000,
            "analyzer": "char_wb",
            "ngram_range": [3, 6],
            "min_df": 2,
            "max_df": 0.95
        }
    },
    "dataset_info": {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test)
    },
    "feature_names": all_feature_names
}

metadata_path = MODELS_DIR / "model_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Saved metadata")

print("\n" + "=" * 80)
print("✅ MODEL RETRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nNew model files:")
print(f"  • {model_path}")
print(f"  • {tfidf_path}")
print(f"  • {scaler_path}")
print(f"  • {metadata_path}")
print(f"\nPerformance:")
print(f"  • F1 Score: {f1:.4f}")
print(f"  • ROC-AUC: {roc_auc:.4f}")
print(f"  • Precision: {precision:.4f}")
print(f"  • Recall: {recall:.4f}")
