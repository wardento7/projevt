# SQL Injection Detection Model Card

## Model Information

**Model Name:** XGBoost Classifier  
**Version:** 1.0  
**Date:** November 3, 2025  
**Framework:** XGBoost 3.1.1  
**Task:** Binary Classification (SQL Injection Detection)  

## Model Description

This model detects SQL injection attempts in HTTP requests using machine learning.  It combines TF-IDF character n-gram features with hand-engineered SQL pattern features to achieve high precision and recall.

### Architecture
- **Base Model:** XGBoost Classifier (Gradient Boosting)
- **Feature Engineering:**
  - TF-IDF vectorization (char-level, ngrams 3-6, max_features=5000)
  - 12 hand-crafted numeric features (SQL keyword counts, suspicious character counts, etc.)
- **Total Features:** 5,012
- **Hyperparameters:**
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 6
  - subsample: 0.8
  - tree_method: hist
  - scale_pos_weight: auto (class-balanced)

## Training Data

**Dataset Sources:**
- `data/dataset.csv` (12,636 rows)
- `FINAL_DATASET_FOR_AI_TEAM_v3.csv` (30,919 rows - filtered to 12,636 after deduplication)

**Final Training Set:**
- Total samples: 12,636
- Train: 10,108 (80%)
- Validation: 1,264 (10%)
- Test: 1,264 (10%)

**Class Distribution:**
- Benign: 7,700 (60.9%)
- Malicious: 4,936 (39.1%)
- Imbalance ratio: 1.56:1

**Data Quality:**
- Deduplicated (max 3 duplicates per unique query)
- Empty queries removed
- IPs anonymized

## Performance Metrics

### Validation Set
- **Precision:** 1.0000 (100%)
- **Recall:** 0.9960 (99.60%)
- **F1 Score:** 0.9980 (99.80%)
- **ROC AUC:** 1.0000 (100%)

### Test Set (Final Evaluation)
- **Precision:** 1.0000 (100%)
- **Recall:** 0.9980 (99.80%)
- **F1 Score:** 0.9990 (99.90%)
- **ROC AUC:** 1.0000 (100%)

### Training Time
- 11.78 seconds (quick mode with reduced grid search)
- Full training estimated: ~30-60 minutes

### Inference Performance
- Estimated: ~1-5ms per request (with feature extraction)
- Batch processing: Highly efficient with vectorized operations

## Model Comparison

| Model | F1 Score | Precision | Recall | ROC AUC | Training Time |
|-------|----------|-----------|--------|---------|---------------|
| **XGBoost** (selected) | **0.9980** | **1.0000** | **0.9960** | **1.0000** | 11.78s |
| Logistic Regression | 0.9959 | 1.0000 | 0.9919 | 1.0000 | 1.55s |
| Random Forest | 0.9908 | 0.9979 | 0.9838 | 0.9999 | 2.76s |

**Selection Rationale:** XGBoost achieved the highest F1 score and recall, critical for minimizing false negatives (missed attacks) while maintaining perfect precision.

## Intended Use

### Primary Use Case
Real-time SQL injection detection in web application firewalls (WAF) and API gateways.

### Intended Users
- Security engineers
- DevOps teams
- Incident response teams
- Automated security systems

### Scope
- **In-Scope:** Detection of SQL injection patterns in:
  - GET/POST parameters
  - URL paths
  - Request headers
  - Request bodies (JSON/form data)
  - Cookies

- **Out-of-Scope:**
  - Other web vulnerabilities (XSS, CSRF, etc.)
  - NoSQL injection (requires separate training)
  - Second-order SQL injection (requires context awareness)
  - Binary payloads

## Recommendations

### Deployment Thresholds

**Score Ranges:**
- `score < 0.3`: **ALLOW** - Benign traffic
- `0.3 ≤ score < 0.7`: **CHALLENGE** - Suspicious, require additional verification (CAPTCHA, MFA)
- `score ≥ 0.7`: **BLOCK** - High confidence malicious

**Tuning Guidance:**
- **High Security (low false negative tolerance):** Lower block threshold to 0.5
- **High Availability (low false positive tolerance):** Raise block threshold to 0.9
- **Balanced (recommended):** Use 0.7 threshold

### Integration Steps

1. **Load model bundle:**
   ```python
   import joblib
   model = joblib.load('best_xgboost_*.joblib')
   tfidf = joblib.load('tfidf_vectorizer.joblib')
   scaler = joblib.load('numeric_scaler.joblib')
   ```

2. **Feature extraction:**
   ```python
   # Extract TF-IDF features
   tfidf_features = tfidf.transform([raw_query])
   
   # Extract numeric features
   numeric_features = extract_numeric_features([raw_query])
   numeric_scaled = scaler.transform(numeric_features)
   
   # Combine
   X = hstack([tfidf_features, csr_matrix(numeric_scaled)])
   ```

3. **Prediction:**
   ```python
   score = model.predict_proba(X)[0, 1]  # Probability of malicious
   action = 'block' if score >= 0.7 else 'allow'
   ```

## Limitations & Risks

### Known Limitations

1. **Novel Attack Patterns:**
   - Model trained on known attack patterns; may miss completely novel injection techniques
   - **Mitigation:** Regular retraining with new attack samples, ensemble with rule-based detection

2. **Obfuscated Payloads:**
   - Heavily obfuscated or encoded payloads might evade detection if encodings not seen during training
   - **Mitigation:** Include diverse encoding variants in training data, pre-decode common formats

3. **Context-Insensitive:**
   - Model evaluates each request independently without application context
   - **Mitigation:** Combine with stateful analysis for multi-step attacks

4. **False Positives:**
   - Legitimate queries with SQL-like syntax (e.g., technical documentation) might be flagged
   - **Mitigation:** Whitelist known benign patterns, use "challenge" action for borderline cases

5. **Language Bias:**
   - Training data primarily English; non-English SQL keywords might have lower detection
   - **Mitigation:** Include multilingual SQL variants in training

### Security Risks

1. **Adversarial Evasion:**
   - Attackers may craft payloads specifically to evade ML detection
   - **Mitigation:** Use ensemble approach (ML + rules), adversarial training

2. **Model Poisoning:**
   - If attacker can influence training data, model integrity compromised
   - **Mitigation:** Secure training pipeline, validate all training samples

3. **Performance Degradation:**
   - Model may degrade over time as attack patterns evolve
   - **Mitigation:** Continuous monitoring, scheduled retraining (quarterly recommended)

## Ethical Considerations

- **Privacy:** Model does not store or log sensitive user data; IPs anonymized in training
- **Fairness:** No demographic features used; applies equally to all users
- **Transparency:** Open documentation of features and decision thresholds
- **Accountability:** All predictions logged for audit; human review available for appeals

## Maintenance & Monitoring

### Monitoring Metrics
- **Daily:**
  - Prediction distribution (benign vs malicious)
  - Average prediction scores
  - False positive rate (from user reports)
  
- **Weekly:**
  - Feature drift detection
  - Model performance on recent data
  
- **Monthly:**
  - Precision/Recall on labeled incidents
  - Compare with rule-based baseline

### Retraining Schedule
- **Quarterly:** Scheduled retrain with accumulated new attack samples
- **On-Demand:** After major security incidents or new attack campaigns
- **Triggers:** Performance degradation >5%, new attack type discovered

### Version Control
- All model versions stored with:
  - Training data hash
  - Hyperparameters
  - Metrics snapshot
  - Deployment timestamp

## References

- **Training Dataset:** Synthetic + real attack samples
- **Framework:** XGBoost 3.1.1, scikit-learn 1.7.2
- **Feature Engineering:** TF-IDF (scikit-learn), custom SQL pattern detection
- **Evaluation:** Stratified K-fold cross-validation (5 folds)

## Contact & Support

**Model Maintainer:** Cyber Security Team  
**Last Updated:** November 3, 2025  
**Review Cycle:** Quarterly  
**Issues/Feedback:** File in project repository

---

**Disclaimer:** This model is designed for authorized security testing and protection purposes only. Unauthorized use to attack systems is illegal and unethical.
