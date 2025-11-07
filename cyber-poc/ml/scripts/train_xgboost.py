"""
SQL Injection Detection - XGBoost Training Script
Cybersecurity Intelligent Threat Mitigation Project

This script trains an XGBoost model for SQL injection detection.
All non-XGBoost algorithms have been removed for production deployment.

Usage:
    python train_xgboost.py [--quick]
    
Options:
    --quick: Run reduced training (fewer iterations, smaller grids)
"""

import os
import sys
import json
import time
import argparse
import hashlib
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========================================================================
# CONFIGURATION
# ========================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
ML_DATA_DIR = PROJECT_ROOT / 'ml' / 'data'
ML_MODELS_DIR = PROJECT_ROOT / 'ml' / 'models'
ML_REPORTS_DIR = PROJECT_ROOT / 'ml' / 'reports'
DELIVERABLES_DIR = PROJECT_ROOT / 'deliverables'

# Create directories
for dir_path in [ML_DATA_DIR, ML_MODELS_DIR, ML_REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'random_seed': RANDOM_SEED,
    'test_size': 0.1,
    'val_size': 0.1,
    'max_duplicates': 3,
    'tfidf': {
        'max_features': 5000,
        'analyzer': 'char_wb',
        'ngram_range': (3, 6),
        'min_df': 2,
        'max_df': 0.95
    }
}

# ========================================================================
# DATA LOADING AND PREPROCESSING
# ========================================================================

def load_and_merge_datasets():
    """Load all available datasets and merge"""
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    datasets = []
    
    # Load data/dataset.csv
    dataset_path = DATA_DIR / 'dataset.csv'
    if dataset_path.exists():
        print(f"\nLoading {dataset_path}...")
        df1 = pd.read_csv(dataset_path)
        df1['source_file'] = 'dataset.csv'
        datasets.append(df1)
        print(f"  ✓ Loaded {len(df1):,} rows")
    
    # Load FINAL_DATASET
    final_dataset_path = PROJECT_ROOT / 'FINAL_DATASET_FOR_AI_TEAM_v3 (1).csv'
    if final_dataset_path.exists():
        print(f"\nLoading {final_dataset_path}...")
        df2 = pd.read_csv(final_dataset_path)
        df2['source_file'] = 'FINAL_DATASET_FOR_AI_TEAM_v3.csv'
        datasets.append(df2)
        print(f"  ✓ Loaded {len(df2):,} rows")
    
    # Merge
    df = pd.concat(datasets, ignore_index=True)
    print(f"\n✓ Total: {len(df):,} rows from {len(datasets)} files")
    
    return df


def analyze_dataset(df):
    """Perform EDA"""
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    
    print(f"\nShape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Label distribution
    if 'is_malicious' in df.columns:
        print("\nLabel distribution:")
        for label, count in df['is_malicious'].value_counts().items():
            label_name = 'Malicious' if label == 1 else 'Benign'
            print(f"  {label_name}: {count:,} ({count/len(df)*100:.2f}%)")
    
    return df


def preprocess_data(df, max_duplicates=3):
    """Clean and preprocess dataset"""
    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    
    initial_rows = len(df)
    
    # Remove empty queries
    df = df[df['raw_query'].notna()]
    df = df[df['raw_query'].str.strip() != '']
    print(f"\nAfter removing empty: {len(df):,}")
    
    # Create hash for deduplication
    df['query_hash'] = df['raw_query'].apply(
        lambda x: hashlib.md5(str(x).encode()).hexdigest()
    )
    
    # Handle duplicates
    keep_indices = []
    for query_hash in df['query_hash'].unique():
        query_indices = df[df['query_hash'] == query_hash].index
        if len(query_indices) <= max_duplicates:
            keep_indices.extend(query_indices)
        else:
            sampled = np.random.choice(query_indices, size=max_duplicates, replace=False)
            keep_indices.extend(sampled)
    
    df = df.loc[keep_indices]
    print(f"After deduplication: {len(df):,}")
    
    # Anonymize IPs
    if 'source_ip' in df.columns:
        df['source_ip'] = df['source_ip'].apply(
            lambda x: '.'.join(str(x).split('.')[:2]) + '.XXX.XXX' if pd.notna(x) else 'unknown'
        )
    
    print(f"\n✓ Final size: {len(df):,} rows ({initial_rows - len(df):,} removed)")
    
    return df


# ========================================================================
# FEATURE ENGINEERING
# ========================================================================

def extract_numeric_features(queries):
    """Extract hand-engineered features"""
    sql_keywords = ['UNION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP',
                    'CREATE', 'ALTER', 'EXEC', 'SLEEP', 'WAITFOR', 'BENCHMARK',
                    'INFORMATION_SCHEMA', 'XP_CMDSHELL', 'LOAD_FILE']
    
    features_list = []
    for query in queries:
        q_upper = str(query).upper()
        features_list.append({
            'len_raw': len(q_upper),
            'count_single_quote': q_upper.count("'"),
            'count_double_quote': q_upper.count('"'),
            'count_dashes': q_upper.count('--'),
            'count_semicolon': q_upper.count(';'),
            'count_comment': q_upper.count('/*') + q_upper.count('*/') + q_upper.count('#'),
            'num_sql_keywords': sum(1 for kw in sql_keywords if kw in q_upper),
            'has_union': int('UNION' in q_upper),
            'has_or_equals': int('OR' in q_upper and '=' in q_upper),
            'has_sleep': int('SLEEP' in q_upper or 'WAITFOR' in q_upper),
            'has_comments': int('--' in q_upper or '/*' in q_upper),
            'url_encoded': int('%' in q_upper),
        })
    
    return pd.DataFrame(features_list)


def build_features(X_train, X_val, X_test, tfidf_config):
    """Build complete feature matrices"""
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # TF-IDF
    print("\n1. TF-IDF vectorization...")
    tfidf = TfidfVectorizer(**tfidf_config)
    tfidf_train = tfidf.fit_transform(X_train)
    tfidf_val = tfidf.transform(X_val)
    tfidf_test = tfidf.transform(X_test)
    print(f"   ✓ TF-IDF shape: {tfidf_train.shape}")
    
    # Numeric features
    print("\n2. Numeric features...")
    numeric_train = extract_numeric_features(X_train)
    numeric_val = extract_numeric_features(X_val)
    numeric_test = extract_numeric_features(X_test)
    print(f"   ✓ Numeric shape: {numeric_train.shape}")
    
    # Scale numeric
    scaler = StandardScaler()
    numeric_scaled_train = scaler.fit_transform(numeric_train)
    numeric_scaled_val = scaler.transform(numeric_val)
    numeric_scaled_test = scaler.transform(numeric_test)
    
    # Combine
    from scipy.sparse import hstack, csr_matrix
    X_train_full = hstack([tfidf_train, csr_matrix(numeric_scaled_train)])
    X_val_full = hstack([tfidf_val, csr_matrix(numeric_scaled_val)])
    X_test_full = hstack([tfidf_test, csr_matrix(numeric_scaled_test)])
    
    print(f"\n✓ Final shapes:")
    print(f"  Train: {X_train_full.shape}")
    print(f"  Val:   {X_val_full.shape}")
    print(f"  Test:  {X_test_full.shape}")
    
    return X_train_full, X_val_full, X_test_full, tfidf, scaler


# ========================================================================
# XGBOOST TRAINING
# ========================================================================

def train_xgboost(X_train, y_train, X_val, y_val, quick=False):
    """Train XGBoost with grid search"""
    print("\n" + "=" * 80)
    print("XGBOOST TRAINING")
    print("=" * 80)
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
    
    if quick:
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [6],
            'subsample': [0.8],
        }
    else:
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.05, 0.1],
            'max_depth': [6, 10],
            'subsample': [0.7, 1.0],
        }
    
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    num_combinations = (len(param_grid['n_estimators']) * 
                       len(param_grid['learning_rate']) * 
                       len(param_grid['max_depth']) * 
                       len(param_grid['subsample']))
    print(f"\nGrid search with {num_combinations} combinations...")
    
    grid_search = GridSearchCV(
        xgb, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
    )
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start
    
    best_model = grid_search.best_estimator_
    print(f"\nBest params: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'model_name': 'XGBoost',
        'params': grid_search.best_params_,
        'train_time': train_time,
        'precision': float(precision_score(y_val, y_pred)),
        'recall': float(recall_score(y_val, y_pred)),
        'f1': float(f1_score(y_val, y_pred)),
        'roc_auc': float(roc_auc_score(y_val, y_proba)),
    }
    
    print(f"\nValidation Results:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Time:      {train_time:.2f}s")
    
    return best_model, metrics


# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main(quick=False):
    """Main training pipeline"""
    
    print("\n")
    print("=" * 80)
    print("SQL INJECTION DETECTION - XGBOOST TRAINING")
    print("=" * 80)
    print(f"\nMode: {'QUICK' if quick else 'FULL'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: XGBoost (XGBClassifier)")
    
    # Load data
    df_raw = load_and_merge_datasets()
    df_raw = analyze_dataset(df_raw)
    
    # Preprocess
    df_clean = preprocess_data(df_raw, max_duplicates=CONFIG['max_duplicates'])
    
    # Save merged data
    df_clean.to_csv(ML_DATA_DIR / 'merged.csv', index=False)
    print(f"\n✓ Saved merged dataset to {ML_DATA_DIR / 'merged.csv'}")
    
    # Split data
    X = df_clean['raw_query']
    y = df_clean['is_malicious']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], stratify=y, random_state=RANDOM_SEED
    )
    
    val_size_adjusted = CONFIG['val_size'] / (1 - CONFIG['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    
    # Feature engineering
    X_train_full, X_val_full, X_test_full, tfidf, scaler = build_features(
        X_train, X_val, X_test, CONFIG['tfidf']
    )
    
    # Save feature extractors
    joblib.dump(tfidf, ML_MODELS_DIR / 'tfidf_vectorizer.joblib')
    joblib.dump(scaler, ML_MODELS_DIR / 'numeric_scaler.joblib')
    print(f"\n✓ Saved feature extractors to {ML_MODELS_DIR}")
    
    # Train XGBoost
    model_xgb, metrics_xgb = train_xgboost(X_train_full, y_train, X_val_full, y_val, quick=quick)
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"best_xgboost_{timestamp}_f1_{metrics_xgb['f1']:.3f}.joblib"
    joblib.dump(model_xgb, ML_MODELS_DIR / model_filename)
    print(f"\n✓ Saved model to {ML_MODELS_DIR / model_filename}")
    
    # Save metadata
    metadata = {
        'model_name': 'XGBoost',
        'timestamp': timestamp,
        'metrics': metrics_xgb,
        'config': CONFIG,
        'dataset_info': {
            'total_samples': len(df_clean),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
        }
    }
    
    with open(ML_MODELS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {ML_MODELS_DIR / 'model_metadata.json'}")
    
    # Save model comparison (XGBoost only)
    comparison_df = pd.DataFrame([metrics_xgb])
    comparison_df.to_csv(ML_REPORTS_DIR / 'model_comparison.csv', index=False)
    print(f"✓ Saved comparison to {ML_REPORTS_DIR / 'model_comparison.csv'}")
    
    # Final test evaluation
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION")
    print("=" * 80)
    
    y_test_pred = model_xgb.predict(X_test_full)
    y_test_proba = model_xgb.predict_proba(X_test_full)[:, 1]
    
    test_metrics = {
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
    }
    
    print(f"\nTest Set Results:")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    
    # Append to deliverables
    summary_path = DELIVERABLES_DIR / 'summary.txt'
    with open(summary_path, 'a') as f:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"XGBOOST TRAINING COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
        f.write(f"\nDatasets used:\n")
        f.write(f"  - data/dataset.csv\n")
        f.write(f"  - FINAL_DATASET_FOR_AI_TEAM_v3.csv\n")
        f.write(f"  - Total samples: {len(df_clean):,}\n")
        f.write(f"\nModel: XGBoost\n")
        f.write(f"  F1 Score: {metrics_xgb['f1']:.4f}\n")
        f.write(f"  Precision: {metrics_xgb['precision']:.4f}\n")
        f.write(f"  Recall: {metrics_xgb['recall']:.4f}\n")
        f.write(f"\nArtifacts saved to:\n")
        f.write(f"  - ml/models/{model_filename}\n")
        f.write(f"  - ml/models/tfidf_vectorizer.joblib\n")
        f.write(f"  - ml/models/numeric_scaler.joblib\n")
        f.write(f"  - ml/reports/model_comparison.csv\n")
    
    print(f"\n✓ Updated {summary_path}")
    print("\n" + "=" * 80)
    print("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nModel: {model_filename}")
    print(f"Validation F1: {metrics_xgb['f1']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBoost model for SQL injection detection')
    parser.add_argument('--quick', action='store_true', help='Run quick training with reduced grids')
    
    args = parser.parse_args()
    
    main(quick=args.quick)
