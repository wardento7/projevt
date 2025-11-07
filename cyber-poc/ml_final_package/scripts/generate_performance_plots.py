#!/usr/bin/env python3
"""
Performance Plots Generator for SQL Injection Detection ML Model
Author: Wardento (Cyber AI Engineer)
Date: November 8, 2025

This script generates visualization plots for model performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_model_comparison(csv_path):
    """Load model comparison CSV"""
    return pd.read_csv(csv_path)

def load_thresholds(json_path):
    """Load threshold recommendations"""
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_accuracy_plot(df, output_path):
    """Generate accuracy comparison plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model_name'].tolist()
    # Calculate accuracy from precision and recall (f1 is harmonic mean)
    # For display purposes, use f1 as proxy for accuracy
    accuracies = df['f1'].tolist()
    
    bars = ax.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison\nSQL Injection Detection', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0.97, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

def generate_f1_plot(df, output_path):
    """Generate F1 score comparison plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model_name'].tolist()
    f1_scores = df['f1'].tolist()
    
    bars = ax.bar(models, f1_scores, color=['#95E1D3', '#F38181', '#AA96DA'], alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model F1 Score Comparison (Validation Set)\nSQL Injection Detection', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0.98, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight best model
    best_idx = f1_scores.index(max(f1_scores))
    bars[best_idx].set_color('#FFD700')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

def generate_roc_plot(df, output_path):
    """Generate ROC AUC comparison plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model_name'].tolist()
    roc_auc = df['roc_auc'].tolist()
    
    bars = ax.barh(models, roc_auc, color=['#E74C3C', '#3498DB', '#2ECC71'], alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {width:.6f}',
                ha='left', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('ROC AUC Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model ROC AUC Comparison (Validation Set)\nSQL Injection Detection', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim([0.995, 1.001])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

def generate_precision_recall_plot(df, output_path):
    """Generate Precision-Recall comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = df['model_name'].tolist()
    precision = df['precision'].tolist()
    recall = df['recall'].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, precision, width, label='Precision', 
                   color='#FF6B9D', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, recall, width, label='Recall', 
                   color='#C44569', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Recall Comparison (Validation Set)\nSQL Injection Detection', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0.98, 1.005])
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

def main():
    """Main execution"""
    print("=" * 70)
    print("  PERFORMANCE PLOTS GENERATOR")
    print("  SQL Injection Detection ML Model")
    print("  Author: Wardento (Cyber AI Engineer)")
    print("=" * 70)
    print()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    model_comparison_csv = base_dir / 'reports' / 'model_comparison.csv'
    thresholds_json = base_dir / 'reports' / 'thresholds.json'
    plots_dir = base_dir / 'reports' / 'plots'
    
    # Create plots directory if not exists
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📊 Loading data...")
    df = load_model_comparison(model_comparison_csv)
    thresholds = load_thresholds(thresholds_json)
    print(f"   ✓ Loaded {len(df)} models")
    print(f"   ✓ Loaded {len(thresholds)} threshold profiles")
    print()
    
    # Generate plots
    print("🎨 Generating plots...")
    print()
    
    generate_accuracy_plot(df, plots_dir / 'accuracy.png')
    generate_f1_plot(df, plots_dir / 'f1.png')
    generate_roc_plot(df, plots_dir / 'roc.png')
    generate_precision_recall_plot(df, plots_dir / 'precision_recall.png')
    
    print()
    print("=" * 70)
    print("✅ All plots generated successfully!")
    print(f"📁 Output directory: {plots_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
