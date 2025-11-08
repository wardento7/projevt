"""
Bestmodel.py - Production wrapper for trained XGBoost cybersecurity model

This module provides a production-ready interface to load and serve predictions
from the trained XGBoost model for SQL injection and cyberattack detection.

Author: AI Agent
Created: 2025-11-08
Version: 1.0.0
"""

import os
import json
import joblib
import numpy as np
import re
from typing import Dict, Any, Tuple
from pathlib import Path


class BestModel:
    """
    Production wrapper for the trained XGBoost model.
    
    Loads the model artifacts and provides a clean prediction interface
    with threshold-based action recommendations.
    """
    
    def __init__(self, models_dir: str = None, threshold_mode: str = "balanced"):
        """
        Initialize the BestModel wrapper.
        
        Args:
            models_dir: Path to models directory. Defaults to backend/models/
            threshold_mode: One of 'balanced', 'high_security', or 'high_availability'
        """
        if models_dir is None:
            # Default to backend/models directory
            models_dir = Path(__file__).parent
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        self.threshold_mode = threshold_mode
        
        # Load model artifacts
        self._load_artifacts()
        
        # Set thresholds based on mode
        self._set_thresholds()
    
    def _load_artifacts(self):
        """Load all required model artifacts."""
        try:
            # Find the XGBoost model file
            model_files = list(self.models_dir.glob("best_xgboost_*.joblib"))
            if not model_files:
                raise FileNotFoundError(f"No XGBoost model found in {self.models_dir}")
            
            model_path = model_files[0]  # Use the first (should be only one)
            print(f"Loading model from: {model_path}")
            
            # Load model
            self.model = joblib.load(model_path)
            
            # Load TF-IDF vectorizer
            tfidf_path = self.models_dir / "tfidf_vectorizer.joblib"
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            
            # Load numeric scaler
            scaler_path = self.models_dir / "numeric_scaler.joblib"
            self.numeric_scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = self.models_dir / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            print("✓ All model artifacts loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {str(e)}")
    
    def _set_thresholds(self):
        """Set decision thresholds based on selected mode."""
        # Default thresholds (balanced mode)
        threshold_configs = {
            "balanced": {
                "challenge": 0.3,
                "block": 0.7
            },
            "high_security": {
                "challenge": 0.2,
                "block": 0.5
            },
            "high_availability": {
                "challenge": 0.5,
                "block": 0.9
            }
        }
        
        config = threshold_configs.get(self.threshold_mode, threshold_configs["balanced"])
        self.threshold_challenge = config["challenge"]
        self.threshold_block = config["block"]
        
        print(f"✓ Thresholds set for '{self.threshold_mode}' mode: "
              f"challenge={self.threshold_challenge}, block={self.threshold_block}")
    
    def _extract_features(self, raw_query: str) -> Dict[str, Any]:
        """
        Extract numeric and text features from raw query.
        
        Args:
            raw_query: Raw input query/request string
            
        Returns:
            Dictionary containing extracted features
        """
        # Numeric features
        len_raw = len(raw_query)
        num_special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', raw_query))
        num_sql_keywords = len(re.findall(
            r'\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b',
            raw_query,
            re.IGNORECASE
        ))
        num_quotes = raw_query.count("'") + raw_query.count('"')
        num_dashes = raw_query.count('-')
        num_semicolons = raw_query.count(';')
        num_equals = raw_query.count('=')
        num_percent = raw_query.count('%')
        
        # Suspicious patterns
        has_union = int(bool(re.search(r'\bUNION\b', raw_query, re.IGNORECASE)))
        has_or_1_1 = int(bool(re.search(r'(\bOR\b\s+\d+\s*=\s*\d+|\d+\s*=\s*\d+)', raw_query, re.IGNORECASE)))
        has_comment = int(bool(re.search(r'(--|/\*|\*/|#)', raw_query)))
        has_script_tag = int(bool(re.search(r'<script', raw_query, re.IGNORECASE)))
        
        features = {
            'len_raw': len_raw,
            'num_special_chars': num_special_chars,
            'num_sql_keywords': num_sql_keywords,
            'num_quotes': num_quotes,
            'num_dashes': num_dashes,
            'num_semicolons': num_semicolons,
            'num_equals': num_equals,
            'num_percent': num_percent,
            'has_union': has_union,
            'has_or_1_1': has_or_1_1,
            'has_comment': has_comment,
            'has_script_tag': has_script_tag,
            'suspicious_chars': num_special_chars + num_quotes + num_dashes
        }
        
        return features
    
    def _prepare_input(self, raw_query: str) -> np.ndarray:
        """
        Prepare input features for model prediction.
        
        Args:
            raw_query: Raw input query string
            
        Returns:
            Feature array ready for model input
        """
        # Extract numeric features
        numeric_features = self._extract_features(raw_query)
        
        # Create numeric feature vector (in expected order)
        numeric_vector = np.array([[
            numeric_features['len_raw'],
            numeric_features['num_special_chars'],
            numeric_features['num_sql_keywords'],
            numeric_features['num_quotes'],
            numeric_features['num_dashes'],
            numeric_features['num_semicolons'],
            numeric_features['num_equals'],
            numeric_features['num_percent'],
            numeric_features['has_union'],
            numeric_features['has_or_1_1'],
            numeric_features['has_comment'],
            numeric_features['has_script_tag']
        ]])
        
        # Scale numeric features
        numeric_scaled = self.numeric_scaler.transform(numeric_vector)
        
        # Transform text with TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform([raw_query])
        
        # Concatenate features
        combined_features = np.hstack([numeric_scaled, tfidf_features.toarray()])
        
        return combined_features, numeric_features
    
    def predict(self, raw_query: str) -> Dict[str, Any]:
        """
        Predict threat score and recommend action for a raw query.
        
        Args:
            raw_query: Raw input query/request string
            
        Returns:
            Dictionary containing:
                - score: Probability of being malicious (0-1)
                - action: Recommended action ('allow', 'challenge', 'block')
                - reason: Human-readable explanation
                - features: Extracted feature dictionary
                - confidence: Model confidence level
        """
        try:
            # Prepare input features
            features_array, numeric_features = self._prepare_input(raw_query)
            
            # Get prediction probability
            proba = self.model.predict_proba(features_array)[0]
            score = float(proba[1])  # Probability of malicious class
            
            # Determine action based on thresholds
            if score < self.threshold_challenge:
                action = "allow"
                reason = f"Low threat score ({score:.3f}). Request appears benign."
            elif score < self.threshold_block:
                action = "challenge"
                reason = f"Moderate threat score ({score:.3f}). Additional verification recommended (CAPTCHA, 2FA)."
            else:
                action = "block"
                reason = f"High threat score ({score:.3f}). Request shows malicious patterns and should be blocked."
            
            # Add specific threat indicators to reason
            threat_indicators = []
            if numeric_features['num_sql_keywords'] > 2:
                threat_indicators.append(f"{numeric_features['num_sql_keywords']} SQL keywords")
            if numeric_features['has_union']:
                threat_indicators.append("UNION statement detected")
            if numeric_features['has_or_1_1']:
                threat_indicators.append("OR 1=1 pattern detected")
            if numeric_features['has_comment']:
                threat_indicators.append("SQL comment markers")
            if numeric_features['has_script_tag']:
                threat_indicators.append("Script tag detected (XSS)")
            
            if threat_indicators:
                reason += f" Indicators: {', '.join(threat_indicators)}."
            
            # Confidence level
            confidence = "high" if max(proba) > 0.9 else "medium" if max(proba) > 0.7 else "low"
            
            result = {
                "score": round(score, 6),
                "action": action,
                "reason": reason,
                "features": numeric_features,
                "confidence": confidence,
                "threshold_mode": self.threshold_mode,
                "model_version": self.metadata.get("timestamp", "unknown")
            }
            
            return result
            
        except Exception as e:
            return {
                "score": -1.0,
                "action": "error",
                "reason": f"Prediction failed: {str(e)}",
                "features": {},
                "confidence": "none",
                "error": str(e)
            }
    
    def predict_batch(self, queries: list) -> list:
        """
        Predict threat scores for multiple queries.
        
        Args:
            queries: List of raw query strings
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(query) for query in queries]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata and configuration
        """
        return {
            "model_type": "XGBoost",
            "version": self.metadata.get("timestamp", "unknown"),
            "metrics": self.metadata.get("metrics", {}),
            "threshold_mode": self.threshold_mode,
            "thresholds": {
                "challenge": self.threshold_challenge,
                "block": self.threshold_block
            },
            "models_dir": str(self.models_dir)
        }


# Convenience function for quick predictions
def quick_predict(raw_query: str, threshold_mode: str = "balanced") -> Dict[str, Any]:
    """
    Quick prediction function that creates a model instance and predicts.
    
    Args:
        raw_query: Raw input query string
        threshold_mode: Threshold mode to use
        
    Returns:
        Prediction dictionary
    """
    model = BestModel(threshold_mode=threshold_mode)
    return model.predict(raw_query)


if __name__ == "__main__":
    # Test the model with sample inputs
    print("=" * 70)
    print("BESTMODEL.PY - Self-Test")
    print("=" * 70)
    
    # Initialize model
    print("\n1. Initializing BestModel...")
    model = BestModel(threshold_mode="balanced")
    
    # Test cases
    test_queries = [
        ("SELECT * FROM users WHERE id=1 OR 1=1 --", "SQL Injection"),
        ("https://shop.example.com/product?id=1", "Benign URL"),
        ("' UNION SELECT username,password FROM users--", "SQL Injection with UNION"),
        ("<script>alert('XSS')</script>", "XSS Attack"),
        ("search?q=python+tutorial", "Normal Search")
    ]
    
    print("\n2. Running predictions on test cases...\n")
    for query, label in test_queries:
        result = model.predict(query)
        print(f"Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"Label: {label}")
        print(f"Score: {result['score']:.4f} | Action: {result['action'].upper()} | Confidence: {result['confidence']}")
        print(f"Reason: {result['reason']}")
        print("-" * 70)
    
    # Model info
    print("\n3. Model Information:")
    info = model.get_model_info()
    print(json.dumps(info, indent=2))
    
    print("\n" + "=" * 70)
    print("✓ Self-test completed successfully!")
    print("=" * 70)
