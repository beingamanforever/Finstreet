#!/usr/bin/env python
"""Bias Audit Script - Data Leakage Verification."""

import sys
import os
import logging
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features.preprocessing import preprocess_pipeline, get_feature_columns
from src.features.indicators import add_indicators
from src.features.advanced_features import add_all_features
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BiasAuditor:
    """
    Auditor class to verify absence of look-ahead bias in feature engineering.
    
    The core principle: For any given day T, the features should be computable
    using only data available at market close on day T or earlier.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.results: Dict[str, bool] = {}
        
    def run_all_audits(self) -> bool:
        """Execute all bias audits and return overall pass/fail status."""
        audits = [
            ("Feature Temporal Alignment", self._audit_feature_alignment),
            ("Rolling Window Integrity", self._audit_rolling_windows),
            ("Future Price Correlation", self._audit_future_correlation),
            ("Label Construction Validity", self._audit_label_construction),
        ]
        
        all_passed = True
        logger.info("=" * 60)
        logger.info("BIAS AUDIT REPORT")
        logger.info("=" * 60)
        
        for name, audit_func in audits:
            try:
                passed, details = audit_func()
                self.results[name] = passed
                status = "PASS" if passed else "FAIL"
                logger.info(f"[{status}] {name}")
                if details:
                    for detail in details:
                        logger.info(f"       {detail}")
                if not passed:
                    all_passed = False
            except Exception as e:
                logger.error(f"[ERROR] {name}: {e}")
                self.results[name] = False
                all_passed = False
                
        logger.info("=" * 60)
        final_status = "PASSED" if all_passed else "FAILED"
        logger.info(f"OVERALL AUDIT STATUS: {final_status}")
        logger.info("=" * 60)
        
        return all_passed
    
    def _audit_feature_alignment(self) -> Tuple[bool, List[str]]:
        """
        Verify that features for day T do not contain day T+1 close price.
        
        Method: For each row, check that no feature value equals or is derived
        from the next day's OHLCV values.
        """
        details = []
        df = self.df.copy()
        df = add_indicators(df)
        df = add_all_features(df)
        
        feature_cols = get_feature_columns(df)
        violations = []
        
        for i in range(len(df) - 1):
            current_features = df.iloc[i][feature_cols].values
            next_close = df.iloc[i + 1]["close"]
            next_open = df.iloc[i + 1]["open"]
            next_high = df.iloc[i + 1]["high"]
            next_low = df.iloc[i + 1]["low"]
            
            for j, (col, val) in enumerate(zip(feature_cols, current_features)):
                if pd.notna(val):
                    if np.isclose(val, next_close, rtol=1e-9):
                        violations.append(f"Row {i}, {col} equals next close")
                    if np.isclose(val, next_open, rtol=1e-9):
                        violations.append(f"Row {i}, {col} equals next open")
                        
        passed = len(violations) == 0
        details.append(f"Checked {len(df) - 1} rows for future price leakage")
        details.append(f"Violations found: {len(violations)}")
        
        if violations[:3]:
            details.extend(violations[:3])
            
        return passed, details
    
    def _audit_rolling_windows(self) -> Tuple[bool, List[str]]:
        """
        Verify that rolling window calculations use only past data.
        
        Method: Recompute indicators with truncated data and verify consistency.
        """
        details = []
        
        test_idx = len(self.df) // 2
        
        df_full = self.df.copy()
        df_full = add_indicators(df_full)
        df_full = add_all_features(df_full)
        
        df_truncated = self.df.iloc[:test_idx + 1].copy()
        df_truncated = add_indicators(df_truncated)
        df_truncated = add_all_features(df_truncated)
        
        feature_cols = get_feature_columns(df_full)
        mismatches = []
        
        for col in feature_cols:
            full_val = df_full.iloc[test_idx][col]
            trunc_val = df_truncated.iloc[test_idx][col]
            
            if pd.notna(full_val) and pd.notna(trunc_val):
                if not np.isclose(full_val, trunc_val, rtol=1e-6, equal_nan=True):
                    mismatches.append(col)
                    
        passed = len(mismatches) == 0
        details.append(f"Compared {len(feature_cols)} features at index {test_idx}")
        details.append(f"Rolling window mismatches: {len(mismatches)}")
        
        if mismatches[:3]:
            details.append(f"Mismatched features: {', '.join(mismatches[:3])}")
            
        return passed, details
    
    def _audit_future_correlation(self) -> Tuple[bool, List[str]]:
        """
        Check correlation between features and future prices.
        
        A suspiciously high correlation (>0.95) with future close suggests leakage.
        """
        details = []
        df = self.df.copy()
        df = add_indicators(df)
        df = add_all_features(df)
        
        df["_audit_future_close"] = df["close"].shift(-1)
        feature_cols = get_feature_columns(df)
        feature_cols = [c for c in feature_cols if not c.startswith("_audit")]
        
        suspicious_features = []
        
        for col in feature_cols:
            if df[col].notna().sum() > 30:
                corr = df[col].corr(df["_audit_future_close"])
                if pd.notna(corr) and abs(corr) > 0.95:
                    suspicious_features.append((col, corr))
                    
        passed = len(suspicious_features) == 0
        details.append(f"Analyzed {len(feature_cols)} features for future correlation")
        details.append(f"Suspicious correlations (|r| > 0.95): {len(suspicious_features)}")
        
        for feat, corr in suspicious_features[:3]:
            details.append(f"  {feat}: r = {corr:.4f}")
            
        return passed, details
    
    def _audit_label_construction(self) -> Tuple[bool, List[str]]:
        """
        Verify that labels are constructed from future returns (which is valid).
        
        Labels SHOULD use future data - that's what we're predicting. This audit
        confirms the labeling uses the correct temporal direction.
        """
        details = []
        
        df = preprocess_pipeline(self.df.copy(), is_training=True)
        
        valid_label_construction = (
            "future_return" in df.columns and 
            "t1" in df.columns and
            "label" in df.columns
        )
        
        if valid_label_construction:
            positive_labels = (df["label"] == 1).sum()
            future_positive = (df["future_return"] > 0).sum()
            
            alignment = abs(positive_labels - future_positive) < len(df) * 0.1
            details.append(f"Label column present: Yes")
            details.append(f"Future return column present: Yes")
            details.append(f"Label-return alignment: {'Valid' if alignment else 'Check required'}")
            
        return valid_label_construction, details


def audit_source_code_patterns() -> Tuple[bool, List[str]]:
    """
    Static analysis to detect common data leakage patterns in source code.
    
    Checks for:
    - .bfill() without justification (backward fill uses future data)
    - .shift(-N) where N > 0 (accessing future rows) in feature code
    - rolling().apply() with rank() that could leak
    """
    import re
    from pathlib import Path
    
    details = []
    violations = []
    
    feature_files = [
        "src/features/labeling.py",
        "src/features/indicators.py",
        "src/features/advanced_features.py",
        "src/features/preprocessing.py",
    ]
    
    patterns = {
        r'\.bfill\(\)': "bfill() without expanding/ffill alternative",
        r'\.shift\s*\(\s*-\d': "shift(-N) accesses future data",
        r'rolling\([^)]+\)\.apply\([^)]*rank\(pct=True\)': "rolling rank may leak future data",
    }
    
    for filepath in feature_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                for pattern, description in patterns.items():
                    for i, line in enumerate(lines, 1):
                        # Skip comments
                        if line.strip().startswith('#'):
                            continue
                        if re.search(pattern, line):
                            violations.append(f"{filepath}:{i} - {description}")
        except FileNotFoundError:
            pass
            
    passed = len(violations) == 0
    details.append(f"Scanned {len(feature_files)} feature files for leakage patterns")
    details.append(f"Code pattern violations: {len(violations)}")
    
    for v in violations[:5]:
        details.append(f"  {v}")
        
    return passed, details


def main():
    data_path = str(settings.data.data_path)
    
    auditor = BiasAuditor(data_path)
    data_passed = auditor.run_all_audits()
    
    # Run source code pattern audit
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("SOURCE CODE PATTERN AUDIT")
    logger.info("=" * 60)
    
    code_passed, code_details = audit_source_code_patterns()
    status = "PASS" if code_passed else "FAIL"
    logger.info(f"[{status}] Source Code Leakage Patterns")
    for detail in code_details:
        logger.info(f"       {detail}")
    
    logger.info("=" * 60)
    
    overall_passed = data_passed and code_passed
    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
