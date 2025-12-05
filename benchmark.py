#!/usr/bin/env python3
"""
Benchmark script for evaluating trained emotion classification models.

This script loads a pre-trained model and evaluates it on test data without
requiring retraining. It supports various metrics and output formats.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss,
    jaccard_score,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
)
import torch

from emotion_config import DEFAULT_EVAL_PATH, DEFAULT_ARTIFACT_DIR, LABELS, TEXT_COL
from emotion_inference import EmotionPredictor


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    model_path: str = DEFAULT_ARTIFACT_DIR
    test_data_path: str = DEFAULT_EVAL_PATH
    output_dir: str = "benchmark_results"
    output_format: str = "json"  # json, csv, or both
    batch_size: int = 32
    device: Optional[str] = None
    save_predictions: bool = True
    save_probabilities: bool = True
    verbose: bool = True


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark a trained emotion classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        default=DEFAULT_ARTIFACT_DIR,
        help="Path to the trained model artifacts directory"
    )
    parser.add_argument(
        "--test-data",
        default=DEFAULT_EVAL_PATH,
        help="Path to the test CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Don't save individual predictions"
    )
    parser.add_argument(
        "--no-save-probabilities",
        action="store_true",
        help="Don't save probability scores"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle device selection
    device = None if args.device == "auto" else args.device
    
    return BenchmarkConfig(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        output_format=args.output_format,
        batch_size=args.batch_size,
        device=device,
        save_predictions=not args.no_save_predictions,
        save_probabilities=not args.no_save_probabilities,
        verbose=not args.quiet,
    )


def load_test_data(test_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and prepare test data."""
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")
    
    df = pd.read_csv(test_path)
    
    # Ensure text column exists and is string
    if TEXT_COL not in df.columns:
        raise ValueError(f"Text column '{TEXT_COL}' not found in test data")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    
    # Extract labels
    missing_labels = [label for label in LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns in test data: {missing_labels}")
    
    texts = df[TEXT_COL].values
    labels = df[LABELS].values.astype(int)
    
    return df, texts, labels


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    # Basic metrics (skip accuracy_score as it's not appropriate for multi-label)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='samples')
    
    # F1 scores
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Precision and Recall
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_samples'] = precision_score(y_true, y_pred, average='samples', zero_division=0)
    
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_samples'] = recall_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Per-label metrics
    per_label_metrics = {}
    for i, label in enumerate(LABELS):
        label_metrics = {
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'support': int(y_true[:, i].sum()),
        }
        
        # AUC-PR and AUC-ROC if we have probabilities
        if y_prob is not None:
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                label_metrics['auc_pr'] = auc(recall_curve, precision_curve)
                label_metrics['auc_roc'] = roc_auc_score(y_true[:, i], y_prob[:, i])
            except ValueError:
                # Handle cases where all labels are the same
                label_metrics['auc_pr'] = 0.0
                label_metrics['auc_roc'] = 0.0
        
        per_label_metrics[label] = label_metrics
    
    metrics['per_label'] = per_label_metrics
    
    # Subset accuracy (exact match)
    metrics['subset_accuracy'] = (y_true == y_pred).all(axis=1).mean()
    
    # Label distribution statistics
    metrics['label_stats'] = {
        'avg_labels_per_sample': float(y_true.mean(axis=1).mean()),
        'label_frequencies': {label: float(y_true[:, i].mean()) for i, label in enumerate(LABELS)},
        'total_samples': len(y_true),
    }
    
    return metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generate a detailed classification report."""
    return classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)


def save_results(metrics: Dict, config: BenchmarkConfig, 
                predictions_df: Optional[pd.DataFrame] = None,
                probabilities_df: Optional[pd.DataFrame] = None,
                classification_report_str: Optional[str] = None) -> None:
    """Save benchmark results to files."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save metrics
    if config.output_format in ["json", "both"]:
        metrics_path = os.path.join(config.output_dir, "benchmark_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        if config.verbose:
            print(f"Metrics saved to: {metrics_path}")
    
    if config.output_format in ["csv", "both"]:
        # Flatten metrics for CSV
        flat_metrics = {}
        for key, value in metrics.items():
            if key == 'per_label':
                for label, label_metrics in value.items():
                    for metric_name, metric_value in label_metrics.items():
                        flat_metrics[f"{label}_{metric_name}"] = metric_value
            elif key == 'label_stats':
                for stat_name, stat_value in value.items():
                    if isinstance(stat_value, dict):
                        for sub_key, sub_value in stat_value.items():
                            flat_metrics[f"{stat_name}_{sub_key}"] = sub_value
                    else:
                        flat_metrics[stat_name] = stat_value
            else:
                flat_metrics[key] = value
        
        metrics_df = pd.DataFrame([flat_metrics])
        csv_path = os.path.join(config.output_dir, "benchmark_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        if config.verbose:
            print(f"Metrics saved to: {csv_path}")
    
    # Save predictions
    if predictions_df is not None and config.save_predictions:
        pred_path = os.path.join(config.output_dir, "predictions.csv")
        predictions_df.to_csv(pred_path, index=False)
        if config.verbose:
            print(f"Predictions saved to: {pred_path}")
    
    # Save probabilities
    if probabilities_df is not None and config.save_probabilities:
        prob_path = os.path.join(config.output_dir, "probabilities.csv")
        probabilities_df.to_csv(prob_path, index=False)
        if config.verbose:
            print(f"Probabilities saved to: {prob_path}")
    
    # Save classification report
    if classification_report_str is not None:
        report_path = os.path.join(config.output_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(classification_report_str)
        if config.verbose:
            print(f"Classification report saved to: {report_path}")


def print_summary(metrics: Dict, config: BenchmarkConfig) -> None:
    """Print a summary of benchmark results."""
    if not config.verbose:
        return
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print(f"Model path: {config.model_path}")
    print(f"Test data: {config.test_data_path}")
    print(f"Total samples: {metrics['label_stats']['total_samples']}")
    print(f"Average labels per sample: {metrics['label_stats']['avg_labels_per_sample']:.3f}")
    
    print("\nOverall Metrics:")
    print(f"  Subset Accuracy (Exact Match): {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  Jaccard Score: {metrics['jaccard_score']:.4f}")
    
    print("\nF1 Scores:")
    print(f"  Micro: {metrics['f1_micro']:.4f}")
    print(f"  Macro: {metrics['f1_macro']:.4f}")
    print(f"  Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Samples: {metrics['f1_samples']:.4f}")
    
    print("\nPrecision/Recall (Micro):")
    print(f"  Precision: {metrics['precision_micro']:.4f}")
    print(f"  Recall: {metrics['recall_micro']:.4f}")
    
    print("\nPer-Label Performance:")
    for label, label_metrics in metrics['per_label'].items():
        support = label_metrics['support']
        print(f"  {label:12s}: F1={label_metrics['f1']:.3f}, "
              f"P={label_metrics['precision']:.3f}, "
              f"R={label_metrics['recall']:.3f}, "
              f"Support={support}")
    
    print("="*60)


def run_benchmark(config: BenchmarkConfig) -> Dict:
    """Run the complete benchmark evaluation."""
    if config.verbose:
        print("Starting benchmark evaluation...")
        print(f"Loading model from: {config.model_path}")
        print(f"Loading test data from: {config.test_data_path}")
    
    # Load test data
    test_df, texts, y_true = load_test_data(config.test_data_path)
    
    # Initialize predictor
    predictor = EmotionPredictor(
        artifact_dir=config.model_path,
        device=config.device,
        batch_size=config.batch_size
    )
    
    if config.verbose:
        print(f"Loaded model on device: {predictor.device}")
        print(f"Evaluating {len(texts)} samples...")
    
    # Get predictions and probabilities
    predictions_df = predictor.predict_dataframe(texts)
    y_pred = predictions_df.values
    
    # Get probabilities
    result = predictor.predict_with_probs(texts)
    y_prob = np.array([list(prob.values()) for prob in result['probabilities']])
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob)
    
    # Generate classification report
    classification_report_str = generate_classification_report(y_true, y_pred)
    
    # Create DataFrames for saving
    predictions_output_df = None
    if config.save_predictions:
        predictions_output_df = pd.concat([
            pd.Series(texts, name=TEXT_COL),
            predictions_df
        ], axis=1)
    
    probabilities_output_df = None
    if config.save_probabilities:
        probabilities_output_df = pd.concat([
            pd.Series(texts, name=TEXT_COL),
            pd.DataFrame(y_prob, columns=LABELS)
        ], axis=1)
    
    # Save results
    save_results(
        metrics, config, 
        predictions_output_df, 
        probabilities_output_df,
        classification_report_str
    )
    
    # Print summary
    print_summary(metrics, config)
    
    return metrics


def main():
    """Main entry point."""
    config = parse_args()
    
    try:
        metrics = run_benchmark(config)
        if config.verbose:
            print(f"\nBenchmark completed successfully!")
            print(f"Results saved to: {config.output_dir}")
        return 0
    except Exception as e:
        import traceback
        print(f"Error during benchmark: {e}", file=sys.stderr)
        print("Full traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())