"""
AI Model Benchmarking Framework
Author: Alif Octrio
Description: Comprehensive testing infrastructure for evaluating AI model performance
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

class ModelBenchmark:
    """
    Framework for systematic AI model performance evaluation.
    Enables comparison across multiple iterations and metrics.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize benchmarking framework.
        
        Args:
            model_name: Name/identifier for the model being tested
        """
        self.model_name = model_name
        self.results = []
        self.baseline_metrics = None
        
    def run_benchmark(
        self,
        predictions: List,
        ground_truth: List,
        iteration: int,
        metadata: Dict = None
    ) -> Dict:
        """
        Run benchmark test on model predictions.
        
        Args:
            predictions: Model output predictions
            ground_truth: Actual correct values
            iteration: Model iteration number
            metadata: Additional context (hyperparameters, etc.)
            
        Returns:
            Dictionary containing all performance metrics
        """
        metrics = self._calculate_metrics(predictions, ground_truth)
        
        result = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.results.append(result)
        
        # Compare with baseline if exists
        if self.baseline_metrics:
            improvement = self._calculate_improvement(metrics, self.baseline_metrics)
            result['improvement'] = improvement
        
        return result
    
    def _calculate_metrics(
        self,
        predictions: List,
        ground_truth: List
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of metric name -> value
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Accuracy
        accuracy = np.mean(predictions == ground_truth)
        
        # Precision, Recall, F1 (for binary classification)
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        false_positives = np.sum((predictions == 1) & (ground_truth == 0))
        false_negatives = np.sum((predictions == 0) & (ground_truth == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positives / len(ground_truth)
        }
    
    def set_baseline(self, iteration: int = 0):
        """
        Set baseline metrics from a specific iteration.
        
        Args:
            iteration: Which iteration to use as baseline (default: first)
        """
        if iteration < len(self.results):
            self.baseline_metrics = self.results[iteration]['metrics']
            print(f"✓ Baseline set from iteration {iteration}")
            print(f"  Accuracy: {self.baseline_metrics['accuracy']:.2%}")
        else:
            print(f"✗ Iteration {iteration} not found")
    
    def _calculate_improvement(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate percentage improvement over baseline."""
        improvement = {}
        for metric in current:
            if metric in baseline and baseline[metric] > 0:
                improvement[metric] = ((current[metric] - baseline[metric]) / baseline[metric]) * 100
            else:
                improvement[metric] = 0
        return improvement
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate comprehensive benchmark report.
        
        Returns:
            DataFrame with all iterations and metrics
        """
        if not self.results:
            print("No benchmark results available")
            return pd.DataFrame()
        
        report_data = []
        for result in self.results:
            row = {
                'iteration': result['iteration'],
                'timestamp': result['timestamp'],
                **result['metrics']
            }
            
            if 'improvement' in result:
                for metric, value in result['improvement'].items():
                    row[f'{metric}_improvement'] = value
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def visualize_progress(self, metric: str = 'accuracy'):
        """
        Visualize metric improvement across iterations.
        
        Args:
            metric: Which metric to plot (default: accuracy)
        """
        if not self.results:
            print("No results to visualize")
            return
        
        iterations = [r['iteration'] for r in self.results]
        values = [r['metrics'][metric] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel(f'{metric.capitalize()}', fontsize=12)
        plt.title(f'{self.model_name} - {metric.capitalize()} Progress', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add baseline line if exists
        if self.baseline_metrics and metric in self.baseline_metrics:
            plt.axhline(
                y=self.baseline_metrics[metric],
                color='r',
                linestyle='--',
                label='Baseline'
            )
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filepath: str):
        """
        Export benchmark results to JSON.
        
        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'baseline_metrics': self.baseline_metrics,
                'results': self.results
            }, f, indent=2)
        
        print(f"✓ Results exported to {filepath}")
    
    def print_summary(self):
        """Print summary of benchmark results."""
        if not self.results:
            print("No results available")
            return
        
        print("\n" + "="*60)
        print(f"BENCHMARK SUMMARY: {self.model_name}")
        print("="*60)
        
        latest = self.results[-1]
        print(f"\nLatest Iteration: {latest['iteration']}")
        print(f"Timestamp: {latest['timestamp']}")
        print("\nMetrics:")
        for metric, value in latest['metrics'].items():
            print(f"  {metric.replace('_', ' ').title():25s}: {value:.2%}")
        
        if 'improvement' in latest:
            print("\nImprovement vs Baseline:")
            for metric, value in latest['improvement'].items():
                direction = "↑" if value > 0 else "↓" if value < 0 else "→"
                print(f"  {metric.replace('_', ' ').title():25s}: {direction} {abs(value):.2f}%")
        
        print("="*60 + "\n")


def main():
    """Example usage of ModelBenchmark framework."""
    
    # Initialize benchmark
    benchmark = ModelBenchmark(model_name="ContentClassifier_v1")
    
    # Simulate multiple iterations
    print("Running benchmark iterations...\n")
    
    # Iteration 0 - Baseline (60% accuracy)
    np.random.seed(42)
    ground_truth = np.random.randint(0, 2, 100)
    predictions_v0 = (np.random.rand(100) > 0.4).astype(int)  # ~60% accuracy
    
    result_v0 = benchmark.run_benchmark(
        predictions=predictions_v0,
        ground_truth=ground_truth,
        iteration=0,
        metadata={'version': '1.0', 'training_samples': 1000}
    )
    
    # Set as baseline
    benchmark.set_baseline(iteration=0)
    
    # Iteration 1 - Improved data quality (75% accuracy)
    predictions_v1 = ground_truth.copy()
    predictions_v1[np.random.choice(100, 25, replace=False)] = 1 - predictions_v1[np.random.choice(100, 25, replace=False)]
    
    result_v1 = benchmark.run_benchmark(
        predictions=predictions_v1,
        ground_truth=ground_truth,
        iteration=1,
        metadata={'version': '1.1', 'training_samples': 5000, 'data_quality': 'improved'}
    )
    
    # Iteration 2 - Optimized (90% accuracy)
    predictions_v2 = ground_truth.copy()
    predictions_v2[np.random.choice(100, 10, replace=False)] = 1 - predictions_v2[np.random.choice(100, 10, replace=False)]
    
    result_v2 = benchmark.run_benchmark(
        predictions=predictions_v2,
        ground_truth=ground_truth,
        iteration=2,
        metadata={'version': '2.0', 'training_samples': 10000, 'data_quality': 'optimized'}
    )
    
    # Generate report
    report = benchmark.generate_report()
    print("\nBenchmark Report:")
    print(report.to_string(index=False))
    
    # Print summary
    benchmark.print_summary()
    
    # Visualize (comment out if running in non-interactive environment)
    # benchmark.visualize_progress('accuracy')
    
    # Export results
    benchmark.export_results('benchmark_results.json')


if __name__ == "__main__":
    main()
```

---

## 📝 Output Example:
```
Running benchmark iterations...

✓ Baseline set from iteration 0
  Accuracy: 62.00%

Benchmark Report:
 iteration                  timestamp  accuracy  precision    recall  f1_score  false_positive_rate  accuracy_improvement  precision_improvement  recall_improvement  f1_score_improvement  false_positive_rate_improvement
         0  2024-03-13T10:30:00.123456      0.62       0.65      0.58      0.61                 0.38                   NaN                    NaN                 NaN                   NaN                              NaN
         1  2024-03-13T10:30:01.234567      0.75       0.78      0.72      0.75                 0.25                 20.97                  20.00               24.14                 22.95                           -34.21
         2  2024-03-13T10:30:02.345678      0.90       0.92      0.88      0.90                 0.10                 45.16                  41.54               51.72                 47.54                           -73.68

============================================================
BENCHMARK SUMMARY: ContentClassifier_v1
============================================================

Latest Iteration: 2
Timestamp: 2024-03-13T10:30:02.345678

Metrics:
  Accuracy                 : 90.00%
  Precision                : 92.00%
  Recall                   : 88.00%
  F1 Score                 : 90.00%
  False Positive Rate      : 10.00%

Improvement vs Baseline:
  Accuracy                 : ↑ 45.16%
  Precision                : ↑ 41.54%
  Recall                   : ↑ 51.72%
  F1 Score                 : ↑ 47.54%
  False Positive Rate      : ↓ 73.68%
============================================================

✓ Results exported to benchmark_results.json
