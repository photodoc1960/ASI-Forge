"""
Result Parser for HRM

Parses evaluation results from CSV files.
"""

import csv
from typing import Dict, List, Optional


# Expected evaluation criteria
EXPECTED_CRITERIA = ["Accuracy metrics improvement", "Generalization tests", "Speedup in training time", "Convergence rate analysis"]


def parse_evaluation_results(filepath: str) -> Dict[str, float]:
    """Parse evaluation results from CSV."""
    results = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get('metric', '')
                value = row.get('value', '0')
                try:
                    results[metric] = float(value)
                except ValueError:
                    results[metric] = 0.0 if value != 'N/A' else None
    except FileNotFoundError:
        pass
    return results


def parse_metrics(filepath: str) -> Dict[str, float]:
    """Parse additional metrics from CSV."""
    return parse_evaluation_results(filepath)


def calculate_score(results: Dict[str, float], metrics: Dict[str, float]) -> float:
    """Calculate overall score from results and metrics."""
    score = 0.0
    count = 0

    # Score from main results
    for key, value in results.items():
        if value is not None and isinstance(value, (int, float)):
            score += value
            count += 1

    # Score from metrics
    for key, value in metrics.items():
        if value is not None and isinstance(value, (int, float)):
            score += value
            count += 1

    return score / count if count > 0 else 0.0


def format_results(results: Dict[str, float], metrics: Dict[str, float]) -> str:
    """Format results for display."""
    lines = ["Evaluation Results:"]
    for key, value in results.items():
        lines.append(f"  {key}: {value}")
    if metrics:
        lines.append("\nMetrics:")
        for key, value in metrics.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)
