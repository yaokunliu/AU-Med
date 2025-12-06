import os
import re
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple 

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


# --- Configuration ---
WORK_DIR_RELATIVE = "workdir/output/auq_results"
LOG_FILENAME = "polygraph_eval.log"


# --- Regular expressions for log parsing ---

ESTIMATOR_PATTERN = re.compile(
    r"\[INFO\] - \('sequence', '(?P<estimator_name>.*?)'\): \[(?P<value>[\d\.]+e?-?[\d\.]+)?\]"
)

LABEL_PATTERN = re.compile(
    r"\[INFO\] - AMBIGUITY LABELS: \[(?P<label>\d+)\]"
)

TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})/(?P<time>\d{2}-\d{2}-\d{2})/"
)


# --- Calibration metric ---

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    y_true: ground-truth labels (0 or 1)
    y_prob: predicted uncertainty scores (higher = more ambiguous)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    y_conf = 1.0 - y_prob                     # confidence = 1 - uncertainty
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        mask = (y_conf >= bin_lower) & (y_conf < bin_upper)
        
        if mask.any():
            ambiguity_rate_bin = y_true[mask].mean()
            conf_bin = y_conf[mask].mean()
            ece += np.abs(ambiguity_rate_bin - conf_bin) * mask.mean()
            
    return ece


# --- Core analysis functions ---

def parse_log_file(log_path: Path) -> Tuple[str, Dict[str, List[float]], List[int]]:
    """
    Parse a single log file and extract estimator scores and ambiguity labels.
    Returns timestamp string, estimator results dict, and list of labels.
    """
    estimator_results = defaultdict(list)
    ambiguity_labels = []
    timestamp = ""

    timestamp_match = TIMESTAMP_PATTERN.search(str(log_path))
    if timestamp_match:
        timestamp = f"{timestamp_match.group('date')}_{timestamp_match.group('time')}"

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            
            label_match = LABEL_PATTERN.search(line)
            if label_match:
                try:
                    ambiguity_labels.append(int(label_match.group('label')))
                except ValueError:
                    print(f"Warning: Could not parse ambiguity label in line: {line.strip()}")
                continue

            estimator_match = ESTIMATOR_PATTERN.search(line)
            if estimator_match:
                name = estimator_match.group('estimator_name')
                try:
                    value = float(estimator_match.group('value'))
                    estimator_results[name].append(value)
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse estimator value for {name} in line: {line.strip()}")
                    continue
    
    return timestamp, estimator_results, ambiguity_labels


def calculate_metrics(estimator_values: List[float], true_labels: List[int]) -> Dict[str, float]:
    """
    Compute AUROC, ECE, and Brier Score.
    """
    scores = np.asarray(estimator_values)
    labels = np.asarray(true_labels)
    
    if len(scores) != len(labels):
        print(f"Error: Score count ({len(scores)}) does not match label count ({len(labels)}).")
        return {"AUROC": np.nan, "ECE": np.nan, "Brier": np.nan}
        
    if len(set(labels)) < 2 or len(scores) == 0:
        return {"AUROC": np.nan, "ECE": np.nan, "Brier": np.nan}

    metrics = {}
    
    # AUROC
    try:
        metrics["AUROC"] = roc_auc_score(labels, scores)
    except ValueError as e:
        print(f"Warning: AUROC calculation failed: {e}")
        metrics["AUROC"] = np.nan
        
    # ECE (using confidence = 1 - uncertainty)
    metrics["ECE"] = compute_ece(labels, scores)
    
    # Brier Score (treating uncertainty score as P(ambiguous))
    try:
        metrics["Brier"] = brier_score_loss(labels, scores)
    except ValueError as e:
        print(f"Warning: Brier Score calculation failed: {e}")
        metrics["Brier"] = np.nan
        
    return metrics


def analyze_all_logs(root_dir: Path):
    """
    Traverse all log files under root_dir and compute metrics.
    """
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    output_dir_abs = root_dir.resolve()
    output_parts = list(output_dir_abs.parts)
    try:
        output_index = output_parts.index("output")
    except ValueError:
        print("Error: 'output' directory not found in the resolved root path.")
        return all_results
    
    for log_path in root_dir.rglob(LOG_FILENAME):
        try:
            path_parts = log_path.parts
            
            if len(path_parts) < output_index + 5:
                print(f"Warning: Skipping {log_path}, path structure too short.")
                continue

            model_path_group = path_parts[output_index + 2]
            model_path_name = path_parts[output_index + 3]
            dataset_name = path_parts[output_index + 4]
            
            model_full = f"{model_path_group}/{model_path_name}"
            
            timestamp, estimator_results, ambiguity_labels = parse_log_file(log_path)
            
            print(f"Processing {model_full} on {dataset_name} ({timestamp})")
            
            for name, scores in estimator_results.items():
                metrics = calculate_metrics(scores, ambiguity_labels)
                
                all_results[dataset_name][model_full][name].append({
                    "Timestamp": timestamp,
                    **metrics
                })
                
        except Exception as e:
            print(f"Error processing file {log_path}: {e}")
            continue
            
    return all_results


def write_results_to_csv(all_results: Dict, output_root: Path):
    """
    Write aggregated results to per-model/per-dataset CSV files.
    """
    print("\n--- Writing results to CSV files ---")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, models_data in all_results.items():
        for model_full, results_by_estimator in models_data.items():
            model_safe_name = model_full.replace('/', '_')
            output_filename = f"{model_safe_name}_{dataset_name}_metrics.csv"
            
            output_dir = output_root / dataset_name / model_safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename
            
            fieldnames = ['Timestamp', 'Estimator', 'AUROC', 'ECE', 'Brier']
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for estimator_name, run_data_list in results_by_estimator.items():
                    for run_data in run_data_list:
                        writer.writerow({
                            'Timestamp': run_data['Timestamp'],
                            'Estimator': estimator_name,
                            'AUROC': f"{run_data['AUROC']:.4f}" if not np.isnan(run_data['AUROC']) else 'N/A',
                            'ECE': f"{run_data['ECE']:.4f}" if not np.isnan(run_data['ECE']) else 'N/A',
                            'Brier': f"{run_data['Brier']:.4f}" if not np.isnan(run_data['Brier']) else 'N/A'
                        })
            
            print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent 
    
    if script_dir.name == 'scripts':
        base_dir = script_dir.parent
    else:
        base_dir = script_dir

    output_root = base_dir / WORK_DIR_RELATIVE
    
    if not output_root.exists():
        print(f"Error: Output directory not found at {output_root}")
        output_root = Path(os.getcwd()) / WORK_DIR_RELATIVE
        if not output_root.exists():
            print(f"Fatal Error: Output directory not found at {base_dir / WORK_DIR_RELATIVE} or {Path(os.getcwd()) / WORK_DIR_RELATIVE}")
            sys.exit(1)
        
    print(f"Starting log analysis in: {output_root}")
    print("-" * 50)
    
    all_metrics_results = analyze_all_logs(output_root)
    
    if not all_metrics_results:
        print("No valid log files processed or no results found.")
        sys.exit(0)
        
    write_results_to_csv(all_metrics_results, output_root)
    
    print("-" * 50)
    print("Analysis complete.")