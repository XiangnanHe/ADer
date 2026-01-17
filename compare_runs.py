"""
Compare All Runs - ADer Results Comparison Tool

This script compares all training runs in the runs/ folder and:
1. Exports all results to a single CSV file with all metrics side-by-side
2. Displays summary table in console
3. Shows detailed rankings for each metric
4. Identifies best models

The CSV file contains one row per run with columns:
- Metadata: trainer, dataset, image_size, epochs, batch_size, config, timestamp, run_dir
- All metrics: mAUROC_sp_max, mAUROC_px, mAUPRO_px, mAP_sp_max, etc.

Usage:
    # Basic - generates runs_comparison.csv
    python compare_runs.py

    # Custom output file
    python compare_runs.py --output my_results.csv

    # Show only top-3 in console (CSV still has all runs)
    python compare_runs.py --top_k 3

    # Combine options
    python compare_runs.py --output best_models.csv --top_k 5
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import glob


def parse_run_name(run_dir):
    """Parse run directory name"""
    dirname = os.path.basename(run_dir)
    parts = dirname.split('_')

    trainer = parts[0] if parts else "Unknown"

    # Find timestamp (YYYYMMDD-HHMMSS pattern)
    timestamp = None
    for i, part in enumerate(parts):
        if re.match(r'\d{8}-\d{6}', part):
            timestamp = part
            config = '_'.join(parts[1:i]) if i > 1 else "Unknown"
            break

    if not timestamp:
        config = '_'.join(parts[1:]) if len(parts) > 1 else "Unknown"
        timestamp = "Unknown"

    return {
        'trainer': trainer,
        'config': config,
        'timestamp': timestamp,
        'name': dirname
    }


def extract_metrics_from_log(log_file):
    """Extract final test metrics from the Avg row in tabulated results"""
    if not os.path.exists(log_file):
        return {}

    metrics = {}

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find the last Avg row (search from end)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]

        # Look for Avg row in markdown table (with various spacing)
        if re.search(r'\|\s*Avg\s*\|', line):
            # Also need the header to match column names
            # Search backwards for the header (Name column)
            header_idx = -1
            for j in range(i - 1, max(0, i - 20), -1):
                if re.search(r'\|\s*Name\s*\|', lines[j]):
                    header_idx = j
                    break

            if header_idx == -1:
                continue

            # Parse header
            header = lines[header_idx]
            header_parts = [p.strip() for p in header.split('|')]

            # Parse values
            value_parts = [p.strip() for p in line.split('|')]

            # Match headers to values (skip empty first/last and Name column)
            for h_idx, h_name in enumerate(header_parts):
                if not h_name or h_name == 'Name' or '(Max)' in h_name:
                    continue

                # Find corresponding value (same index)
                if h_idx < len(value_parts):
                    value_str = value_parts[h_idx]
                    # Extract first number
                    match = re.search(r'([\d.]+)', value_str)
                    if match:
                        metrics[h_name] = float(match.group(1))

            # Found metrics, done
            break

    return metrics


def extract_config_info(log_file):
    """Extract configuration info from log file"""
    info = {
        'dataset': 'Unknown',
        'image_size': 'Unknown',
        'epochs': 'Unknown',
        'batch_size': 'Unknown',
    }

    if not os.path.exists(log_file):
        return info

    with open(log_file, 'r') as f:
        lines = f.readlines()[:500]  # Only check first 500 lines

    for line in lines:
        # Dataset
        if 'data.root' in line and ':' in line:
            match = re.search(r"data\.root\s*:\s*'?data/([^'\"\\s]+)", line)
            if match:
                info['dataset'] = match.group(1).strip()

        # Image size
        if line.startswith('size ') and ':' in line:
            match = re.search(r'size\s*:\s*(\d+)', line)
            if match:
                info['image_size'] = match.group(1).strip()

        # Epochs
        if line.startswith('epoch_full ') and ':' in line:
            match = re.search(r'epoch_full\s*:\s*(\d+)', line)
            if match:
                info['epochs'] = match.group(1).strip()

        # Batch size
        if line.startswith('batch_train ') and ':' in line:
            match = re.search(r'batch_train\s*:\s*(\d+)', line)
            if match:
                info['batch_size'] = match.group(1).strip()

    return info


def compare_runs(runs_dir='runs', output_file=None, top_k=None):
    """Compare all runs in the runs directory"""
    print(f"\n{'='*80}")
    print(f"ADer Results Comparison")
    print(f"{'='*80}\n")

    runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        print(f"❌ Runs directory not found: {runs_dir}")
        return

    # Find all run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name != 'tmp']

    if not run_dirs:
        print(f"❌ No run directories found in {runs_dir}")
        return

    print(f"Found {len(run_dirs)} run directories\n")

    # Collect results from all runs
    results = []

    for run_dir in sorted(run_dirs, key=lambda x: x.name):
        log_file = run_dir / 'log_train.txt'

        run_info = parse_run_name(str(run_dir))
        metrics = extract_metrics_from_log(str(log_file))
        config_info = extract_config_info(str(log_file))

        result = {
            'run_dir': run_dir.name,
            'trainer': run_info['trainer'],
            'config': run_info['config'],
            'timestamp': run_info['timestamp'],
            **config_info,
            **metrics
        }

        results.append(result)

    # Auto-generate CSV output if not specified
    if output_file is None:
        output_file = 'runs_comparison.csv'

    if not results:
        print("❌ No results found")
        return

    # Export to CSV immediately (always)
    export_to_csv(results, output_file)
    print(f"✅ Results exported to {output_file}\n")

    # Get all metric names
    metric_names = set()
    for result in results:
        metric_names.update([k for k in result.keys() if k.startswith('m')])

    metric_names = sorted(metric_names)

    # Print summary table
    print(f"{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}\n")

    # Print header
    header = f"{'Trainer':<20} {'Dataset':<10} {'Size':<6} {'Epochs':<7}"
    if 'mAUROC_sp_max' in metric_names:
        header += f" {'Img-AUROC':<10}"
    if 'mAUROC_px' in metric_names:
        header += f" {'Px-AUROC':<10}"
    if 'mAUPRO_px' in metric_names:
        header += f" {'Px-AUPRO':<10}"

    print(header)
    print('-' * len(header))

    # Print rows
    for result in results:
        row = f"{result['trainer']:<20} {result['dataset']:<10} {result['image_size']:<6} {result['epochs']:<7}"

        if 'mAUROC_sp_max' in metric_names:
            val = result.get('mAUROC_sp_max', 0.0)
            row += f" {val:>9.3f}"

        if 'mAUROC_px' in metric_names:
            val = result.get('mAUROC_px', 0.0)
            row += f" {val:>9.3f}"

        if 'mAUPRO_px' in metric_names:
            val = result.get('mAUPRO_px', 0.0)
            row += f" {val:>9.3f}"

        print(row)

    # Print detailed metrics for each metric type
    print(f"\n{'='*120}")
    print("DETAILED METRICS (Sorted by Performance)")
    print(f"{'='*120}\n")

    for metric in metric_names:
        values = [(r['trainer'], r['config'], r.get(metric, 0.0)) for r in results if metric in r]

        if not values:
            continue

        # Sort by value (descending)
        values.sort(key=lambda x: x[2], reverse=True)

        if top_k:
            values = values[:top_k]

        print(f"\n{metric}:")
        print('-' * 100)

        for i, (trainer, config, value) in enumerate(values, 1):
            rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"  {rank_symbol} {i:2d}. {value:>7.3f}  {trainer:<20} {config[:55]}")

    # Find best models
    print(f"\n{'='*120}")
    print("BEST MODELS")
    print(f"{'='*120}\n")

    key_metrics = ['mAUROC_sp_max', 'mAUROC_px', 'mAUPRO_px', 'mAP_sp_max', 'mF1_max_sp_max']

    for metric in key_metrics:
        if metric not in metric_names:
            continue

        best = max(results, key=lambda x: x.get(metric, 0.0))
        best_value = best.get(metric, 0.0)

        print(f"🏆 {metric}:")
        print(f"   Value:   {best_value:.3f}")
        print(f"   Trainer: {best['trainer']}")
        print(f"   Dataset: {best['dataset']}")
        print(f"   Config:  {best['config']}")
        print(f"   Run:     {best['run_dir']}")
        print()

    print(f"\n{'='*120}")
    print(f"✅ Comparison complete! Results saved to: {output_file}")
    print(f"{'='*120}\n")


def export_to_csv(results, output_file):
    """Export results to CSV file with organized columns"""
    import csv

    if not results:
        return

    # Define column order: metadata first, then metrics
    metadata_cols = ['trainer', 'dataset', 'image_size', 'epochs', 'batch_size', 'config', 'timestamp', 'run_dir']

    # Get all metric columns
    all_cols = set()
    for result in results:
        all_cols.update(result.keys())

    metric_cols = sorted([col for col in all_cols if col.startswith('m')])

    # Final column order: metadata + metrics
    columns = [col for col in metadata_cols if col in all_cols] + metric_cols

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description='Compare ADer training runs and export to CSV')
    parser.add_argument('--runs_dir', type=str, default='runs',
                       help='Directory containing run folders (default: runs)')
    parser.add_argument('--output', type=str, default=None,
                       help='CSV output file (default: runs_comparison.csv)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Show only top-k results for each metric in console output')

    args = parser.parse_args()

    compare_runs(args.runs_dir, args.output, args.top_k)


if __name__ == '__main__':
    main()
