#!/usr/bin/env python3
"""
Convert runs_comparison.csv to a paper-ready table format

Usage:
    python csv_to_paper_table.py [--input runs_comparison.csv] [--output paper_table.csv] [--format csv|latex|markdown]
"""

import pandas as pd
import argparse


def create_paper_table(input_file='runs_comparison.csv', output_file=None, format='csv'):
    """Convert comparison CSV to paper-ready format"""

    # Read the CSV
    df = pd.read_csv(input_file)

    # Remove incomplete runs (with 0 or NaN values)
    df = df[df['mAUROC_sp_max'] > 0]

    # Select key metrics for paper (matching benchmark tables)
    paper_cols = {
        'trainer': 'Method',
        'dataset': 'Dataset',
        'image_size': 'Size',
        'epochs': 'Epochs',
        # Image-level metrics
        'mAUROC_sp_max': 'Img-AUROC',
        'mAP_sp_max': 'Img-AP',
        'mF1_max_sp_max': 'Img-F1',
        # Pixel-level metrics
        'mAUROC_px': 'Px-AUROC',
        'mAP_px': 'Px-AP',
        'mF1_max_px': 'Px-F1',
        'mAUPRO_px': 'Px-AUPRO'
    }

    # Select and rename columns
    available_cols = [col for col in paper_cols.keys() if col in df.columns]
    paper_table = df[available_cols].copy()
    paper_table = paper_table.rename(columns=paper_cols)

    # Round numeric columns to 1 decimal place
    numeric_cols = paper_table.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ['Size', 'Epochs']:
            paper_table[col] = paper_table[col].round(1)

    # Sort by Image-level AUROC (descending)
    if 'Img-AUROC' in paper_table.columns:
        paper_table = paper_table.sort_values('Img-AUROC', ascending=False)

    # Output based on format
    if output_file is None:
        output_file = f'paper_table.{format}'

    if format == 'csv':
        paper_table.to_csv(output_file, index=False)
        print(f"✅ Paper table saved to: {output_file}")
        print("\n" + "="*100)
        print(paper_table.to_string(index=False))
        print("="*100)

    elif format == 'latex':
        latex_str = paper_table.to_latex(index=False, float_format='%.1f')
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"✅ LaTeX table saved to: {output_file}")
        print("\nPreview:")
        print(latex_str)

    elif format == 'markdown':
        md_str = paper_table.to_markdown(index=False, floatfmt='.1f')
        with open(output_file, 'w') as f:
            f.write(md_str)
        print(f"✅ Markdown table saved to: {output_file}")
        print("\nPreview:")
        print(md_str)

    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"  - Total methods compared: {len(paper_table)}")
    if 'Dataset' in paper_table.columns:
        print(f"  - Datasets: {', '.join(paper_table['Dataset'].unique())}")
    if 'Img-AUROC' in paper_table.columns:
        best_method = paper_table.iloc[0]['Method']
        best_score = paper_table.iloc[0]['Img-AUROC']
        print(f"  - Best method: {best_method} (Img-AUROC: {best_score:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description='Convert runs_comparison.csv to paper-ready table',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, default='runs_comparison.csv',
                       help='Input CSV file (default: runs_comparison.csv)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file (default: paper_table.{format})')
    parser.add_argument('--format', '-f', type=str, default='csv',
                       choices=['csv', 'latex', 'markdown'],
                       help='Output format (default: csv)')

    args = parser.parse_args()

    create_paper_table(args.input, args.output, args.format)


if __name__ == '__main__':
    main()
