#!/usr/bin/env python3
"""Visualize LAS F1 scores across time periods for three parsing models."""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_scores_file(filepath):
    """Extract average LAS F1 scores per time period from a scores file."""
    with open(filepath, 'r') as f:
        content = f.read()

    scores = {}

    # Pattern for time period averages
    period_pattern = r'Average scores for time period (\d{4}-\d{4}).*?\nMetric\t.*?\nUPOS\t.*?\nUAS\t.*?\nLAS\t[\d.]+\t[\d.]+\t([\d.]+)'
    for match in re.finditer(period_pattern, content, re.DOTALL):
        period = match.group(1)
        las_f1 = float(match.group(2))
        scores[period] = las_f1

    # Pattern for overall average
    overall_pattern = r'Overall scores across all time periods.*?\nMetric\t.*?\nUPOS\t.*?\nUAS\t.*?\nLAS\t[\d.]+\t[\d.]+\t([\d.]+)'
    match = re.search(overall_pattern, content, re.DOTALL)
    if match:
        scores['Overall'] = float(match.group(1))

    return scores

def main():
    # Parse scores from all three files
    scores_dir = 'eval/scores'

    talbanken = parse_scores_file(f'{scores_dir}/scores_talbanken.txt')
    transformer_silver = parse_scores_file(f'{scores_dir}/scores_transformer_silver.txt')
    transformer_no_silver = parse_scores_file(f'{scores_dir}/scores_transformer_no_silver.txt')

    # Define time periods in order
    time_periods = ['1700-1750', '1750-1800', '1800-1850', '1850-1900', '1900-1950', 'Overall']

    # Extract values for each model
    talbanken_vals = [talbanken.get(p, 0) for p in time_periods]
    silver_vals = [transformer_silver.get(p, 0) for p in time_periods]
    no_silver_vals = [transformer_no_silver.get(p, 0) for p in time_periods]

    # Create grouped bar chart
    x = np.arange(len(time_periods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, talbanken_vals, width, label='Talbanken', color='#2ecc71')
    bars2 = ax.bar(x, silver_vals, width, label='Transformer Silver', color='#3498db')
    bars3 = ax.bar(x + width, no_silver_vals, width, label='Transformer No Silver', color='#e74c3c')

    # Customize the plot
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('LAS F1 Score', fontsize=12)
    ax.set_title('LAS F1 Scores by Time Period and Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(time_periods, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0.4, 0.9)

    # Add value labels on bars, with bold for winners
    def add_labels(bars, all_vals, model_idx):
        """Add labels to bars, bold if this model is the winner for that category."""
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # Check if this model is the winner for this time period
            period_vals = [all_vals[0][i], all_vals[1][i], all_vals[2][i]]
            is_winner = (model_idx == period_vals.index(max(period_vals)))
            fontweight = 'bold' if is_winner else 'normal'
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90,
                       fontweight=fontweight)

    all_vals = [talbanken_vals, silver_vals, no_silver_vals]
    add_labels(bars1, all_vals, 0)
    add_labels(bars2, all_vals, 1)
    add_labels(bars3, all_vals, 2)

    plt.tight_layout()
    plt.savefig('plot/las_f1_scores_comparison.png', dpi=400)
    print("Saved plot to plot/las_f1_scores_comparison.png and .pdf")
    plt.show()

if __name__ == '__main__':
    main()
