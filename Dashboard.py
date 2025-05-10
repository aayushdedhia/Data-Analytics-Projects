#!/usr/bin/env python3
"""
final_dashboard.py

Generates a polished 2×2 executive dashboard for the
Online Shoppers Purchasing Intention dataset, with robust
input path fallback.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def find_input_file(filename):
    # Try multiple locations for the input CSV
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd(),
        os.getcwd(),
        '/mnt/data'
    ]
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find input file '{filename}' in {search_dirs}")

def main():
    filename = 'online_shoppers_processed.csv'
    input_csv = find_input_file(filename)
    # Save output next to input
    output_png = os.path.join(os.path.dirname(input_csv), 'Final_Dashboard.png')

    # Load processed dataset
    df = pd.read_csv(input_csv)

    # Ensure Month column is categorical with full order
    full_months = ['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
    df['Month'] = pd.Categorical(df['Month'], categories=full_months, ordered=True)

    # Prepare data
    pie_labels = ['No Purchase', 'Purchase']
    pie_sizes = df['Revenue'].value_counts().sort_index()
    conv_by_month = df.groupby('Month')['Revenue'].mean().reindex(full_months)
    no_p = df[df['Revenue'] == 0]['BounceRates']
    p = df[df['Revenue'] == 1]['BounceRates']
    x_vals = df['PageValues']
    y_vals = df['Predicted_Probability']

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Online Shoppers Purchase Insights', fontsize=22, fontweight='bold', y=0.95)

    # Pie
    ax1 = axes[0,0]
    ax1.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
            startangle=90, colors=['#FFA726', '#F4511E'])
    ax1.set_title('Session Outcomes')

    # Line
    ax2 = axes[0,1]
    ax2.plot(full_months, conv_by_month, marker='o', linewidth=2, color='#FB8C00')
    ax2.set_title('Conversion Rate Trend')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Conversion Rate')
    ax2.set_xticks(full_months)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # Boxplot
    ax3 = axes[1,0]
    ax3.boxplot([no_p, p], labels=pie_labels, patch_artist=True,
                boxprops=dict(facecolor='#FFF176'),
                medianprops=dict(color='red'))
    ax3.set_title('Bounce Rate by Outcome')
    ax3.set_ylabel('Bounce Rate')
    ax3.grid(axis='y', linestyle='--', linewidth=0.5)

    # Scatter
    ax4 = axes[1,1]
    ax4.scatter(x_vals, y_vals, s=12, alpha=0.6, color='#FFA726')
    ax4.set_title('Purchase Probability vs. PageValues')
    ax4.set_xlabel('PageValues')
    ax4.set_ylabel('Predicted Probability')
    ax4.grid(True, linestyle='--', linewidth=0.5)

    # Save and close
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    print(f"Dashboard image created at: {output_png}")

if __name__ == '__main__':
    main()
