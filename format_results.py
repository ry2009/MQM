#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQM Results Formatter
--------------------------------
Formats and displays the results from MQM analysis in a clean
and professional way, suitable for presentations.

Author: Ryan Mathieu
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import argparse
from tabulate import tabulate
import json

# Set up styling for plots
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.figsize'] = [12, 8]
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

def percentage_formatter(x, pos):
    """Format numbers as percentages"""
    return f'{x:.1f}%'

def format_metrics_table(results_dir):
    """Create a nicely formatted table of metrics"""
    # Load the metrics summary
    metrics_file = os.path.join(results_dir, 'all_results.csv')
    
    if not os.path.exists(metrics_file):
        print(f"Error: Results file not found at {metrics_file}")
        return None
    
    df = pd.read_csv(metrics_file)
    
    # Format the table nicely
    # Convert numeric columns to appropriate formats
    formatted_df = df.copy()
    formatted_df['annual_return'] = formatted_df['annual_return'].map(lambda x: f'{x:.2f}%')
    formatted_df['max_drawdown'] = formatted_df['max_drawdown'].map(lambda x: f'{x:.2f}%')
    formatted_df['sharpe_ratio'] = formatted_df['sharpe_ratio'].map(lambda x: f'{x:.2f}')
    formatted_df['win_rate'] = formatted_df['win_rate'].map(lambda x: f'{x:.2f}%')
    formatted_df['profit_factor'] = formatted_df['profit_factor'].map(lambda x: f'{x:.2f}')
    
    # Rename columns
    formatted_df = formatted_df.rename(columns={
        'symbol': 'Symbol',
        'timeframe': 'Timeframe',
        'sharpe_ratio': 'Sharpe',
        'annual_return': 'Ann. Return',
        'max_drawdown': 'Max DD',
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor'
    })
    
    return formatted_df

def create_comparison_chart(results_dir, output_file=None):
    """Create and save a chart comparing performance across symbols/timeframes"""
    # Load the metrics summary
    metrics_file = os.path.join(results_dir, 'all_results.csv')
    
    if not os.path.exists(metrics_file):
        print(f"Error: Results file not found at {metrics_file}")
        return
    
    df = pd.read_csv(metrics_file)
    
    # Create a more visually appealing plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MQM Performance Metrics', fontsize=16, fontweight='bold')
    
    # Create labels
    labels = df['symbol'] + ' (' + df['timeframe'] + ')'
    
    # 1. Sharpe Ratio
    ax = axes[0, 0]
    bars = ax.bar(range(len(labels)), df['sharpe_ratio'], color='steelblue')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Sharpe Ratio by Symbol/Timeframe')
    ax.set_ylabel('Sharpe Ratio')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Annual Return
    ax = axes[0, 1]
    bars = ax.bar(range(len(labels)), df['annual_return'], color='forestgreen')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Annualized Return by Symbol/Timeframe')
    ax.set_ylabel('Annual Return (%)')
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Win Rate
    ax = axes[1, 0]
    bars = ax.bar(range(len(labels)), df['win_rate'], color='darkorange')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Win Rate by Symbol/Timeframe')
    ax.set_ylabel('Win Rate (%)')
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Max Drawdown
    ax = axes[1, 1]
    bars = ax.bar(range(len(labels)), np.abs(df['max_drawdown']), color='firebrick')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Maximum Drawdown by Symbol/Timeframe')
    ax.set_ylabel('Max Drawdown (%)')
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def generate_performance_report(results_dir, output_dir=None):
    """Generate a comprehensive performance report"""
    if output_dir is None:
        output_dir = results_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get formatted metrics table
    metrics_df = format_metrics_table(results_dir)
    
    if metrics_df is None:
        return
    
    # Generate HTML report
    html_file = os.path.join(output_dir, 'performance_report.html')
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MQM Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f5f5f5; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ padding: 20px; background-color: #f8f9fa; margin-bottom: 20px; 
                      border-radius: 5px; border-left: 5px solid #2c3e50; }}
            .summary {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
            .metric-card {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; 
                          background-color: #f8f9fa; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MQM Model Performance Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Performance Metrics Summary</h2>
            <div class="summary">
                <div class="metric-card">
                    <div class="metric-label">Avg. Sharpe Ratio</div>
                    <div class="metric-value">{metrics_df['Sharpe'].str.replace('%', '').astype(float).mean():.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg. Annual Return</div>
                    <div class="metric-value">{metrics_df['Ann. Return'].str.replace('%', '').astype(float).mean():.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg. Win Rate</div>
                    <div class="metric-value">{metrics_df['Win Rate'].str.replace('%', '').astype(float).mean():.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg. Max Drawdown</div>
                    <div class="metric-value">{metrics_df['Max DD'].str.replace('%', '').astype(float).mean():.2f}%</div>
                </div>
            </div>
            
            <h2>Detailed Performance by Symbol/Timeframe</h2>
            {metrics_df.to_html(index=False)}
            
            <h2>Performance Visualization</h2>
            <img src="comparison_chart.png" alt="Performance Comparison Chart">
            
            <footer>
                <p>MQM Model - &copy; {pd.Timestamp.now().year} Ryan Mathieu</p>
                <p>Note: This is a blackbox implementation that demonstrates performance without revealing
                   proprietary signal generation techniques. Past performance is not indicative of future results.</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    # Create comparison chart
    chart_file = os.path.join(output_dir, 'comparison_chart.png')
    create_comparison_chart(results_dir, output_file=chart_file)
    
    print(f"Performance report generated at: {html_file}")
    return html_file

def main():
    """Main function to generate formatted results"""
    parser = argparse.ArgumentParser(description='Format and display MQM results')
    
    parser.add_argument('--results_dir', type=str, default='./mqm_results',
                      help='Directory containing MQMresults')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save formatted results (defaults to results_dir)')
    parser.add_argument('--show_table', action='store_true',
                      help='Display metrics table in console')
    parser.add_argument('--create_chart', action='store_true',
                      help='Create and display performance comparison chart')
    parser.add_argument('--generate_report', action='store_true',
                      help='Generate comprehensive HTML report')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    # Check if the results file exists
    results_file = os.path.join(args.results_dir, 'all_results.csv')
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return 1
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    # If no specific action is requested, generate the report by default
    if not (args.show_table or args.create_chart or args.generate_report):
        args.generate_report = True
    
    # Show metrics table
    if args.show_table:
        metrics_df = format_metrics_table(args.results_dir)
        if metrics_df is not None:
            print("\nMQM Performance Metrics:")
            print(tabulate(metrics_df, headers='keys', tablefmt='fancy_grid', showindex=False))
            print()
    
    # Create chart
    if args.create_chart:
        create_comparison_chart(args.results_dir)
    
    # Generate report
    if args.generate_report:
        generate_performance_report(args.results_dir, output_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
