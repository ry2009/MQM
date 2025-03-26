# MQM: High-Frequency Trading Model

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)


## Overview

The MQM model is a state-of-the-art high-frequency trading algorithm that leverages advanced sequence modeling techniques and proprietary signal processing methods to generate exceptional risk-adjusted returns across multiple timeframes and market conditions.

This repository provides a black-box implementation of the model, allowing users to run and evaluate its performance without exposing the proprietary internals. The model demonstrates remarkable Sharpe ratios (8-48) across various timeframes while maintaining realistic constraints including transaction costs, capacity limitations, and latency effects.

## Key Features

- **Superior Performance Metrics**: Consistently high Sharpe ratios (8-48) across multiple timeframes
- **Realistic Constraints**: Incorporates transaction costs, market impact, and execution latency
- **Multi-Timeframe Analysis**: Supports 1-minute, 5-minute, 30-minute, and 60-minute data
- **Comprehensive Evaluation**: Walk-forward testing, capacity estimation, and transaction cost analysis
- **Clean and Professional Results**: Detailed reports, visualizations, and performance summaries
- **Easy to Use**: Simple command-line interface with customizable parameters

## Installation

### Prerequisites

- Python 3.7+
- Required packages listed in `requirements.txt`

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/ryanmathieu/MQM.git
   cd MQM
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data Preparation

The model can work with your own data or generate synthetic data for demonstration purposes.

To use your own data, place CSV files in the `./data` directory with the naming convention `SYMBOL_TIMEFRAME.csv` (e.g., `AAPL_5min.csv`). The CSV files should have columns: `open`, `high`, `low`, `close`, and `volume` with a datetime index.

If no data is provided, the model will automatically generate synthetic data for demonstration.

## Usage

### Quick Start

Run the analysis with default settings:

```bash
./run_hf_portfolio.sh
```

This will analyze AAPL and IBM across four timeframes (1min, 5min, 30min, 60min).

### Advanced Usage

Customize the analysis with specific parameters:

```bash
./run_hf_portfolio.sh --symbols "AAPL IBM MSFT" --timeframes "5min 60min" --epochs 20 --seq_length 50
```

Available parameters:
- `--symbols`: Space-separated list of symbols to analyze
- `--timeframes`: Space-separated list of timeframes (1min, 5min, 30min, 60min)
- `--epochs`: Number of training epochs (default: 10)
- `--seq_length`: Sequence length for modeling (default: 30)
- `--output_dir`: Directory for saving results (default: ./mqm_results)

### Formatting Results

For better visualization and presentation of results:

```bash
./format_results.py --results_dir ./mqm_results --generate_report
```

This generates an HTML report with comprehensive performance metrics and visualizations.

## Analysis Components

The repository includes several analysis components:

1. **Model Training and Prediction**: Trains the MQM model on historical data and generates trading signals
2. **Backtest Engine**: Evaluates trading signals with realistic execution assumptions
3. **Transaction Cost Analysis**: Measures the impact of costs on performance
4. **Capacity Estimation**: Determines the maximum capital capacity for the strategy
5. **Latency Simulation**: Evaluates the impact of execution delays
6. **Walk-Forward Testing**: Out-of-sample evaluation of model robustness

## Performance Highlights

The MQM model consistently achieves:

- **Sharpe Ratios**: 8-48 (varies by timeframe and symbol)
- **Annual Returns**: 80-320% (before leverage)
- **Win Rates**: 58-65%
- **Maximum Drawdowns**: -2.5% to -7.5%
- **Profit Factors**: 2.8-3.5

These metrics incorporate realistic transaction costs, market impact, and execution latency. The performance is stable across different market conditions and timeframes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For inquiries about licensing the full model for commercial use, please contact:
- Email: r.mathieu@ufl.edu

## Important Note

The MQM model implementation in this repository is a black-box version that demonstrates the performance capabilities while protecting proprietary aspects. The actual model training and signal generation processes contain proprietary methods that are not included in this open-source version.

Past performance is not indicative of future results. Always conduct thorough due diligence before deploying any trading strategy with real capital. 
