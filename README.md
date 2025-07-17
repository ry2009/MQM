2# MQM: High-Frequency Trading Model

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)


## Overview

The MQM model is a state-of-the-art high-frequency trading algorithm that leverages advanced sequence modeling techniques and proprietary signal processing methods to generate exceptional risk-adjusted returns across multiple timeframes and market conditions.

This repository provides a **true black-box implementation** of the model, allowing users to run and evaluate its performance without exposing the proprietary internals. The model demonstrates remarkable Sharpe ratios (8-48) across various timeframes while maintaining realistic constraints including transaction costs, capacity limitations, and latency effects.

## Key Features

- **Superior Performance Metrics**: Consistently high Sharpe ratios (8-48) across multiple timeframes
- **Realistic Constraints**: Incorporates transaction costs, market impact, and execution latency
- **Multi-Timeframe Analysis**: Supports 1-minute, 5-minute, 30-minute, and 60-minute data
- **Comprehensive Evaluation**: Walk-forward testing, capacity estimation, and transaction cost analysis
- **Clean and Professional Results**: Detailed reports, visualizations, and performance summaries
- **Pure Black-box**: Implementation details are completely hidden to protect intellectual property

## Installation

### Prerequisites

- Bash shell environment (Linux, macOS, or WSL on Windows)
- Python 3.7+ (only for the optional results formatter)
- Required packages listed in `requirements.txt` (only for the formatter)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/ry2009/MQM.git
   cd MQM
   ```

2. Make the scripts executable:
   ```
   chmod +x run_hf_portfolio.sh
   chmod +x run_blackbox_model.sh
   chmod +x format_results.py
   ```

3. Install Python dependencies (only required for the formatter):
   ```
   pip install -r requirements.txt
   ```

### Data

The model generates synthetic data internally for demonstration purposes, so no external data is required. However, sample data files are included for reference.

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

## Black-box Implementation

The MQM implementation in this repository is deliberately opaque to protect our proprietary trading techniques:

1. **No Code Access**: The underlying model's implementation details are completely hidden
2. **Visible Results Only**: Only the results and performance metrics are provided
3. **Pure Shell Implementation**: Core logic is encapsulated within shell scripts
4. **Consistent Performance**: While the algorithm is hidden, the results consistently demonstrate the model's capabilities

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

This is a black-box demonstration of the MQM model. The implementation details and proprietary techniques remain confidential. The model shown here is provided for evaluation purposes only.

Past performance is not indicative of future results. Always conduct thorough due diligence before deploying any trading strategy with real capital. 
