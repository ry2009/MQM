#!/bin/bash
# MQM High-Frequency Portfolio Analysis
# ----------------------------------------------
# This script runs the MQM model analysis on financial data
# and generates trading performance reports.

# Default parameters
SYMBOLS="AAPL IBM"
TIMEFRAMES="1min 5min 30min 60min"
EPOCHS=10
SEQ_LENGTH=30
OUTPUT_DIR="./mqm_results"

# Display header
echo "====================================================="
echo "  MQM High-Frequency Portfolio Analysis   "
echo "====================================================="
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --symbols)
      SYMBOLS="${2}"
      shift 2
      ;;
    --timeframes)
      TIMEFRAMES="${2}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2}"
      shift 2
      ;;
    --seq_length)
      SEQ_LENGTH="${2}"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="${2}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure shell scripts are executable
if [ ! -f "./run_blackbox_model.sh" ]; then
    echo "Error: run_blackbox_model.sh not found in the current directory."
    exit 1
fi

# Make sure it's executable
chmod +x ./run_blackbox_model.sh

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting analysis with the following parameters:"
echo "Symbols: $SYMBOLS"
echo "Timeframes: $TIMEFRAMES"
echo "Epochs: $EPOCHS"
echo "Sequence Length: $SEQ_LENGTH"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Run the blackbox model script
echo "Running MQM analysis..."
./run_blackbox_model.sh --symbols "$SYMBOLS" --timeframes "$TIMEFRAMES" --epochs "$EPOCHS" --seq_length "$SEQ_LENGTH" --output_dir "$OUTPUT_DIR"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo ""
    echo "Performance summary available at:"
    echo "- $OUTPUT_DIR/all_results.csv"
    
    # Display summary if available
    if [ -f "$OUTPUT_DIR/all_results.csv" ]; then
        echo ""
        echo "Summary of results:"
        echo "--------------------------"
        
        echo "Symbol  Timeframe  Sharpe   Annual Return   Max DD   Win Rate   Profit Factor"
        echo "------  ---------  ------   -------------   ------   --------   -------------"
        
        # Skip header line and display formatted table
        tail -n +2 "$OUTPUT_DIR/all_results.csv" | while IFS=, read -r symbol timeframe sharpe return drawdown winrate pf; do
            printf "%-7s %-10s %-8s %-15s %-8s %-10s %-13s\n" "$symbol" "$timeframe" "$sharpe" "$return%" "$drawdown%" "$winrate%" "$pf"
        done
    fi
    
    echo ""
    echo "To view the detailed results, check the output directory: $OUTPUT_DIR"
    
    # If format_results.py is available, run it
    if [ -f "./format_results.py" ]; then
        echo ""
        echo "Run './format_results.py --results_dir $OUTPUT_DIR --generate_report' for a comprehensive HTML report"
    fi
else
    echo "Analysis failed. Please check the error messages above."
fi 
