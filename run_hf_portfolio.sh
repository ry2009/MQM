#!/bin/bash
# MatQuant Mamba High-Frequency Portfolio Analysis
# ----------------------------------------------
# This script runs the MatQuant Mamba model analysis on financial data
# and generates trading performance reports.

# Default parameters
SYMBOLS="AAPL IBM"
TIMEFRAMES="1min 5min 30min 60min"
EPOCHS=10
SEQ_LENGTH=30
OUTPUT_DIR="./mqm_results"

# Display header
echo "====================================================="
echo "  MatQuant Mamba High-Frequency Portfolio Analysis   "
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

# Ensure Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is required but not found on your system."
    exit 1
fi

# Check if run_MQM.py exists
if [ ! -f "run_MQM.py" ]; then
    echo "Error: run_MQM.py not found in the current directory."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting analysis with the following parameters:"
echo "Symbols: $SYMBOLS"
echo "Timeframes: $TIMEFRAMES"
echo "Epochs: $EPOCHS"
echo "Sequence Length: $SEQ_LENGTH"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Convert space-separated arguments to command-line format
SYMBOLS_ARG=$(echo $SYMBOLS | tr ' ' ' --symbols ')
SYMBOLS_ARG="--symbols $SYMBOLS_ARG"

TIMEFRAMES_ARG=$(echo $TIMEFRAMES | tr ' ' ' --timeframes ')
TIMEFRAMES_ARG="--timeframes $TIMEFRAMES_ARG"

# Run the Python script
echo "Running MatQuant Mamba analysis..."
python run_MQM.py $SYMBOLS_ARG $TIMEFRAMES_ARG --epochs $EPOCHS --seq_length $SEQ_LENGTH --output_dir "$OUTPUT_DIR"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo ""
    echo "Performance summary available at:"
    echo "- $OUTPUT_DIR/performance_summary.txt"
    echo "- $OUTPUT_DIR/performance_summary.csv"
    echo "- $OUTPUT_DIR/performance_summary.png"
    echo ""
    
    # Display the text summary if available
    if [ -f "$OUTPUT_DIR/performance_summary.txt" ]; then
        echo "Summary of results:"
        echo "-------------------"
        cat "$OUTPUT_DIR/performance_summary.txt"
    fi
    
    echo ""
    echo "To view the detailed results, check the output directory: $OUTPUT_DIR"
else
    echo "Analysis failed. Please check the error messages above."
fi 