#!/bin/bash
# Test script for MatQuant Mamba blackbox GitHub repository
# -----------------------------------------------------
# This script runs a quick test of the entire setup

echo "====================================================="
echo "  MatQuant Mamba Blackbox GitHub Repository Test     "
echo "====================================================="
echo ""

# Create test output directory
TEST_DIR="./test_results"
mkdir -p $TEST_DIR

echo "Running quick test with minimal parameters..."
echo "This will run a shortened version of the analysis to verify functionality."
echo ""

# Make sure scripts are executable
chmod +x ./run_blackbox_model.sh
chmod +x ./run_hf_portfolio.sh
chmod +x ./format_results.py

# Run with minimal parameters for quick testing
./run_hf_portfolio.sh --symbols "AAPL" --timeframes "5min" --epochs 2 --output_dir $TEST_DIR

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Test completed successfully!"
    echo ""
    
    # Run the formatter to test it as well
    echo "Testing results formatter..."
    ./format_results.py --results_dir $TEST_DIR --show_table
    
    echo ""
    echo "All tests completed successfully!"
    echo ""
    echo "The repository is ready for deployment to GitHub."
    echo "The following files are included in the blackbox implementation:"
    echo "- run_blackbox_model.sh: Core blackbox implementation (doesn't reveal techniques)"
    echo "- run_hf_portfolio.sh: Shell script for easy execution"
    echo "- format_results.py: Results formatter and report generator"
    echo "- README.md: Documentation for users"
    echo "- requirements.txt: Dependencies for installation"
    echo "- LICENSE: MIT license file"
    echo ""
    echo "To run the full analysis:"
    echo "./run_hf_portfolio.sh"
    echo ""
    echo "Note: The full analysis will take longer but will produce more comprehensive results."
else
    echo "Test failed. Please check the error messages above."
fi 