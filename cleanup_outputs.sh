#!/bin/bash

# Cleanup script to remove all experimental outputs
# Usage: ./cleanup_outputs.sh

echo "🧹 Cleaning up all experimental outputs..."

# Remove logs directory
if [ -d "logs" ]; then
    echo "  Removing logs/"
    rm -rf logs/*
fi

# Remove results directory
if [ -d "results" ]; then
    echo "  Removing results/"
    rm -rf results/*
fi

# Remove publication outputs
if [ -d "publication_outputs" ]; then
    echo "  Removing publication_outputs/"
    rm -rf publication_outputs/*
fi

# Remove any generated models
if [ -d "models" ]; then
    echo "  Removing models/"
    rm -rf models/*
fi

# Remove temporary files
echo "  Removing temporary files..."
find . -name "*.png" -not -path "./docs/*" -delete
find . -name "*.pdf" -not -path "./docs/*" -delete
find . -name "*.csv" -not -path "./docs/*" -delete
find . -name "*.json" -not -path "./docs/*" -not -name "requirements.txt" -not -name "package.json" -not -name "manifest.json" -delete
find . -name "*_history.png" -delete
find . -name "stability_history.png" -delete
find . -name "training_data_distribution.png" -delete

# Remove Python cache files
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Remove Jupyter checkpoints
find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# Remove any nohup output files
rm -f nohup.out

echo "✅ Cleanup completed!"
echo ""
echo "Preserved directories:"
echo "  - src/ (source code)"
echo "  - docs/ (documentation)" 
echo "  - scripts/ (utility scripts)"
echo "  - examples/ (example code)"
echo "  - tests/ (test files)"
echo ""
echo "You can now run fresh experiments."
