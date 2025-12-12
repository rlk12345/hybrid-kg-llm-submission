#!/bin/bash
# Quick start script for clean_repo
# This script helps set up and run the project

set -e

echo "=========================================="
echo "Hybrid-KG-LLM-Project Quick Start"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/generate_all_datasets.sh" ]; then
    echo "ERROR: Please run this script from the clean_repo directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
if [ ! -f "venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.deps_installed
    echo "✓ Dependencies installed"
    echo ""
else
    echo "✓ Dependencies already installed"
    echo ""
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✓ PYTHONPATH set"
echo ""

# Verify setup
echo "Verifying setup..."
python3 verify_setup.py
echo ""

echo "=========================================="
echo "Setup complete! You can now run:"
echo "=========================================="
echo ""
echo "1. Generate all datasets:"
echo "   bash scripts/generate_all_datasets.sh"
echo ""
echo "2. Or follow the instructions in README.md"
echo ""
echo "Note: Remember to activate the virtual environment in new terminals:"
echo "   source venv/bin/activate"
echo "   export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\""
echo ""

