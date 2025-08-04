#!/bin/bash
# Cleanup script for the project

echo "🧹 Cleaning up project files..."

# Remove temporary files
rm -f .coverage
rm -f model_training.log
rm -rf __pycache__
rm -f .DS_Store
rm -f bandit-report.json
rm -f coverage.xml

# Remove generated directories
rm -rf htmlcov
rm -rf .pytest_cache

# Remove any Python cache files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

echo "✅ Cleanup completed!"
