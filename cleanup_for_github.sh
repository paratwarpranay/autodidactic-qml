#!/bin/bash
# cleanup_for_github.sh
# Run this before pushing to GitHub to remove dev artifacts

echo "Cleaning repository for GitHub publication..."

# Remove virtual environments
rm -rf .venv venv env ENV

# Remove Python caches
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null

# Remove pytest/coverage caches
rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml

# Remove any output/temp directories
rm -rf outputs runs tmp temp

echo "Done. Repository is clean for GitHub."
echo ""
echo "Next steps:"
echo "  1. Review with: git status"
echo "  2. Stage: git add ."
echo "  3. Commit: git commit -m 'v1.0 - Negative results release'"
echo "  4. Tag: git tag -a v1.0-negative-results -m 'Functional basin falsification'"
echo "  5. Push: git push origin main --tags"
