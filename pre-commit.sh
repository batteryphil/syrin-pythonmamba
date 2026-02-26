#!/bin/bash
# Pre-commit hook script
# Runs linting, formatting, type checking, and tests

set -e

echo "Running pre-commit checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if commands exist
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Run ruff check (linting)
run_ruff_check() {
    echo -e "${YELLOW}Running ruff (linting)...${NC}"
    ruff check src/ tests/ examples/ --output-format=concise
    echo -e "${GREEN}✓ Ruff check passed${NC}"
}

# Run ruff format (formatting)
run_ruff_format() {
    echo -e "${YELLOW}Running ruff (formatting check)...${NC}"
    ruff format --check src/ tests/ examples/
    echo -e "${GREEN}✓ Ruff format check passed${NC}"
}

# Run mypy (type checking) — src/ only; tests/examples have relaxed typing in pyproject.toml
run_mypy() {
    echo -e "${YELLOW}Running mypy (type checking)...${NC}"
    rm -rf .mypy_cache
    python -m mypy --strict src/
    echo -e "${GREEN}✓ Mypy check passed${NC}"
}

# Run pytest (all tests must pass)
run_pytest() {
    echo -e "${YELLOW}Running pytest (tests)...${NC}"
    python -m pytest tests/ -q --tb=short
    echo -e "${GREEN}✓ All tests passed${NC}"
}

# Main execution
echo ""
echo "========================================="
echo "Pre-commit Quality Checks"
echo "========================================="
echo ""

# Run checks
run_ruff_check
run_ruff_format
run_mypy
run_pytest

echo ""
echo "========================================="
echo -e "${GREEN}All pre-commit checks passed!${NC}"
echo "========================================="
