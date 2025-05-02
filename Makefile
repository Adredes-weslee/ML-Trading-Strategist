# TradingStrategist Makefile
# Supports both Windows (PowerShell) and Unix (Bash) environments

# Variables
PYTHON := python
VENV_NAME := tradingstrategist-env
CONFIG_DIR := configs
OUTPUT_DIR := output

# Detect OS and set appropriate commands
ifeq ($(OS),Windows_NT)
    # PowerShell commands
    VENV_ACTIVATE := $(VENV_NAME)/Scripts/Activate.ps1
    ACTIVATE_CMD := powershell -Command ". $(VENV_ACTIVATE);"
    RM_CMD := powershell -Command "Remove-Item -Recurse -Force"
    MKDIR_CMD := powershell -Command "if (-not (Test-Path"
    MKDIR_END := ")) { New-Item -ItemType Directory -Path"
else
    # Linux/macOS commands
    VENV_ACTIVATE := $(VENV_NAME)/bin/activate
    ACTIVATE_CMD := source $(VENV_ACTIVATE) &&
    RM_CMD := rm -rf
    MKDIR_CMD := mkdir -p
endif

# Default target
.PHONY: help
help:
    @echo "TradingStrategist Makefile"
    @echo "-------------------------"
    @echo "setup              - Create virtual environment and install dependencies"
    @echo "test               - Run tests"
    @echo "lint               - Run code linting"
    @echo "clean              - Remove build artifacts and cache files"
    @echo ""
    @echo "--- Experiments ---"
    @echo "manual-strategy    - Run manual strategy evaluation"
    @echo "experiment1        - Run experiment 1 (Manual vs TreeStrategy)"
    @echo "experiment2        - Run experiment 2 (Impact of Transaction Costs)"
    @echo "all-experiments    - Run all experiments"
    @echo ""
    @echo "--- ML Models ---"
    @echo "train-tree         - Train TreeStrategyLearner model"
    @echo "train-q            - Train QStrategyLearner model"
    @echo "evaluate-tree      - Evaluate TreeStrategyLearner model"
    @echo "evaluate-q         - Evaluate QStrategyLearner model"

# Setup for different environments
.PHONY: setup
setup:
ifeq ($(OS),Windows_NT)
    $(PYTHON) -m venv $(VENV_NAME)
    powershell -Command ". $(VENV_ACTIVATE); pip install -r requirements.txt; pip install -e ."
    @echo "Virtual environment created and dependencies installed"
    @echo "Activate with: . $(VENV_ACTIVATE)"
else
    $(PYTHON) -m venv $(VENV_NAME)
    source $(VENV_ACTIVATE) && pip install -r requirements.txt && pip install -e .
    @echo "Virtual environment created and dependencies installed"
    @echo "Activate with: source $(VENV_ACTIVATE)"
endif
    # Create output directories
ifeq ($(OS),Windows_NT)
    powershell -Command "if (-not (Test-Path $(OUTPUT_DIR))) { New-Item -ItemType Directory -Path $(OUTPUT_DIR) }"
    powershell -Command "if (-not (Test-Path $(OUTPUT_DIR)/models)) { New-Item -ItemType Directory -Path $(OUTPUT_DIR)/models }"
    powershell -Command "if (-not (Test-Path $(OUTPUT_DIR)/figures)) { New-Item -ItemType Directory -Path $(OUTPUT_DIR)/figures }"
else
    mkdir -p $(OUTPUT_DIR)/models
    mkdir -p $(OUTPUT_DIR)/figures
endif

# Testing
.PHONY: test
test:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); pytest"
else
    source $(VENV_ACTIVATE) && pytest
endif

# Linting
.PHONY: lint
lint:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); flake8 src tests; black --check src tests"
else
    source $(VENV_ACTIVATE) && flake8 src tests && black --check src tests
endif

# Cleanup
.PHONY: clean
clean:
ifeq ($(OS),Windows_NT)
    powershell -Command "if (Test-Path build) { Remove-Item -Recurse -Force build }"
    powershell -Command "if (Test-Path dist) { Remove-Item -Recurse -Force dist }"
    powershell -Command "if (Test-Path *.egg-info) { Remove-Item -Recurse -Force *.egg-info }"
    powershell -Command "if (Test-Path .pytest_cache) { Remove-Item -Recurse -Force .pytest_cache }"
    powershell -Command "if (Test-Path .coverage) { Remove-Item -Force .coverage }"
    powershell -Command "if (Test-Path htmlcov) { Remove-Item -Recurse -Force htmlcov }"
    powershell -Command "Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force"
    powershell -Command "Get-ChildItem -Path . -Include *.pyc -Recurse -File | Remove-Item -Force"
else
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf .pytest_cache/
    rm -rf .coverage
    rm -rf htmlcov/
    find . -type d -name __pycache__ -delete 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
endif

# Manual Strategy
.PHONY: manual-strategy
manual-strategy:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); $(PYTHON) -m TradingStrategist.experiments.manual_strategy_evaluation --config $(CONFIG_DIR)/manual_strategy_config.yaml"
else
    source $(VENV_ACTIVATE) && $(PYTHON) -m TradingStrategist.experiments.manual_strategy_evaluation --config $(CONFIG_DIR)/manual_strategy_config.yaml
endif

# Experiments
.PHONY: experiment1
experiment1:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); $(PYTHON) -m TradingStrategist.experiments.experiment1 --config $(CONFIG_DIR)/experiment1.yaml"
else
    source $(VENV_ACTIVATE) && $(PYTHON) -m TradingStrategist.experiments.experiment1 --config $(CONFIG_DIR)/experiment1.yaml
endif

.PHONY: experiment2
experiment2:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); $(PYTHON) -m TradingStrategist.experiments.experiment2 --config $(CONFIG_DIR)/experiment2.yaml"
else
    source $(VENV_ACTIVATE) && $(PYTHON) -m TradingStrategist.experiments.experiment2 --config $(CONFIG_DIR)/experiment2.yaml
endif

# Training ML Models
.PHONY: train-tree
train-tree:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); $(PYTHON) -m TradingStrategist.train --config $(CONFIG_DIR)/tree_strategy.yaml"
else
    source $(VENV_ACTIVATE) && $(PYTHON) -m TradingStrategist.train --config $(CONFIG_DIR)/tree_strategy.yaml
endif

.PHONY: train-q
train-q:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); $(PYTHON) -m TradingStrategist.train --config $(CONFIG_DIR)/qstrategy.yaml"
else
    source $(VENV_ACTIVATE) && $(PYTHON) -m TradingStrategist.train --config $(CONFIG_DIR)/qstrategy.yaml
endif

# Evaluating ML Models
.PHONY: evaluate-tree
evaluate-tree:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); $(PYTHON) -m TradingStrategist.evaluate --config $(CONFIG_DIR)/tree_strategy.yaml"
else
    source $(VENV_ACTIVATE) && $(PYTHON) -m TradingStrategist.evaluate --config $(CONFIG_DIR)/tree_strategy.yaml
endif

.PHONY: evaluate-q
evaluate-q:
ifeq ($(OS),Windows_NT)
    powershell -Command ". $(VENV_ACTIVATE); $(PYTHON) -m TradingStrategist.evaluate --config $(CONFIG_DIR)/qstrategy.yaml"
else
    source $(VENV_ACTIVATE) && $(PYTHON) -m TradingStrategist.evaluate --config $(CONFIG_DIR)/qstrategy.yaml
endif

# Run all experiments
.PHONY: all-experiments
all-experiments: manual-strategy experiment1 experiment2

# Run all training
.PHONY: train-all
train-all: train-tree train-q

# Run all evaluations
.PHONY: evaluate-all
evaluate-all: evaluate-tree evaluate-q