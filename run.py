"""
Launcher script for TradingStrategist - Run from project root directory

Usage:
    python run.py train --config configs/tree_strategy.yaml
    python run.py evaluate --config configs/tree_strategy.yaml
    python run.py experiment1 --config configs/experiment1.yaml
    python run.py experiment2 --config configs/experiment2.yaml
    python run.py manual-strategy --config configs/manual_strategy_config.yaml
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path for reliable imports
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root / "src"))

# Set the MARKET_DATA_DIR environment variable
os.environ["MARKET_DATA_DIR"] = str(project_root / "data")
print(f"Set MARKET_DATA_DIR to: {os.environ['MARKET_DATA_DIR']}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [command] [options]")
        print("Commands: train, evaluate, experiment1, experiment2, manual-strategy")
        return 1
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    # Check if data directory exists
    data_dir = project_root / "data"
    if not data_dir.exists():
        print(f"WARNING: Data directory not found at {data_dir}")
        print("Please ensure your data files are in the correct location.")
    else:
        print(f"Data directory found at {data_dir}")
        # List available data files
        data_files = list(data_dir.glob("*.csv"))
        if data_files:
            print(f"Found {len(data_files)} CSV files in data directory")
            if len(data_files) < 5:  # Only show if there aren't too many
                print("Available data files:")
                for file in data_files[:5]:
                    print(f"  - {file.name}")
                if len(data_files) > 5:
                    print(f"  - ... and {len(data_files) - 5} more files")
        else:
            print("No CSV files found in data directory!")
    
    if command == "train":
        from src.TradingStrategist.train import main as train_main
        sys.argv = [sys.argv[0]] + args
        return train_main()
    elif command == "evaluate":
        from src.TradingStrategist.evaluate import main as evaluate_main
        sys.argv = [sys.argv[0]] + args
        return evaluate_main()
    elif command == "experiment1":
        from src.TradingStrategist.experiments.experiment1 import main as experiment1_main
        sys.argv = [sys.argv[0]] + args
        return experiment1_main()
    elif command == "experiment2":
        from src.TradingStrategist.experiments.experiment2 import main as experiment2_main
        sys.argv = [sys.argv[0]] + args
        return experiment2_main()
    elif command == "manual-strategy":
        from src.TradingStrategist.experiments.manual_strategy_evaluation import main as manual_strategy_main
        sys.argv = [sys.argv[0]] + args
        return manual_strategy_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, evaluate, experiment1, experiment2, manual-strategy")
        return 1

if __name__ == "__main__":
    sys.exit(main())