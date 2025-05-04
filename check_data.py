"""
Check data files for TradingStrategist project
"""
import os
import pandas as pd
from pathlib import Path

def check_data_files():
    # Get project root directory
    project_root = Path(__file__).parent.absolute()
    data_dir = project_root / "data"
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for data files in: {data_dir}")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found at {data_dir}")
        return False
    
    # List available data files
    data_files = list(data_dir.glob("*.csv"))
    if not data_files:
        print("ERROR: No CSV files found in data directory!")
        return False
    
    print("\nFound data files:")
    for file in data_files:
        try:
            df = pd.read_csv(file)
            print(f"  ✓ {file.name} ({len(df)} rows, {len(df.columns)} columns)")
            
            # Check for required columns
            if "Date" not in df.columns or "Adj Close" not in df.columns:
                print(f"    ⚠️ WARNING: Missing required columns in {file.name}")
                print(f"    Available columns: {', '.join(df.columns)}")
        except Exception as e:
            print(f"  ✗ Error reading {file.name}: {str(e)}")
    
    # Check for specific required files
    required_files = ["SPY.csv", "JPM.csv"]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        print("\nWARNING: Some required data files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("\nAll required data files are present.")
    return True

if __name__ == "__main__":
    check_data_files()