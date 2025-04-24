#!/usr/bin/env python3
"""
Utility script to clear existing analysis files.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the project
sys.path.append(str(Path(__file__).parent.parent))

def clear_analysis_files():
    """Clear all analysis files in the data directory."""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Create the data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Delete all .pkl files in the data directory
    for file in data_dir.glob("*.pkl"):
        try:
            file.unlink()
            print(f"Deleted {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    print("Analysis files cleared.")

if __name__ == "__main__":
    clear_analysis_files()
