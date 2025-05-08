# check_environment.py
import os
import shutil
import sys

def check_environment():
    print("Checking project environment...")
    
    # Create required directories
    required_dirs = ['data/raw', 'data/processed', 'models']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")
    
    # Check for CSV files
    csv_files = ['bau515od5155.csv', 'bau515od5156.csv']
    for filename in csv_files:
        raw_path = f'data/raw/{filename}'
        if os.path.exists(raw_path):
            print(f"Found {filename} in data/raw/")
        elif os.path.exists(filename):
            shutil.move(filename, raw_path)
            print(f"Moved {filename} to data/raw/")
        else:
            print(f"ERROR: {filename} not found. Please place it in data/raw/")
            return False
    
    return True

if __name__ == "__main__":
    if check_environment():
        print("\nEnvironment check passed. You can proceed with the data preparation.")
        print("Run: python scripts/data_preparation.py")
    else:
        print("\nEnvironment check failed. Please fix the issues before proceeding.")
        sys.exit(1)