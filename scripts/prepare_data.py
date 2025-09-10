import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ci.data.loader import build_loader_from_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    loader = build_loader_from_yaml(args.config)
    
    print("=== Multi-Dataset Loader Test ===")
    print(f"Configured datasets: {loader.cfg.datasets}")
    
    # Test PTB dataset
    if "ptb" in loader.cfg.datasets:
        print("\n--- PTB Dataset ---")
        ptb_files = loader.list_ptb_files()
        print(f"Found {len(ptb_files)} PTB files")
        if ptb_files:
            df = loader.load_file("ptb", ptb_files[0])
            df = loader.attach_labels(df, 'Normal', 'ptb')
            print(f"PTB sample shape: {df.shape}")
            print(f"PTB sampling rate: {loader.get_sampling_rate('ptb')} Hz")
            print(df.head().to_string())
    
    # Test MIT-BIH dataset
    if "mitbih" in loader.cfg.datasets:
        print("\n--- MIT-BIH Dataset ---")
        mitbih_files = loader.list_mitbih_files()
        print(f"Found {len(mitbih_files)} MIT-BIH records")
        if mitbih_files:
            df = loader.load_file("mitbih", mitbih_files[0])
            if not df.empty:
                df = loader.attach_labels(df, 'Normal', 'mitbih')
                print(f"MIT-BIH sample shape: {df.shape}")
                print(f"MIT-BIH sampling rate: {loader.get_sampling_rate('mitbih')} Hz")
                print(df.head().to_string())
            else:
                print("Failed to load MIT-BIH record")


if __name__ == '__main__':
    main()
