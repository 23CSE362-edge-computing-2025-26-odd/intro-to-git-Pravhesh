import argparse
from ci.data.loader import build_loader_from_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    loader = build_loader_from_yaml(args.config)
    files = loader.list_files()
    print(f"Found {len(files)} files")
    if files:
        df = loader.load_csv(files[0])
        df = loader.attach_labels(df, 'Normal')
        print(df.head().to_string())


if __name__ == '__main__':
    main()
