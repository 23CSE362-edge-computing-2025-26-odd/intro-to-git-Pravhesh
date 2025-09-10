import os
import shutil
import tempfile
import yaml
import pandas as pd
import pytest

from src.ci.data.loader import build_loader_from_yaml


def make_temp_env(tmpdir):
    base = tmpdir
    ptb_dir = os.path.join(base, 'ptb')
    mitbih_dir = os.path.join(base, 'mitbih')
    os.makedirs(ptb_dir, exist_ok=True)
    os.makedirs(mitbih_dir, exist_ok=True)

    # Create a small PTB CSV (with lead II)
    csv_path = os.path.join(ptb_dir, 'sample.csv')
    pd.DataFrame({
        'i': [0.1, 0.2, 0.15, 0.05],
        'ii': [0.3, 0.4, 0.35, 0.25],
        'iii': [0.2, 0.2, 0.2, 0.2],
    }).to_csv(csv_path, index=False)

    # Labels mapping
    labels_yaml = {
        'classes': {
            'Normal': {'unified': 'Normal', 'risk': 0},
            'AtRisk': {'unified': 'At-Risk', 'risk': 50},
            'Critical': {'unified': 'Critical', 'risk': 90},
        }
    }

    cfg = {
        'paths': {
            'ptb_data_dir': ptb_dir,
            'mitbih_data_dir': mitbih_dir,
        },
        'loader': {
            'datasets': ['ptb'],
            'ptb': {
                'timestamp_column': None,
                'value_column': 'ii',
                'sampling_rate_hz': 1000,
                'file_glob': '**/*.csv'
            },
            'mitbih': {
                'sampling_rate_hz': 360
            }
        },
        'labels': {
            'mapping_file': os.path.join(base, 'labels.yaml')
        }
    }

    cfg_path = os.path.join(base, 'config.yaml')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)

    with open(cfg['labels']['mapping_file'], 'w', encoding='utf-8') as f:
        yaml.safe_dump(labels_yaml, f)

    return cfg_path, csv_path


def test_loader_lists_and_reads(tmp_path):
    cfg_path, csv_path = make_temp_env(str(tmp_path))
    loader = build_loader_from_yaml(cfg_path)

    files = loader.list_ptb_files()
    assert len(files) == 1
    assert files[0].endswith('sample.csv')

    df = loader.load_file("ptb", files[0])
    assert list(df.columns) == ['time', 'value']
    assert len(df) == 4


def test_label_mapping(tmp_path):
    cfg_path, csv_path = make_temp_env(str(tmp_path))
    loader = build_loader_from_yaml(cfg_path)

    df = loader.load_file("ptb", csv_path)
    df2 = loader.attach_labels(df, 'AtRisk', 'ptb')
    assert 'label' in df2 and 'risk_target' in df2 and 'dataset' in df2
    assert df2['label'].iloc[0] == 'At-Risk'
    assert df2['risk_target'].iloc[0] == 50
    assert df2['dataset'].iloc[0] == 'ptb'

    df3 = loader.attach_labels(df, 'UnknownClass', 'ptb')
    assert df3['label'].iloc[0] == 'Unknown'
    assert df3['risk_target'].iloc[0] == 0
