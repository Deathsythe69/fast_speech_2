import argparse
import os
import sys

import yaml

# Add current folder and workspace root to path
dir_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(dir_path)
for p in [dir_path, parent_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessor.preprocessor import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telugu dataset for FastSpeech 2.")
    parser.add_argument("--config", type=str, default=os.path.join(dir_path, "config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
