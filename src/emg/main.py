import argparse
import torch

from .train_lstm import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description="EMG LSTM entry point")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run LSTM training/evaluation pipeline",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        train_main()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("EMG module ready. Specify --train to run the training pipeline.")
        print(f"Detected device: {device}")


if __name__ == "__main__":
    main()


