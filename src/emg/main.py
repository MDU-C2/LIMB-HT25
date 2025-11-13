import argparse
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="EMG LSTM entry point")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run LSTM training/evaluation pipeline (passes all other args to train_lstm)",
    )
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create sequence dataset from raw .mat files (passes all other args to create_sequence_dataset)",
    )
    # Parse known args only, pass rest to submodules
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    if args.train:
        # Import here to avoid circular imports
        # Pass remaining args to train_lstm by modifying sys.argv
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]] + remaining
        try:
            from .train_lstm import main as train_main
            train_main()
        finally:
            sys.argv = original_argv
    elif args.create_dataset:
        # Pass remaining args to create_sequence_dataset
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]] + remaining
        try:
            from .create_sequence_dataset import main as create_dataset_main
            exit(create_dataset_main())
        finally:
            sys.argv = original_argv
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("EMG module ready.")
        logger.info("Options:")
        logger.info("  --train: Run LSTM training pipeline")
        logger.info("  --create-dataset: Create sequence dataset from raw .mat files")
        logger.info(f"Detected device: {device}")
        logger.info("Use --help with --train or --create-dataset for more options")


if __name__ == "__main__":
    main()


