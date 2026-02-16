import argparse
from .train import train

def run_cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    inference_parser = subparsers.add_parser("inference")
    inference_parser.add_argument("--image", type=str, required=True)
    inference_parser.add_argument("--model", type=str, default="unet_bg_removal.pth")

    args = parser.parse_args()

    if args.command == "train":
        train("data/train/images", "images/train/masks", "data/val/images", "images/val/masks", args.epochs, args.batch, args.lr)
    else:
        parser.print_help()