import argparse
from .train import train
from .inference import inference, save_images
from .api import main

def run_cli():
    parser = argparse.ArgumentParser(
        description="Train a UNet model for background removal or perform inference on an image with an existing model."
    )

    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the UNet model")
    train_parser.add_argument(
        "--epochs", 
        type=int, 
        default=20,
        help="Number of training epochs (default: 20)"
    )
    train_parser.add_argument(
        "--batch", 
        type=int, 
        default=4,
        help="Batch size for training (default: 4)"
    )
    train_parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4,
        help= "Learning rate for the optimizer (default: 1e-4)"
    )
    train_parser.add_argument(
        "--early-stopping", 
        action="store_true",
        help="Use early stopping during training"
    )
    train_parser.add_argument(
        "--output-dir", 
        type=str,
        default="./model",
        help="Path to the output directory for saving model checkpoints (default: ./model)"
    )
    train_parser.add_argument(
        "--data-dir", 
        type=str,
        default="./data",
        help="Path to the data directory (default: ./data/)"
    )
    train_parser.add_argument(
        "--resume-from", 
        type=str,
        help="Path to a model checkpoint to resume training from (default: None)"
    )
    train_parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print training progress and device information"
    )
    train_parser.add_argument(
        "--data-type",
        choices=( "flat", "split"),
        default="flat",
        help="Data directory structure: 'split' expects data/train|val/images|masks, 'flat' expects data/images|masks and auto-splits (default: flat)"
    )
    train_parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation when --data-type=flat (default: 0.2)"
    )

    # Inference subcommand
    inference_parser = subparsers.add_parser("inference", help="")
    inference_parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Path to the input image used for inference"
    )
    inference_parser.add_argument(
        "--model", 
        required=True,
        type=str, 
        help="Path to the trained UNet model"
    )
    inference_parser.add_argument(
        "--output-dir", 
        type=str,
        default="./output",
        help="Path to the output directory for saving inference results (default: ./output)"
    )

    api_parser = subparsers.add_parser("api", help="Start the FastAPI server")
    api_parser.add_argument(
        "--run",
        action="store_true",
        help="Run the FastAPI server for inference (default: False)"
    )

    args = parser.parse_args()

    if args.command == "train":
        if args.data_type == "split":
            train(f"{args.data_dir}/train/images", 
                  f"{args.data_dir}/train/masks", 
                  f"{args.data_dir}/val/images", 
                  f"{args.data_dir}/val/masks", 
                  args.epochs, 
                  args.batch, 
                  args.lr, 
                  early_stopping=args.early_stopping, 
                  resume_from=args.resume_from,
                  verbose=args.verbose)
        else:
            train(f"{args.data_dir}/images", 
                  f"{args.data_dir}/masks", 
                  None,
                  None,
                  args.epochs, 
                  args.batch, 
                  args.lr, 
                  early_stopping=args.early_stopping, 
                  resume_from=args.resume_from,
                  verbose=args.verbose,
                  val_split=args.val_split)
    elif args.command == "inference":
        img, mask = inference(args.image, args.model)
        save_images(args.output_dir, img, mask)
    elif args.command == "api":
        if args.run:
            main.run()
    else:
        parser.print_help()