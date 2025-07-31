"""
train.py

Command-line script to run training for the AerialSegmentation model.
Imports common functions and dataclasses from common.py.
"""
import argparse
from dataclasses import asdict
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from common import (
    DataConfiguration,
    TrainingConfiguration,
    ModelConfiguration,
    AerialSeg_DataModule,
    load_model_for_TransferLearningandFT,
    training_validation,
    setup_system
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an Aerial Segmentation model with configurable parameters."
    )
    # Data configuration
    parser.add_argument(
        "--data-root", type=str, default="./dataset",
        help="Root directory of the dataset."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for training and validation."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of DataLoader worker processes."
    )
    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Total number of training epochs."
    )
    parser.add_argument(
        "--data-augmentation", action="store_true", dest="data_augmentation",
        help="Enable data augmentation during training."
    )
    parser.add_argument(
        "--no-data-augmentation", action="store_false", dest="data_augmentation",
        help="Disable data augmentation during training."
    )
    parser.set_defaults(data_augmentation=False)
    parser.add_argument(
        "--learning-rate", type=float, default=2e-2,
        help="Initial learning rate for the optimizer."
    )
    parser.add_argument(
        "--fine-tune-start", type=int, default=99,
        help="Index of layer-group at which to start fine-tuning."
    )
    parser.add_argument(
        "--precision", type=bool, choices=["16-mixed", "32"], default="16-mixed",
        help="Floating-point precision."
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Patience for early stopping."
    )
    parser.add_argument(
        "--load-from", type=str, default="last",
        help="Checkpoint tag to resume from (e.g., 'last' or 'best')."
    )
    parser.add_argument(
        "--image-min-size", type=int, default=720,
        help="Minimum side length to resize input images to."
    )
    # Model configuration
    parser.add_argument(
        "--model-name", type=str, default="deeplabv3_mobilenet_v3_large",
        help="Model architecture name."
    )
    parser.add_argument(
        "--use-pretrained", action="store_true", dest="use_pretrained",
        help="Use ImageNet-pretrained weights."
    )
    parser.add_argument(
        "--no-use-pretrained", action="store_false", dest="use_pretrained",
        help="Do not use pretrained weights."
    )
    parser.set_defaults(use_pretrained=True)
    parser.add_argument(
        "--aux-loss", action="store_true", dest="aux_loss",
        help="Enable auxiliary loss in model."
    )
    parser.add_argument(
        "--no-aux-loss", action="store_false", dest="aux_loss",
        help="Disable auxiliary loss in model."
    )
    parser.set_defaults(aux_loss=True)
    parser.add_argument(
        "--aux-weight", type=float, default=0.5,
        help="Weight for auxiliary loss."
    )
    parser.add_argument(
        "--num-classes", type=int, default=12,
        help="Number of segmentation classes."
    )
    # Checkpoints and logging
    parser.add_argument(
        "--ckpt-path", type=str, default=None,
        help="Path to a checkpoint to resume/fine-tune from."
    )
    parser.add_argument(
        "--log-dir", type=str, default="tb_logs",
        help="Directory in which to save TensorBoard logs."
    )
    parser.add_argument(
        "--log-name", type=str, default="segmentation_training",
        help="Name of the run for TensorBoard."
    )
    parser.add_argument(
        "--log-version", type=int, default=None,
        help="Version number for TensorBoard run (auto-increment if None)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Instantiate configurations
    data_config = DataConfiguration(
        data_root=args.data_root,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    train_config = TrainingConfiguration(
        epochs=args.epochs,
        data_augmentation=args.data_augmentation,
        learning_rate=args.learning_rate,
        fine_tune_start=args.fine_tune_start,
        precision=args.precision=="16-mixed",
        patience=args.patience,
        load_from=args.load_from,
        image_min_size=args.image_min_size
    )
    model_config = ModelConfiguration(
        model_name=args.model_name,
        use_pretrained=args.use_pretrained,
        aux_loss=args.aux_loss,
        aux_weight=args.aux_weight,
        num_classes=args.num_classes
    )

    # Set random seed and CUDNN options
    setup_system(train_config, torch)
    
    # Prepare data
    data_module = AerialSeg_DataModule(
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        image_min_size=train_config.image_min_size,
        test_csv =data_config.data_root+"/test.csv",
    )
    data_module.setup()

    # Load or initialize model
    model=None
    if args.ckpt_path:
        model = load_model_for_TransferLearningandFT(
            data_module, train_config, model_config, ckpt_path=args.ckpt_path
        )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.log_name,
        version=args.log_version
    )

    # Start training and validation
    model, data_module, best_ckpt = training_validation(
        tensorboard_logger=tb_logger,
        train_config=train_config,
        data_config=data_config,
        model_config=model_config,
        model=model,
        data_module=data_module,
        ckpt_path=args.ckpt_path
    )

    print(f"Training completed. Best checkpoint saved")


if __name__ == "__main__":
    main()