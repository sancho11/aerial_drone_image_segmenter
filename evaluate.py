import argparse
import torch
from common import (
    AerialSeg_DataModule,
    load_model_for_TransferLearningandFT,
    show_predictions_with_metrics,
    generate_submission,
)
from common import DataConfiguration, TrainingConfiguration, ModelConfiguration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model and generate submission CSV"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./dataset",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="./dataset/test.csv",
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for DataModule",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--image-min-size",
        type=int,
        default=720,
        help="Minimum size to which images are resized",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to visualize with metrics",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="./predictions.png",
        help="Path to save the PIL image of predictions with metrics",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./submission.csv",
        help="Path to save generated submission CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # DataModule setup
    data_module = AerialSeg_DataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_min_size=args.image_min_size,
        test_csv=args.test_csv,
    )
    data_module.setup()

    # Configurations
    data_config = DataConfiguration(
        data_root=args.data_root,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    train_config = TrainingConfiguration()
    model_config = ModelConfiguration()

    # Load model for evaluation
    model = load_model_for_TransferLearningandFT(
        data_module,
        train_config,
        model_config,
        ckpt_path=args.ckpt_path,
    )

    idmap = {
        0:  (( 28,  49,  68), "Background"  ),  #
        1:  ((241,  91, 181), "Person"      ),  #
        2:  (( 220,  0, 115), "Bike"        ),  #
        3:  (( 220,  0, 115), "Car"         ),  #
        4:  ((252, 234, 222), "Drone"       ),  #
        5:  ((  0, 245, 212), "Boat"        ),  #
        6:  (( 94,  74, 227), "Animal"      ),  #
        7:  ((208,   0,   0), "Obstacle"    ),  #
        8:  ((255, 186,   8), "Construction"),  #
        9:  (( 81, 203,  32), "Vegetation"  ),  #
        10: ((162, 174, 187), "Road"        ),  #
        11: ((  0, 187, 249), "Sky"         ),  #
    }

    # Move model to device and set to eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Show predictions with metrics and save the returned image
    print(f"Visualizing {args.num_images} predictions with metrics...")
    pil_img = show_predictions_with_metrics(
        data_module=data_module,
        color_map=idmap,
        model=model,
        num_images=args.num_images,
    )
    if pil_img:
        pil_img.save(args.predictions_out)
        print(f"Predictions image saved to: {args.predictions_out}")

    # Generate submission file
    print(f"Generating submission CSV at {args.output_csv}...")
    generate_submission(data_module, model, output_csv=args.output_csv)
    print("Evaluation and submission generation completed.")



if __name__ == "__main__":
    main()
