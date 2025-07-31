import argparse
import torch
from PIL import Image
from torchvision import transforms
from common import (
    AerialSeg_DataModule,
    load_model_for_TransferLearningandFT,
    num_to_rgb,
)
from common import DataConfiguration, TrainingConfiguration, ModelConfiguration

# Define color map (class index -> RGB tuple)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a single image and save the segmentation mask"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to input image for inference",
    )
    parser.add_argument(
        "--mask-out",
        type=str,
        default="./pred_mask.png",
        help="Path to save the raw predicted mask image (grayscale)",
    )
    parser.add_argument(
        "--save-color",
        action="store_true",
        help="If set, save a human-readable color segmentation image",
    )
    parser.add_argument(
        "--color-out",
        type=str,
        default="./pred_color.png",
        help="Path to save the color segmentation image",
    )
    parser.add_argument(
        "--image-min-size",
        type=int,
        default=720,
        help="Minimum size to which image is resized",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize DataModule to get normalization stats
    data_module = AerialSeg_DataModule(
        batch_size=1,
        num_workers=0,
        image_min_size=args.image_min_size,
        test_csv=None,  # not used for single-image inference
    )
    data_module.setup()

    # Configurations for model loading
    train_config = TrainingConfiguration()
    model_config = ModelConfiguration()

    # Load model
    model = load_model_for_TransferLearningandFT(
        data_module,
        train_config,
        model_config,
        ckpt_path=args.ckpt_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and preprocess image
    img = Image.open(args.image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(args.image_min_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_module.mean, std=data_module.std),
    ])
    img_tensor = preprocess(img)

    # Model inference
    input_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_batch)["out"]  # shape [1, num_classes, H, W]
        pred_mask = torch.argmax(out, dim=1).squeeze(0)  # [H, W]
    pred_mask_cpu = pred_mask.cpu()

    # Save raw mask (grayscale indices)
    mask_np = pred_mask_cpu.numpy().astype("uint8")
    mask_img = Image.fromarray(mask_np)
    mask_img.save(args.mask_out)
    print(f"Saved raw mask to {args.mask_out}")

    # Optionally save color image
    if args.save_color:
        color_np = num_to_rgb(mask_np, idmap)  # returns HxWx3 array
        color_img = Image.fromarray(color_np.astype("uint8"))
        color_img.save(args.color_out)
        print(f"Saved color mask to {args.color_out}")


if __name__ == "__main__":
    main()