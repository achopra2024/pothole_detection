"""
YOLO Training Script for Multi-Weather Pothole Detection

This script provides functionality to train a YOLO model for pothole detection
using the multi-weather pothole detection dataset.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


def train_pothole_detector(
    data_config: str = "config/dataset.yaml",
    model: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "",
    project: str = "runs/train",
    name: str = None,
    resume: bool = False,
    patience: int = 50,
    workers: int = 8,
    pretrained: bool = True,
    augment: bool = True,
):
    """
    Train a YOLO model for pothole detection.
    
    Args:
        data_config: Path to dataset configuration YAML file
        model: Path to pretrained model or model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        device: Training device ('', 'cpu', '0', '0,1,2,3', etc.)
        project: Project directory for saving results
        name: Experiment name
        resume: Resume training from last checkpoint
        patience: Early stopping patience (epochs without improvement)
        workers: Number of data loader workers
        pretrained: Use pretrained weights
        augment: Enable data augmentation
        
    Returns:
        Path to the best trained model
    """
    # Generate experiment name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"pothole_detector_{timestamp}"
    
    # Load the model
    print(f"Loading model: {model}")
    yolo_model = YOLO(model)
    
    # Training configuration
    train_args = {
        "data": data_config,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "patience": patience,
        "project": project,
        "name": name,
        "device": device if device else None,
        "workers": workers,
        "pretrained": pretrained,
        "resume": resume,
        
        # Augmentation settings for multi-weather robustness
        "hsv_h": 0.015 if augment else 0.0,  # Hue augmentation
        "hsv_s": 0.7 if augment else 0.0,    # Saturation augmentation
        "hsv_v": 0.4 if augment else 0.0,    # Value augmentation
        "degrees": 10.0 if augment else 0.0,  # Rotation
        "translate": 0.1 if augment else 0.0, # Translation
        "scale": 0.5 if augment else 0.0,     # Scale
        "shear": 5.0 if augment else 0.0,     # Shear
        "flipud": 0.0,                         # No vertical flip for potholes
        "fliplr": 0.5 if augment else 0.0,    # Horizontal flip
        "mosaic": 1.0 if augment else 0.0,    # Mosaic augmentation
        "mixup": 0.1 if augment else 0.0,     # Mixup augmentation
    }
    
    print(f"\nStarting training with configuration:")
    print(f"  Dataset: {data_config}")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Augmentation: {augment}")
    print(f"  Output: {project}/{name}")
    print()
    
    # Train the model
    results = yolo_model.train(**train_args)
    
    # Path to the best model
    best_model_path = Path(project) / name / "weights" / "best.pt"
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {best_model_path}")
    
    return str(best_model_path)


def validate_model(
    model_path: str,
    data_config: str = "config/dataset.yaml",
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "",
):
    """
    Validate a trained YOLO model.
    
    Args:
        model_path: Path to trained model
        data_config: Path to dataset configuration
        img_size: Input image size
        batch_size: Validation batch size
        device: Validation device
        
    Returns:
        Validation metrics dictionary
    """
    print(f"Validating model: {model_path}")
    
    model = YOLO(model_path)
    
    results = model.val(
        data=data_config,
        imgsz=img_size,
        batch=batch_size,
        device=device if device else None,
    )
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }


def main():
    """Main function to run training from command line."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model for multi-weather pothole detection"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="config/dataset.yaml",
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Training device ('' for auto, 'cpu', '0', '0,1', etc.)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory for results"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training"
    )
    
    args = parser.parse_args()
    
    # Train the model
    best_model = train_pothole_detector(
        data_config=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers,
        augment=not args.no_augment,
    )
    
    # Validate if requested
    if args.validate:
        validate_model(
            model_path=best_model,
            data_config=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
