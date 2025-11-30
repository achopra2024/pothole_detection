"""
Data Preprocessing Utilities for Multi-Weather Pothole Detection

This module provides utilities for:
- Converting annotations between different formats
- Splitting datasets into train/val/test sets
- Applying weather-specific augmentations
- Organizing dataset structure
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import yaml


def create_dataset_structure(base_path: str) -> dict:
    """
    Create the required directory structure for YOLO training.
    
    Args:
        base_path: Base directory for the dataset
        
    Returns:
        Dictionary with created directory paths
    """
    dirs = {
        "train_images": os.path.join(base_path, "images", "train"),
        "val_images": os.path.join(base_path, "images", "val"),
        "test_images": os.path.join(base_path, "images", "test"),
        "train_labels": os.path.join(base_path, "labels", "train"),
        "val_labels": os.path.join(base_path, "labels", "val"),
        "test_labels": os.path.join(base_path, "labels", "test"),
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")
    
    return dirs


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_base: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[int, int, int]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_base: Base directory for output
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_count, val_count, test_count)
    """
    random.seed(seed)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    # Create output directories
    dirs = create_dataset_structure(output_base)
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    # Filter to only include images with corresponding labels
    valid_files = []
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        if os.path.exists(os.path.join(labels_dir, label_file)):
            valid_files.append(img_file)
    
    print(f"Found {len(valid_files)} valid image-label pairs")
    
    # Shuffle and split
    random.shuffle(valid_files)
    
    n_train = int(len(valid_files) * train_ratio)
    n_val = int(len(valid_files) * val_ratio)
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train + n_val]
    test_files = valid_files[n_train + n_val:]
    
    # Copy files to respective directories
    def copy_files(files: List[str], split: str):
        for img_file in files:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(dirs[f"{split}_images"], img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(dirs[f"{split}_labels"], label_file)
            shutil.copy2(src_label, dst_label)
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    print(f"\nDataset split complete:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    
    return len(train_files), len(val_files), len(test_files)


def convert_voc_to_yolo(
    xml_file: str,
    img_width: int,
    img_height: int,
    class_map: Dict[str, int]
) -> List[str]:
    """
    Convert Pascal VOC annotation to YOLO format.
    
    Args:
        xml_file: Path to VOC XML annotation file
        img_width: Image width
        img_height: Image height
        class_map: Dictionary mapping class names to class IDs
        
    Returns:
        List of YOLO format annotation strings
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_map:
            continue
        
        class_id = class_map[class_name]
        
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # Convert to YOLO format (center_x, center_y, width, height) normalized
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    
    return yolo_annotations


def convert_coco_to_yolo(
    coco_annotation: dict,
    img_width: int,
    img_height: int,
    category_map: Dict[int, int]
) -> List[str]:
    """
    Convert COCO annotation to YOLO format.
    
    Args:
        coco_annotation: COCO annotation dictionary
        img_width: Image width
        img_height: Image height
        category_map: Dictionary mapping COCO category IDs to YOLO class IDs
        
    Returns:
        List of YOLO format annotation strings
    """
    yolo_annotations = []
    
    for ann in coco_annotation:
        category_id = ann["category_id"]
        if category_id not in category_map:
            continue
        
        class_id = category_map[category_id]
        
        # COCO format: [x_min, y_min, width, height]
        x_min, y_min, w, h = ann["bbox"]
        
        # Convert to YOLO format
        x_center = (x_min + w / 2) / img_width
        y_center = (y_min + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    
    return yolo_annotations


def apply_weather_augmentation(
    image: np.ndarray,
    weather_type: str,
    intensity: float = 0.5
) -> np.ndarray:
    """
    Apply weather-specific augmentations to simulate different conditions.
    
    Args:
        image: Input image as numpy array
        weather_type: Type of weather ('rain', 'fog', 'snow', 'night', 'overcast')
        intensity: Augmentation intensity (0.0 to 1.0)
        
    Returns:
        Augmented image
    """
    augmented = image.copy()
    
    if weather_type == "rain":
        # Simulate rain with streaks
        rain_layer = np.zeros_like(augmented)
        for _ in range(int(500 * intensity)):
            x = random.randint(0, augmented.shape[1] - 1)
            y = random.randint(0, augmented.shape[0] - 20)
            length = random.randint(5, 20)
            cv2.line(
                rain_layer,
                (x, y),
                (x + random.randint(-2, 2), y + length),
                (200, 200, 200),
                1
            )
        augmented = cv2.addWeighted(augmented, 1.0, rain_layer, intensity * 0.5, 0)
        
        # Darken image slightly
        augmented = cv2.convertScaleAbs(augmented, alpha=1.0 - intensity * 0.2, beta=0)
        
    elif weather_type == "fog":
        # Simulate fog with gaussian blur and brightness
        fog_layer = np.ones_like(augmented) * 255
        fog_layer = cv2.GaussianBlur(fog_layer, (21, 21), 0)
        augmented = cv2.addWeighted(augmented, 1.0 - intensity * 0.6, fog_layer, intensity * 0.6, 0)
        
    elif weather_type == "snow":
        # Simulate snow with white spots
        snow_layer = np.zeros_like(augmented)
        for _ in range(int(1000 * intensity)):
            x = random.randint(0, augmented.shape[1] - 1)
            y = random.randint(0, augmented.shape[0] - 1)
            radius = random.randint(1, 3)
            cv2.circle(snow_layer, (x, y), radius, (255, 255, 255), -1)
        augmented = cv2.addWeighted(augmented, 1.0, snow_layer, intensity * 0.5, 0)
        
        # Brighten image
        augmented = cv2.convertScaleAbs(augmented, alpha=1.0 + intensity * 0.2, beta=10)
        
    elif weather_type == "night":
        # Simulate night/low light conditions
        augmented = cv2.convertScaleAbs(augmented, alpha=1.0 - intensity * 0.6, beta=-50 * intensity)
        
        # Add slight blue tint
        augmented[:, :, 0] = np.clip(augmented[:, :, 0] + 20 * intensity, 0, 255)
        
    elif weather_type == "overcast":
        # Simulate overcast/cloudy conditions
        augmented = cv2.convertScaleAbs(augmented, alpha=1.0 - intensity * 0.2, beta=-20 * intensity)
        
        # Reduce saturation
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - intensity * 0.3)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return augmented


def generate_weather_variants(
    images_dir: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    weather_types: List[str] = None,
    intensity_range: Tuple[float, float] = (0.3, 0.7)
) -> int:
    """
    Generate weather-augmented variants of images.
    
    Args:
        images_dir: Directory containing original images
        labels_dir: Directory containing label files
        output_images_dir: Directory to save augmented images
        output_labels_dir: Directory to save copied labels
        weather_types: List of weather types to apply
        intensity_range: Range of augmentation intensities
        
    Returns:
        Number of augmented images generated
    """
    if weather_types is None:
        weather_types = ["rain", "fog", "snow", "night", "overcast"]
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    count = 0
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        for weather in weather_types:
            intensity = random.uniform(*intensity_range)
            augmented = apply_weather_augmentation(image, weather, intensity)
            
            # Save augmented image
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            new_img_name = f"{base_name}_{weather}{ext}"
            new_label_name = f"{base_name}_{weather}.txt"
            
            cv2.imwrite(os.path.join(output_images_dir, new_img_name), augmented)
            shutil.copy2(label_path, os.path.join(output_labels_dir, new_label_name))
            
            count += 1
    
    print(f"Generated {count} weather-augmented images")
    return count


def create_dataset_yaml(
    output_path: str,
    dataset_path: str,
    class_names: List[str],
    train_path: str = "images/train",
    val_path: str = "images/val",
    test_path: str = "images/test"
) -> str:
    """
    Create a YOLO dataset configuration YAML file.
    
    Args:
        output_path: Path to save the YAML file
        dataset_path: Root path of the dataset
        class_names: List of class names
        train_path: Relative path to training images
        val_path: Relative path to validation images
        test_path: Relative path to test images
        
    Returns:
        Path to created YAML file
    """
    config = {
        "path": dataset_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data preprocessing utilities for pothole detection"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Split dataset command
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("--images", required=True, help="Images directory")
    split_parser.add_argument("--labels", required=True, help="Labels directory")
    split_parser.add_argument("--output", required=True, help="Output directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.7)
    split_parser.add_argument("--val-ratio", type=float, default=0.2)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    
    # Weather augmentation command
    aug_parser = subparsers.add_parser("augment", help="Generate weather variants")
    aug_parser.add_argument("--images", required=True, help="Images directory")
    aug_parser.add_argument("--labels", required=True, help="Labels directory")
    aug_parser.add_argument("--output-images", required=True)
    aug_parser.add_argument("--output-labels", required=True)
    aug_parser.add_argument(
        "--weather",
        nargs="+",
        default=["rain", "fog", "snow", "night", "overcast"]
    )
    
    args = parser.parse_args()
    
    if args.command == "split":
        split_dataset(
            args.images,
            args.labels,
            args.output,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
    elif args.command == "augment":
        generate_weather_variants(
            args.images,
            args.labels,
            args.output_images,
            args.output_labels,
            args.weather
        )
