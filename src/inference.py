"""
Inference Script for Multi-Weather Pothole Detection

This script provides command-line functionality to run pothole detection
on images, directories, or video files.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import PotholeDetector


def process_image(
    detector: PotholeDetector,
    image_path: str,
    output_dir: str,
    show: bool = False
) -> dict:
    """
    Process a single image for pothole detection.
    
    Args:
        detector: PotholeDetector instance
        image_path: Path to input image
        output_dir: Directory to save results
        show: Whether to display the result
        
    Returns:
        Dictionary with detection results
    """
    print(f"Processing: {image_path}")
    
    # Run detection
    detections, annotated_img = detector.detect(image_path)
    
    # Save annotated image
    filename = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{filename}_detected.jpg")
    cv2.imwrite(output_path, annotated_img)
    
    # Print results
    print(f"  Found {len(detections)} pothole(s)")
    for i, det in enumerate(detections):
        print(f"    {i+1}. Confidence: {det['confidence']:.2f}, "
              f"BBox: {det['bbox']}")
    
    # Show result if requested
    if show:
        cv2.imshow("Pothole Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return {
        "image": image_path,
        "output": output_path,
        "detections": detections,
        "count": len(detections)
    }


def process_directory(
    detector: PotholeDetector,
    input_dir: str,
    output_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
) -> list:
    """
    Process all images in a directory.
    
    Args:
        detector: PotholeDetector instance
        input_dir: Directory containing images
        output_dir: Directory to save results
        extensions: Tuple of valid image extensions
        
    Returns:
        List of detection results for all images
    """
    results = []
    image_files = []
    
    # Find all image files
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for image_path in sorted(image_files):
        result = process_image(detector, str(image_path), output_dir)
        results.append(result)
    
    # Summary
    total_potholes = sum(r["count"] for r in results)
    print(f"\nSummary:")
    print(f"  Processed: {len(results)} images")
    print(f"  Total potholes detected: {total_potholes}")
    
    return results


def process_video(
    detector: PotholeDetector,
    video_path: str,
    output_dir: str,
    show: bool = False
) -> dict:
    """
    Process a video for pothole detection.
    
    Args:
        detector: PotholeDetector instance
        video_path: Path to input video
        output_dir: Directory to save results
        show: Whether to display real-time preview
        
    Returns:
        Dictionary with detection results
    """
    print(f"Processing video: {video_path}")
    
    # Output path
    filename = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{filename}_detected.mp4")
    
    # Run detection
    all_detections = detector.detect_video(
        video_path,
        output_path=output_path,
        show_preview=show
    )
    
    # Calculate statistics
    frames_with_potholes = sum(1 for d in all_detections if d)
    total_potholes = sum(len(d) for d in all_detections)
    
    print(f"\nVideo Processing Complete:")
    print(f"  Total frames: {len(all_detections)}")
    print(f"  Frames with potholes: {frames_with_potholes}")
    print(f"  Total pothole detections: {total_potholes}")
    print(f"  Output saved to: {output_path}")
    
    return {
        "video": video_path,
        "output": output_path,
        "total_frames": len(all_detections),
        "frames_with_potholes": frames_with_potholes,
        "total_detections": total_potholes
    }


def main():
    """Main function to run inference from command line."""
    parser = argparse.ArgumentParser(
        description="Run pothole detection on images or videos"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, directory, or video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained YOLO model (default: uses pretrained YOLOv8n)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/detect",
        help="Output directory for results"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = PotholeDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    source = Path(args.source)
    
    if not source.exists():
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)
    
    # Determine source type and process
    if source.is_file():
        ext = source.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            # Single image
            process_image(detector, str(source), args.output, args.show)
        elif ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            # Video file
            process_video(detector, str(source), args.output, args.show)
        else:
            print(f"Error: Unsupported file format: {ext}")
            sys.exit(1)
    elif source.is_dir():
        # Directory of images
        process_directory(detector, str(source), args.output)
    else:
        print(f"Error: Invalid source: {args.source}")
        sys.exit(1)


if __name__ == "__main__":
    main()
