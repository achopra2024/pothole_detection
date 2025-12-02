"""
YOLO-based Pothole Detection Module

This module provides functionality for detecting potholes in images
using the YOLO (You Only Look Once) deep learning model.
Designed to work with multi-weather conditions including:
- Clear/Sunny
- Rainy/Wet roads
- Foggy
- Snowy
- Night/Low light
- Overcast/Cloudy
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


class PotholeDetector:
    """
    A class for detecting potholes in images using YOLO model.
    
    Attributes:
        model: The YOLO model used for detection
        confidence_threshold: Minimum confidence score for detections
        iou_threshold: IoU threshold for Non-Maximum Suppression
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize the PotholeDetector.
        
        Args:
            model_path: Path to a trained YOLO model. If None, uses a default model.
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use YOLOv8n as default base model
            self.model = YOLO("yolov8n.pt")
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        save_result: bool = False,
        output_path: Optional[str] = None
    ) -> Tuple[List[dict], np.ndarray]:
        """
        Detect potholes in an image.
        
        Args:
            image: Path to image file or numpy array of the image
            save_result: Whether to save the annotated result
            output_path: Path to save the annotated image
            
        Returns:
            Tuple containing:
                - List of detection dictionaries with keys:
                    'bbox': [x1, y1, x2, y2]
                    'confidence': float
                    'class': str (always 'pothole')
                - Annotated image as numpy array
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img = image.copy()
        
        # Run inference
        results = self.model(
            img,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        annotated_img = img.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class': 'pothole'
                    }
                    detections.append(detection)
                    
                    # Draw bounding box on annotated image
                    cv2.rectangle(
                        annotated_img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),  # Red color for potholes
                        2
                    )
                    
                    # Add label with confidence
                    label = f"Pothole: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        annotated_img,
                        (int(x1), int(y1) - label_size[1] - 10),
                        (int(x1) + label_size[0], int(y1)),
                        (0, 0, 255),
                        -1
                    )
                    cv2.putText(
                        annotated_img,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
        
        # Save result if requested
        if save_result and output_path:
            cv2.imwrite(output_path, annotated_img)
        
        return detections, annotated_img
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8
    ) -> List[Tuple[List[dict], np.ndarray]]:
        """
        Detect potholes in multiple images.
        
        Args:
            images: List of image paths or numpy arrays
            batch_size: Number of images to process at once
            
        Returns:
            List of tuples, each containing detections and annotated image
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                result = self.detect(img)
                results.append(result)
        return results
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = False
    ) -> List[List[dict]]:
        """
        Detect potholes in a video.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the annotated video
            show_preview: Whether to show real-time preview
            
        Returns:
            List of detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections, annotated_frame = self.detect(frame)
            all_detections.append(detections)
            
            if writer:
                writer.write(annotated_frame)
            
            if show_preview:
                cv2.imshow("Pothole Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        return all_detections


def create_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> PotholeDetector:
    """
    Factory function to create a PotholeDetector instance.
    
    Args:
        model_path: Path to trained YOLO model
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        PotholeDetector instance
    """
    return PotholeDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )
