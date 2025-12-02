# Pothole Detection using YOLO

A computer vision system for detecting potholes in road images and videos using YOLO (You Only Look Once) deep learning model. This project is designed to work with multi-weather pothole detection datasets, ensuring robust detection performance across various weather conditions.

## Features

- **YOLO-based Detection**: Uses YOLOv8 for fast and accurate pothole detection
- **Multi-Weather Support**: Trained and tested on images captured in various weather conditions:
  - Clear/Sunny
  - Rainy/Wet roads
  - Foggy
  - Snowy
  - Night/Low light
  - Overcast/Cloudy
- **Flexible Input**: Supports images, directories of images, and video files
- **Training Pipeline**: Complete training pipeline with data augmentation
- **Data Preprocessing**: Tools for dataset preparation and weather-specific augmentation

## Project Structure

```
pothole_detection/
├── config/
│   └── dataset.yaml          # Dataset configuration for YOLO
├── src/
│   ├── __init__.py
│   ├── detector.py           # Main detection module
│   ├── train.py              # Training script
│   ├── inference.py          # Inference/prediction script
│   └── preprocess.py         # Data preprocessing utilities
├── models/                   # Directory for trained models
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/achopra2024/pothole_detection.git
cd pothole_detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Detection

```python
from src.detector import PotholeDetector

# Initialize detector
detector = PotholeDetector(
    model_path="models/best.pt",  # Path to your trained model
    confidence_threshold=0.25
)

# Detect potholes in an image
detections, annotated_image = detector.detect("path/to/image.jpg")

# Print results
for detection in detections:
    print(f"Pothole found at {detection['bbox']} with confidence {detection['confidence']:.2f}")
```

### Command Line - Inference

```bash
# Detect potholes in a single image
python src/inference.py --source path/to/image.jpg --model models/best.pt --output results/

# Process a directory of images
python src/inference.py --source path/to/images/ --model models/best.pt --output results/

# Process a video file
python src/inference.py --source path/to/video.mp4 --model models/best.pt --output results/
```

### Training

1. Prepare your dataset in YOLO format:
```
datasets/pothole_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. Update `config/dataset.yaml` with your dataset path.

3. Run training:
```bash
python src/train.py --data config/dataset.yaml --model yolov8n.pt --epochs 100 --batch-size 16
```

Training options:
- `--model`: Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--img-size`: Input image size
- `--device`: Training device ('', 'cpu', '0', '0,1', etc.)
- `--patience`: Early stopping patience

### Data Preprocessing

Split a dataset into train/val/test sets:
```bash
python src/preprocess.py split --images data/images --labels data/labels --output datasets/pothole_dataset
```

Generate weather-augmented images:
```bash
python src/preprocess.py augment --images data/images --labels data/labels \
    --output-images data/augmented/images --output-labels data/augmented/labels \
    --weather rain fog snow night
```

## Model Performance

The YOLO model is optimized for real-time pothole detection across various weather conditions. Expected performance metrics:
- **mAP50**: ~0.85 (may vary based on dataset quality)
- **Inference Speed**: ~30 FPS on GPU, ~5 FPS on CPU
- **Input Resolution**: 640x640 (default)

## Multi-Weather Dataset

This project is designed to work with multi-weather pothole detection datasets. The data augmentation pipeline includes:
- Rain simulation with streaks
- Fog/mist effects
- Snow particle overlay
- Night/low-light conditions
- Overcast/cloudy lighting

## API Reference

### PotholeDetector Class

```python
class PotholeDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize the PotholeDetector.
        
        Args:
            model_path: Path to a trained YOLO model
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
        """
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        save_result: bool = False,
        output_path: Optional[str] = None
    ) -> Tuple[List[dict], np.ndarray]:
        """
        Detect potholes in an image.
        
        Returns:
            Tuple of (detections list, annotated image)
        """
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = False
    ) -> List[List[dict]]:
        """
        Detect potholes in a video.
        
        Returns:
            List of detections for each frame
        """
```

## Requirements

- Python 3.8+
- PyTorch 1.7+
- ultralytics (YOLOv8)
- OpenCV
- NumPy
- See `requirements.txt` for complete list

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Multi-weather pothole detection research community
