**Project Overview**
- **Goal:** Detect potholes in roadway images using YOLO-based object detection. The repo provides three Jupyter notebooks covering a baseline model, transfer learning, and a multi-weather dataset workflow.
- **Platform:** Windows with Python and Jupyter Notebooks (tested in VS Code).
- **Scope:** Data preparation, training, evaluation, and visualization for pothole detection.

**Repository Contents**
- `baseline_yolo_model.ipynb`: Baseline YOLO training and evaluation workflow on a pothole dataset. Includes environment setup, dataset loading, training, inference, and basic metrics/visualizations.
- `transfer_learning_yolo model.ipynb`: Transfer learning flow that fine-tunes a YOLO model on the pothole dataset. Focuses on leveraging pre-trained weights and comparing against baseline results.
- `multiweather_pothole.ipynb`: End-to-end pipeline tailored for a multi-weather pothole dataset (e.g., images from different conditions). Covers data exploration, training runs, and performance analysis across weather subsets.

**Kaggle Usage**
- You can run `baseline_yolo_model.ipynb` and `transfer_learning_yolo model.ipynb` on Kaggle Notebooks for GPU-backed training.
- Steps:
	- Upload this repository notebooks to Kaggle or create a new Kaggle Notebook and copy the notebook cells.
	- In Notebook Settings, enable GPU (e.g., Tesla T4) and Internet.
	- Add your dataset via Kaggle Datasets or mount it using Kaggle’s `/kaggle/input/...` path. If using a YOLO YAML, point `path` to `/kaggle/working` or the dataset’s input path.
	- Install dependencies within the first cell:

		```python
		!pip install --quiet ultralytics opencv-python matplotlib pandas seaborn tqdm
		# For specific torch builds, Kaggle usually preinstalls GPU-ready torch; verify with:
		import torch; print(torch.__version__, torch.cuda.is_available())
		```

	- Adjust data paths in the notebook to Kaggle locations, e.g., `/kaggle/input/<your-dataset>/train/images`.
	- Save artifacts (weights, predictions) to `/kaggle/working` so they persist in the session output.
	- The Kaggle links are already included in the GitHub code/notebooks. You can use those links to open the exact notebooks on Kaggle and run them directly.

**Dataset**
- **Name/Type:** Pothole object detection dataset; the `multiweather_pothole.ipynb` indicates multiple weather conditions. Use any YOLO-format dataset (images + labels in TXT with class and bounding boxes) organized by train/val/test.
- **Structure (example):**
	- `datasets/pothole/train/images`, `datasets/pothole/train/labels`
	- `datasets/pothole/val/images`, `datasets/pothole/val/labels`
	- `datasets/pothole/test/images`, `datasets/pothole/test/labels`
- **Dataset YAML:** Provide a YOLO dataset YAML (e.g., `pothole.yaml`) declaring paths and classes. Example:
  
	```yaml
	path: datasets/pothole
	train: train/images
	val: val/images
	test: test/images
	names:
		0: pothole
	```

**Environment Setup**
- Install Python 3.10+ and VS Code; use Jupyter/VS Code notebooks.
- Create an environment and install dependencies commonly needed for YOLO workflows:

```powershell
# From the repo root
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# Core ML/vision stack
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# YOLO toolkit (Ultralytics). If using GPU, install the matching torch build.
pip install ultralytics
# Utilities
pip install opencv-python matplotlib pandas seaborn tqdm
```

Notes:
- If you have an NVIDIA GPU, install the CUDA-matched PyTorch wheels instead of the CPU ones. See https://pytorch.org for the correct command.
- If your notebooks reference other packages, install them similarly in the same environment.

**How to Run**
- Open the notebook in VS Code and select the `.venv` Python kernel:
	- `baseline_yolo_model.ipynb`: Run cells from top to bottom to train/evaluate a baseline YOLO model. Update dataset paths and `pothole.yaml` as needed.
	- `transfer_learning_yolo model.ipynb`: Set `weights` to a pre-trained YOLO checkpoint (e.g., `yolov8n.pt`) and run to fine-tune on the pothole dataset. Compare metrics against the baseline.
	- `multiweather_pothole.ipynb`: Point to the multi-weather dataset splits; run the exploratory cells and training/evaluation sections. Use per-weather analysis cells to compare performance.

- Kaggle (alternative run):
	- Open the notebook on Kaggle with GPU enabled and run the same training/evaluation cells as above, ensuring dataset paths reference `/kaggle/input` and outputs go to `/kaggle/working`.

Typical training snippets used by Ultralytics YOLO look like:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or a custom checkpoint
model.train(data='pothole.yaml', epochs=50, imgsz=640)
results = model.val()
model.predict(source='datasets/pothole/test/images', save=True)
```

**Usage Notes**
- Ensure your dataset is in YOLO format and paths in the notebooks point to valid locations.
- Adjust hyperparameters (e.g., `epochs`, `imgsz`, `lr`) for your hardware and dataset size.
- The notebooks include visualization cells (plots/predictions). Enable image saving if you want outputs written to disk.

**Results & Evaluation**
- After training, examine metrics such as mAP, precision/recall, and loss curves.
- Compare baseline vs. transfer learning runs to justify improvements.
- For multi-weather analysis, track per-condition performance to identify robustness.

**Troubleshooting**
- Kernel/package issues: Re-open VS Code, re-select the `.venv` kernel, and re-run `pip install` commands.
- Path errors: Verify `pothole.yaml` and folder structure. Use absolute paths on Windows if needed.
- GPU not used: Confirm CUDA drivers and that your PyTorch build matches the installed CUDA version.
# pothole_detection
By Using computer vision ( YOLO model) and use multi weather pothole detection dataset
