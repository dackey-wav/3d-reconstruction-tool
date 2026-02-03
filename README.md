# 3D Reconstruction Tool

A PyQt5-based application for creating 3D point cloud models from 2D images using Structure from Motion (SfM) and dense reconstruction techniques.

The tool supports multiple reconstruction pipelines: custom SIFT/Neural feature matching with sparse/dense reconstruction, and COLMAP integration for high-quality results.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

2. Install dependencies
```python
pip install -r requirements.txt
```

3. (Optional) For neural matching with LightGlue:
```python
pip install torch torchvision
pip install git+https://github.com/cvg/LightGlue.git
```

## Usage

Camera Calibration (First Time Only)
Place chessboard calibration images in data/calibration_images/chessboard/ and run
```python
python -m src.calibration.calibration
```

View 3D Models
Launch the GUI viewer to explore existing PLY point clouds:
```python
python -m src.gui.main_app
```
Then select the "View Model" tab and open a .ply file.

Create 3D Models
**Option 1: GUI (Recommended)**
```python
python -m src.gui.main_app
```
    1. Go to "Create Model" tab
    2. Select dataset from dropdown
    3. Choose reconstruction method
    4. Click "START RECONSTRUCTION"

**Option 2: Command Line**
```python
# Place images in data/samples/your_dataset/
python -m src.run_reconstruction your_dataset --neural --mvs
```

Available options:

- --neural - Use LightGlue neural matcher (recommended)
- --mvs - PatchMatch MVS dense reconstruction (best quality, GPU)
- --stereo - Plane sweep stereo (faster, GPU)
- --fast - Quick sparse-only mode
- --max-images N - Limit number of images


## Project Structure

├── data/
│   ├── samples/          # Input image datasets
│   └── calibration_images/
├── src/
│   ├── core/            # Reconstruction algorithms
│   ├── gui/             # PyQt5 interface
│   ├── calibration/     # Camera calibration tools
│   └── tools/           # COLMAP integration
└── requirements.txt