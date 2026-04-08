# MASt3R → COLMAP → 3D Gaussian Splatting Pipeline

A Gradio UI that runs MASt3R 3D reconstruction and exports a COLMAP-format dataset ready for 3DGS training.

MASt3R improves over DUSt3R with dense local descriptors for much more accurate matching, metric-scale reconstruction, and sparse global alignment.

## Setup

```bash
# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# (Optional) Build RoPE CUDA kernels for faster inference
cd mast3r/dust3r/croco/models/curope && python setup.py build_ext --inplace && cd ../../../../..
```

## Usage

```bash
python app.py
```

Or double-click `launch.bat`.

Then open http://127.0.0.1:7860 in your browser.

## Workflow

1. **Upload images** of your scene
2. **Configure** scene graph, optimization level, iterations
3. **Reconstruct** — runs MASt3R sparse global alignment
4. **Preview** — view 3D model, adjust confidence threshold
5. **Export** — download COLMAP dataset zip
6. **Train 3DGS**:

```bash
python train.py -s path/to/exported/colmap_dataset
```

## Output Format

```
colmap_dataset/
  images/           # original full-resolution images
  sparse/0/
    cameras.txt     # PINHOLE intrinsics (scaled to original resolution)
    images.txt      # camera poses
    points3D.txt    # initial 3D point cloud
```
