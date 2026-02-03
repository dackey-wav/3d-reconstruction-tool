"""
Utility functions for 3D reconstruction
"""
import numpy as np
from pathlib import Path


def save_ply(points: np.ndarray, colors: np.ndarray, output_path: str):
    """
    Save point cloud to PLY format
    
    Args:
        points: Nx3 array of 3D coordinates
        colors: Nx3 array of RGB colors (0-255)
        output_path: path to save PLY file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"Saved {len(points):,} points to {output_path}")


def save_cameras_ply(poses: dict, output_path: str, scale: float = 0.5):
    """
    Save camera positions as PLY for visualization
    
    Args:
        poses: dict of {idx: CameraPose}
        output_path: path to save PLY
        scale: size of camera frustum
    """
    output_path = Path(output_path)
    
    points = []
    colors = []
    
    for idx, pose in poses.items():
        center = pose.center
        
        # Camera center (red)
        points.append(center)
        colors.append([255, 0, 0])
        
        # Camera direction (green)
        forward = -pose.R[2, :]  # Camera looks along -Z
        points.append(center + forward * scale)
        colors.append([0, 255, 0])
    
    points = np.array(points)
    colors = np.array(colors, dtype=np.uint8)
    
    save_ply(points, colors, str(output_path))


def compute_scene_bounds(points: np.ndarray) -> dict:
    """Compute bounding box and statistics of point cloud"""
    if len(points) == 0:
        return {'min': np.zeros(3), 'max': np.zeros(3), 'center': np.zeros(3), 'size': 0}
    
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    center = (min_pt + max_pt) / 2
    size = np.linalg.norm(max_pt - min_pt)
    
    return {
        'min': min_pt,
        'max': max_pt,
        'center': center,
        'size': size
    }