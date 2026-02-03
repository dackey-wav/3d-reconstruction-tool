"""
Camera model and calibration utilities
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Camera:
    """
    Camera with intrinsic parameters
    
    K - intrinsic matrix (3x3):
        [fx  0  cx]
        [0  fy  cy]
        [0   0   1]
    
    dist - distortion coefficients [k1, k2, p1, p2, k3]
    """
    K: np.ndarray
    dist: np.ndarray
    
    @property
    def fx(self) -> float:
        return self.K[0, 0]
    
    @property
    def fy(self) -> float:
        return self.K[1, 1]
    
    @property
    def cx(self) -> float:
        return self.K[0, 2]
    
    @property
    def cy(self) -> float:
        return self.K[1, 2]
    
    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: Nx3 array of 3D points in camera frame
            
        Returns:
            Nx2 array of 2D pixel coordinates
        """
        # Perspective division
        points_2d = points_3d[:, :2] / points_3d[:, 2:3]
        
        # Apply intrinsics: [u, v] = K @ [x, y, 1]
        u = self.fx * points_2d[:, 0] + self.cx
        v = self.fy * points_2d[:, 1] + self.cy
        
        return np.column_stack([u, v])
    
    def unproject(self, points_2d: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """
        Unproject 2D pixels to 3D rays
        
        Args:
            points_2d: Nx2 array of pixel coordinates
            depth: depth value (default 1.0 for unit rays)
            
        Returns:
            Nx3 array of 3D points
        """
        x = (points_2d[:, 0] - self.cx) / self.fx
        y = (points_2d[:, 1] - self.cy) / self.fy
        z = np.ones(len(points_2d)) * depth
        
        return np.column_stack([x * depth, y * depth, z])


@dataclass 
class CameraPose:
    """
    Camera pose in world coordinates
    
    R - rotation matrix (3x3): world to camera
    t - translation vector (3x1): world to camera
    
    Transform: X_camera = R @ X_world + t
    """
    R: np.ndarray
    t: np.ndarray
    
    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates: C = -R^T @ t"""
        return -self.R.T @ self.t.ravel()
    
    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 projection matrix [R|t]"""
        return np.hstack([self.R, self.t.reshape(3, 1)])
    
    def transform_points(self, points_world: np.ndarray) -> np.ndarray:
        """Transform points from world to camera frame"""
        return (self.R @ points_world.T).T + self.t.ravel()
    
    @staticmethod
    def identity() -> 'CameraPose':
        """Create identity pose (camera at origin)"""
        return CameraPose(R=np.eye(3), t=np.zeros(3))


def load_calibration(calibration_path: str) -> Camera:
    """
    Load camera calibration from .npz file
    
    Args:
        calibration_path: path to calibration_data.npz
        
    Returns:
        Camera object with loaded parameters
    """
    path = Path(calibration_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    
    data = np.load(str(path))
    
    K = data['mtx'].astype(np.float64)
    dist = data['dist'].astype(np.float64).ravel()
    
    # Ensure dist has 5 coefficients
    if len(dist) < 5:
        dist = np.pad(dist, (0, 5 - len(dist)))
    
    print(f"Loaded calibration from {path.name}")
    print(f"  Focal length: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  Principal point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    return Camera(K=K, dist=dist)