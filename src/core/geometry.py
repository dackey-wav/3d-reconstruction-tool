"""
Geometric primitives for 3D reconstruction

Contains:
- Triangulation (DLT method)
- Essential matrix decomposition
- Epipolar geometry utilities
"""
import numpy as np
import cv2 as cv
from typing import Tuple, Optional
from .camera import Camera, CameraPose


def triangulate_points(camera: Camera,
                      pose1: CameraPose,
                      pose2: CameraPose,
                      points1: np.ndarray,
                      points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from two views using DLT
    
    Args:
        camera: Camera with intrinsics
        pose1: first camera pose
        pose2: second camera pose
        points1: Nx2 points in first image
        points2: Nx2 points in second image
        
    Returns:
        points_3d: Nx3 triangulated points
        valid_mask: boolean mask of valid points
    """
    if len(points1) == 0:
        return np.array([]), np.array([], dtype=bool)
    
    # Build projection matrices: P = K @ [R|t]
    P1 = camera.K @ pose1.projection_matrix
    P2 = camera.K @ pose2.projection_matrix
    
    # OpenCV triangulation (DLT)
    points_4d = cv.triangulatePoints(P1, P2, points1.T, points2.T)
    
    # Convert from homogeneous coordinates
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    # Validate triangulated points
    valid_mask = validate_triangulation(
        camera, pose1, pose2, points_3d, points1, points2
    )
    
    return points_3d, valid_mask


def validate_triangulation(camera: Camera,
                          pose1: CameraPose,
                          pose2: CameraPose,
                          points_3d: np.ndarray,
                          points1: np.ndarray,
                          points2: np.ndarray,
                          max_reproj_error: float = 4.0,
                          min_parallax_deg: float = 1.0) -> np.ndarray:
    """
    Validate triangulated points by checking:
    1. Positive depth in both cameras
    2. Reasonable depth range
    3. Sufficient parallax angle
    4. Low reprojection error
    
    Returns:
        Boolean mask of valid points
    """
    n_points = len(points_3d)
    valid = np.ones(n_points, dtype=bool)
    
    # Camera centers
    C1 = pose1.center
    C2 = pose2.center
    baseline = np.linalg.norm(C2 - C1)
    
    for i in range(n_points):
        pt = points_3d[i]
        
        # 1. Check depth in camera 1
        pt_cam1 = pose1.transform_points(pt.reshape(1, 3))[0]
        if pt_cam1[2] <= 0.01:
            valid[i] = False
            continue
        
        # 2. Check depth in camera 2
        pt_cam2 = pose2.transform_points(pt.reshape(1, 3))[0]
        if pt_cam2[2] <= 0.01:
            valid[i] = False
            continue
        
        # 3. Check reasonable depth range (relative to baseline)
        max_depth = baseline * 200
        if pt_cam1[2] > max_depth or pt_cam2[2] > max_depth:
            valid[i] = False
            continue
        
        # 4. Check parallax angle
        ray1 = pt - C1
        ray2 = pt - C2
        cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-8)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        if angle_deg < min_parallax_deg:
            valid[i] = False
            continue
        
        # 5. Check reprojection error
        # Project back to image 1
        proj1 = camera.project(pt_cam1.reshape(1, 3))[0]
        err1 = np.linalg.norm(proj1 - points1[i])
        
        # Project back to image 2
        proj2 = camera.project(pt_cam2.reshape(1, 3))[0]
        err2 = np.linalg.norm(proj2 - points2[i])
        
        if err1 > max_reproj_error or err2 > max_reproj_error:
            valid[i] = False
            continue
    
    return valid


def compute_essential_matrix(camera: Camera, F: np.ndarray) -> np.ndarray:
    """
    Compute Essential matrix from Fundamental matrix
    
    E = K^T @ F @ K
    """
    return camera.K.T @ F @ camera.K


def decompose_essential(E: np.ndarray,
                        camera: Camera,
                        points1: np.ndarray,
                        points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose Essential matrix into R and t
    
    Uses OpenCV's recoverPose which:
    1. Decomposes E into 4 possible (R, t) combinations
    2. Tests all 4 using cheirality constraint
    3. Returns the one where most points have positive depth
    
    Args:
        E: Essential matrix
        camera: Camera with intrinsics
        points1: Nx2 points in first image
        points2: Nx2 points in second image
        
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector (unit length)
        mask: inlier mask from recoverPose
    """
    _, R, t, mask = cv.recoverPose(E, points1, points2, camera.K)
    return R, t, mask


def compute_reprojection_error(camera: Camera,
                               pose: CameraPose,
                               points_3d: np.ndarray,
                               points_2d: np.ndarray) -> np.ndarray:
    """
    Compute reprojection error for points
    
    Returns:
        Per-point reprojection errors (in pixels)
    """
    # Transform to camera frame
    points_cam = pose.transform_points(points_3d)
    
    # Project to image
    points_proj = camera.project(points_cam)
    
    # Compute errors
    errors = np.linalg.norm(points_proj - points_2d, axis=1)
    
    return errors