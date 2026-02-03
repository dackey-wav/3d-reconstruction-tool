"""
Dense Point Cloud Reconstruction

Creates dense point cloud by matching ALL features between camera pairs
and triangulating with quality filtering.

This gives geometrically accurate points unlike stereo matching approaches.
"""
import numpy as np
import cv2 as cv
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
import time

from .camera import Camera, CameraPose


class DenseReconstructor:
    """
    Dense point cloud from multi-view triangulation
    
    Strategy:
    1. Use dense SIFT features (MAXIMUM density)
    2. Match between all nearby camera pairs
    3. Triangulate with strict quality filtering
    4. Merge and deduplicate points
    """
    
    def __init__(self, camera: Camera):
        self.camera = camera
        
        # Dense feature extractor - ULTRA settings
        # nfeatures=0 (or usually 0 means unlimited in some versions, but stick to huge number)
        # contrastThreshold lowered to 0.01 to catch faint textures
        self.extractor = cv.SIFT_create(
            nfeatures=100000,      # Пытаемся найти до 100к точек на фото
            contrastThreshold=0.01,# Очень чувствительный порог
            edgeThreshold=20,      # Позволяем больше точек на гранях
            sigma=1.4              # Чуть меньше размытие для мелких деталей
        )
        
        # Matcher - FLANN parameters
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100) # Чуть быстрее checks, так как точек будет тьма
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        
        # Quality thresholds
        self.min_parallax = 0.3      # Снижаем требование к параллаксу (0.3 градуса)
        self.max_reproj_error = 6.0  # Чуть мягче фильтр ошибок для объема
    
    def reconstruct(self, images: List[dict],
                   poses: Dict[int, CameraPose],
                   window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dense point cloud with aggressive feature matching
        """
        print("\n" + "=" * 60)
        print("DENSE RECONSTRUCTION (HIGH DENSITY MODE)")
        print("=" * 60)
        
        camera_indices = sorted(poses.keys())
        n_cameras = len(camera_indices)
        
        # Extract dense features
        print(f"Extracting dense features from {n_cameras} images...")
        t0 = time.time()
        
        features = {}
        for idx in camera_indices:
            if idx >= len(images):
                continue
            
            # CLAHE (усиление локального контраста) помогает найти детали на однотонных поверхностях
            gray = images[idx]['gray']
            clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            kp, desc = self.extractor.detectAndCompute(enhanced_gray, None)
            if desc is not None and len(kp) > 0:
                features[idx] = {'keypoints': kp, 'descriptors': desc}
                # Оптимизация памяти: конвертируем в float32 сразу
                features[idx]['descriptors'] = features[idx]['descriptors'].astype(np.float32)

        total_kp = sum(len(f['keypoints']) for f in features.values())
        print(f"  {total_kp:,} keypoints found ({time.time()-t0:.1f}s)")
        
        # Build camera pairs
        pairs = []
        for i, idx1 in enumerate(camera_indices):
            for j, idx2 in enumerate(camera_indices):
                if j <= i:
                    continue
                # Match nearby cameras + loop closure
                if abs(i - j) <= window or abs(i - j) >= n_cameras - window:
                    pairs.append((idx1, idx2))
        
        print(f"Matching {len(pairs)} camera pairs...")
        
        # Triangulate all pairs
        all_points = []
        all_colors = []
        
        t0 = time.time()
        # Process in chunks to enable progress reporting
        mapped_points = 0
        
        for pair_idx, (idx1, idx2) in enumerate(pairs):
            if idx1 not in features or idx2 not in features:
                continue
            
            desc1 = features[idx1]['descriptors']
            desc2 = features[idx2]['descriptors']
            
            if len(desc1) < 2 or len(desc2) < 2:
                continue
            
            try:
                # k=2 for ratio test
                raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            except cv.error:
                continue
            
            # Relaxed Ratio test for Dense cloud (0.85 instead of 0.75-0.8)
            # Мы хотим больше точек, пусть и немного шумных (фильтр потом уберет)
            matches = []
            for m_n in raw_matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.85 * n.distance:
                        matches.append(m)
            
            if len(matches) < 10:
                continue
            
            # Get points
            kp1 = features[idx1]['keypoints']
            kp2 = features[idx2]['keypoints']
            
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # Triangulate
            pose1 = poses[idx1]
            pose2 = poses[idx2]
            
            points_3d, colors = self._triangulate_pair(
                pose1, pose2, pts1, pts2, images[idx1]['image']
            )
            
            if len(points_3d) > 0:
                all_points.append(points_3d)
                all_colors.append(colors)
                mapped_points += len(points_3d)
            
            if (pair_idx + 1) % 20 == 0:
                print(f"  [{pair_idx+1}/{len(pairs)}] Total accumulated: {mapped_points:,} points")
        
        if not all_points:
            print("No points generated.")
            return np.array([]), np.array([])
        
        # Merge
        print("Merging point clouds...")
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        print(f"Raw points generated: {len(points):,}")
        
        # Filter and deduplicate
        points, colors = self._filter_points(points, colors)
        
        print(f"Final filtered points: {len(points):,}")
        print(f"Dense reconstruction time: {time.time()-t0:.1f}s")
        
        return points, colors
    
    def _triangulate_pair(self, pose1: CameraPose, pose2: CameraPose,
                         pts1: np.ndarray, pts2: np.ndarray,
                         image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Triangulate points between two views"""
        
        P1 = self.camera.K @ pose1.projection_matrix
        P2 = self.camera.K @ pose2.projection_matrix
        
        # Triangulate (OpenCV DLT)
        pts4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        
        # Filter
        C1 = pose1.center
        C2 = pose2.center
        
        # Vectorized checks are faster
        # 1. Cheirality (Positive depth)
        # Transform to camera CS
        pts_c1 = (pose1.R @ pts3d.T).T + pose1.t.ravel()
        pts_c2 = (pose2.R @ pts3d.T).T + pose2.t.ravel()
        
        # Depth > 0 and Depth < reasonable max
        depth_mask = (pts_c1[:, 2] > 0.1) & (pts_c1[:, 2] < 50.0) & \
                     (pts_c2[:, 2] > 0.1) & (pts_c2[:, 2] < 50.0)
        
        pts3d = pts3d[depth_mask]
        pts1 = pts1[depth_mask]
        pts2 = pts2[depth_mask]
        
        if len(pts3d) == 0:
            return np.array([]), np.array([])
        
        # 2. Parallax Check
        vec1 = pts3d - C1
        vec2 = pts3d - C2
        norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
        
        cos_angle = np.sum(vec1 * vec2, axis=1) / (norm1.ravel() * norm2.ravel() + 1e-8)
        angles = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        parallax_mask = angles > self.min_parallax
        
        pts3d = pts3d[parallax_mask]
        pts1 = pts1[parallax_mask]
        pts2 = pts2[parallax_mask]
        
        if len(pts3d) == 0:
            return np.array([]), np.array([])

        # 3. Reprojection Error Check
        proj1 = self.camera.project(pose1.transform_points(pts3d))
        proj2 = self.camera.project(pose2.transform_points(pts3d))
        
        err1 = np.linalg.norm(proj1 - pts1, axis=1)
        err2 = np.linalg.norm(proj2 - pts2, axis=1)
        
        reproj_mask = (err1 < self.max_reproj_error) & (err2 < self.max_reproj_error)
        
        valid_indices = np.where(reproj_mask)[0]
        final_points = pts3d[valid_indices]
        final_pts2d = pts1[valid_indices]
        
        # Get colors
        h, w = image.shape[:2]
        x_coords = np.clip(final_pts2d[:, 0], 0, w-1).astype(int)
        y_coords = np.clip(final_pts2d[:, 1], 0, h-1).astype(int)
        
        colors = image[y_coords, x_coords][:, ::-1] # BGR to RGB, heavily vectorized
        
        return final_points, colors
    
    def _filter_points(self, points: np.ndarray, 
                      colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers"""
        if len(points) < 100:
            return points, colors
            
        print("  Filtering outliers...")
        
        # 1. Statistical outlier removal (OpenCV equivalent via KDTree)
        # This removes "flying pixels" noise which is common with relaxed matches
        try:
            tree = cKDTree(points)
            # Check 20 nearest neighbors
            dists, _ = tree.query(points, k=20) 
            mean_dists = np.mean(dists[:, 1:], axis=1) # skip self
            
            global_mean = np.mean(mean_dists)
            global_std = np.std(mean_dists)
            
            # Keep points within 2 sigma (standard)
            # Relaxed to 2.5 sigma to keep more "surface" points
            thresh = global_mean + 2.5 * global_std
            mask = mean_dists < thresh
            
            points = points[mask]
            colors = colors[mask]
            print(f"  Statistical filter: kept {len(points)} points")
        except Exception as e:
            print(f"  Warning: Statistical filtering failed ({e}), skipping")

        if len(points) == 0:
            return points, colors

        # 2. Voxel Grid Downsampling (Deduplication)
        # Using a finer grid to keep more density
        
        # Calculate bounding box size
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        bbox_size = np.linalg.norm(max_pt - min_pt)
        
        # Voxel size:
        # Was /500. Now /1200 means voxels are >2x smaller -> >4x more points kept
        voxel_size = bbox_size / 1200.0  
        
        voxel_indices = np.floor((points - min_pt) / voxel_size).astype(np.int64)
        
        # Fast unique hashing
        # Create a single 64-bit int hash from x,y,z (assuming grid isn't > 2^20 side)
        # Handle scaling to avoid overflow for large indices
        v_max = np.max(voxel_indices, axis=0) + 1
        
        # Simple unique checking
        # Lexicographical sort is safer than hashing for arbitrary bounds
        # Combining columns to view as strict rows
        # This acts as `unique(..., axis=0)` but slightly optimized
        dtype = [('x', np.int64), ('y', np.int64), ('z', np.int64)]
        view = np.zeros(len(voxel_indices), dtype=dtype)
        view['x'] = voxel_indices[:, 0]
        view['y'] = voxel_indices[:, 1]
        view['z'] = voxel_indices[:, 2]
        
        _, unique_idx = np.unique(view, return_index=True)
        
        print(f"  Voxel grid: downsampled to {len(unique_idx)} points")
        
        return points[unique_idx], colors[unique_idx]