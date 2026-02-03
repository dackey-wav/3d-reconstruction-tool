"""
Dense Stereo Reconstruction using Plane Sweep with GPU acceleration

Key improvements:
- GPU acceleration via PyTorch
- Multi-view consistency check (point must be visible from 3+ cameras)
- Adaptive depth sampling
- Aggressive filtering
"""
import numpy as np
import cv2 as cv
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .camera import Camera, CameraPose


class DenseStereoReconstructor:
    """
    GPU-accelerated dense reconstruction using plane sweep stereo
    with multi-view consistency filtering
    """
    
    def __init__(self, camera: Camera, 
                 scale: float = 0.25,
                 num_depths: int = 64,
                 patch_size: int = 5,
                 min_views: int = 3,
                 consistency_thresh: float = 0.8):
        
        self.camera = camera
        self.scale = scale
        self.num_depths = num_depths
        self.patch_size = patch_size
        self.min_views = min_views
        self.consistency_thresh = consistency_thresh
        
        # Use GPU if available
        if HAS_TORCH and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Dense stereo using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("Dense stereo using CPU (install PyTorch with CUDA for GPU)")
        
        # Scaled camera matrix
        self.K_scaled = camera.K.copy()
        self.K_scaled[0, 0] *= scale
        self.K_scaled[1, 1] *= scale
        self.K_scaled[0, 2] *= scale
        self.K_scaled[1, 2] *= scale
    
    def reconstruct(self, images: List[dict],
                   poses: Dict[int, CameraPose],
                   max_pairs: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dense point cloud with multi-view consistency
        """
        print("\n" + "=" * 60)
        print(f"GPU DENSE STEREO")
        print(f"  Scale: {self.scale}x, Depths: {self.num_depths}, Min views: {self.min_views}")
        print("=" * 60)
        
        t0 = time.time()
        
        camera_indices = sorted(poses.keys())
        n_cameras = len(camera_indices)
        
        if n_cameras < 3:
            print("Need at least 3 cameras for multi-view stereo")
            return np.array([]), np.array([])
        
        # Preprocess images
        print("\nPreparing images...")
        processed = self._prepare_images(images, camera_indices)
        
        # Compute depth range from sparse reconstruction
        centers = np.array([poses[idx].center for idx in camera_indices])
        scene_center = np.median(centers, axis=0)
        scene_radius = np.percentile(np.linalg.norm(centers - scene_center, axis=1), 90)
        
        depth_min = max(0.1, scene_radius * 0.1)
        depth_max = scene_radius * 5.0
        
        print(f"  Depth range: {depth_min:.2f} - {depth_max:.2f}")
        
        # Process reference views
        all_points = []
        all_colors = []
        
        # Select subset of reference views for speed
        step = max(1, n_cameras // max_pairs)
        ref_indices = camera_indices[::step]
        
        print(f"\nProcessing {len(ref_indices)} reference views...")
        
        for i, ref_idx in enumerate(ref_indices):
            t1 = time.time()
            
            # Find neighbor cameras
            neighbors = self._find_neighbors(ref_idx, camera_indices, poses, k=6)
            
            if len(neighbors) < 2:
                continue
            
            # Compute depth map with multi-view consistency
            depth_map, confidence, color_map = self._compute_depth_map_gpu(
                ref_idx, neighbors, processed, poses, depth_min, depth_max
            )
            
            # Back-project to 3D
            points, colors = self._backproject(
                depth_map, confidence, color_map, 
                poses[ref_idx], min_confidence=self.min_views - 0.5
            )
            
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
            
            elapsed = time.time() - t1
            print(f"  [{i+1}/{len(ref_indices)}] Cam {ref_idx}: {len(points):,} pts ({elapsed:.1f}s)")
        
        if not all_points:
            print("No points reconstructed!")
            return np.array([]), np.array([])
        
        # Merge and filter
        print("\nMerging point clouds...")
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        print(f"  Raw points: {len(points):,}")
        
        # Statistical outlier removal
        points, colors = self._filter_outliers(points, colors)
        print(f"  After outlier removal: {len(points):,}")
        
        # Voxel grid downsampling
        points, colors = self._voxel_down_sample(points, colors, voxel_size=0.02)
        print(f"  After voxel downsample: {len(points):,}")
        
        total_time = time.time() - t0
        print(f"\nDense stereo completed in {total_time:.1f}s")
        
        return points, colors
    
    def _prepare_images(self, images: List[dict], indices: List[int]) -> Dict:
        """Prepare scaled grayscale and color images"""
        processed = {}
        
        for idx in indices:
            img = images[idx]['image']
            
            # Scale down
            h, w = img.shape[:2]
            new_h, new_w = int(h * self.scale), int(w * self.scale)
            img_scaled = cv.resize(img, (new_w, new_h))
            
            gray = cv.cvtColor(img_scaled, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            processed[idx] = {
                'gray': gray,
                'color': img_scaled,
                'shape': (new_h, new_w)
            }
        
        return processed
    
    def _find_neighbors(self, ref_idx: int, all_indices: List[int], 
                       poses: Dict[int, CameraPose], k: int = 6) -> List[int]:
        """Find k nearest cameras by position"""
        ref_center = poses[ref_idx].center
        
        distances = []
        for idx in all_indices:
            if idx == ref_idx:
                continue
            dist = np.linalg.norm(poses[idx].center - ref_center)
            distances.append((idx, dist))
        
        distances.sort(key=lambda x: x[1])
        return [idx for idx, _ in distances[:k]]  # <-- FIXED: was missing closing bracket
    
    def _compute_depth_map_gpu(self, ref_idx: int, neighbor_indices: List[int],
                               processed: Dict, poses: Dict[int, CameraPose],
                               depth_min: float, depth_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute depth map using plane sweep with GPU acceleration
        Returns: depth_map, confidence_map (number of consistent views), color_map
        """
        ref_data = processed[ref_idx]
        H, W = ref_data['shape']
        
        # Create depth hypotheses (inverse depth for better sampling)
        inv_depths = np.linspace(1/depth_max, 1/depth_min, self.num_depths)
        depths = 1.0 / inv_depths
        
        # Reference image
        ref_gray = ref_data['gray']
        ref_pose = poses[ref_idx]
        
        if HAS_TORCH:
            return self._plane_sweep_torch(
                ref_gray, ref_data['color'], ref_pose, 
                neighbor_indices, processed, poses, depths, H, W
            )
        else:
            return self._plane_sweep_numpy(
                ref_gray, ref_data['color'], ref_pose,
                neighbor_indices, processed, poses, depths, H, W
            )
    
    def _plane_sweep_torch(self, ref_gray: np.ndarray, ref_color: np.ndarray,
                          ref_pose: CameraPose, neighbor_indices: List[int],
                          processed: Dict, poses: Dict[int, CameraPose],
                          depths: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GPU-accelerated plane sweep using PyTorch"""
        
        # Convert to torch tensors
        ref_tensor = torch.from_numpy(ref_gray).float().to(self.device)
        depths_tensor = torch.from_numpy(depths).float().to(self.device)
        
        K = torch.from_numpy(self.K_scaled).float().to(self.device)
        K_inv = torch.inverse(K)
        
        # Reference camera matrices
        R_ref = torch.from_numpy(ref_pose.R).float().to(self.device)
        t_ref = torch.from_numpy(ref_pose.t).float().to(self.device)
        
        # Pixel coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        ones = torch.ones_like(x_coords)
        pixels = torch.stack([x_coords, y_coords, ones], dim=-1).float()
        
        # Prepare neighbor data
        neighbor_grays = []
        neighbor_transforms = []
        
        for idx in neighbor_indices:
            gray = torch.from_numpy(processed[idx]['gray']).float().to(self.device)
            neighbor_grays.append(gray)
            
            pose = poses[idx]
            R = torch.from_numpy(pose.R).float().to(self.device)
            t = torch.from_numpy(pose.t).float().to(self.device)
            neighbor_transforms.append((R, t))
        
        # Cost volume: (num_depths, H, W) - count of consistent views
        consistency_volume = torch.zeros(len(depths), H, W, device=self.device)
        
        # For each depth plane
        for d_idx, depth in enumerate(depths_tensor):
            # Back-project to 3D: X = depth * K^-1 * pixel
            rays = torch.matmul(pixels, K_inv.T)
            points_ref = rays * depth
            
            # Transform to world: X_world = R_ref.T @ (X_ref - t_ref)
            points_world = torch.matmul(points_ref - t_ref, R_ref)
            
            # Check consistency with each neighbor
            for n_idx, (R_n, t_n) in enumerate(neighbor_transforms):
                # Project to neighbor: x = K @ (R @ X_world + t)
                points_n = torch.matmul(points_world, R_n.T) + t_n
                
                # Perspective divide
                z = points_n[..., 2:3]
                valid_z = z > 0.1
                
                proj = points_n[..., :2] / (z + 1e-8)
                proj = torch.matmul(proj, K[:2, :2].T) + K[:2, 2]
                
                # Sample neighbor image
                # Normalize to [-1, 1] for grid_sample
                proj_norm = proj.clone()
                proj_norm[..., 0] = 2.0 * proj[..., 0] / (W - 1) - 1.0
                proj_norm[..., 1] = 2.0 * proj[..., 1] / (H - 1) - 1.0
                
                # Grid sample
                neighbor_gray = neighbor_grays[n_idx].unsqueeze(0).unsqueeze(0)
                grid = proj_norm.unsqueeze(0)
                
                sampled = F.grid_sample(neighbor_gray, grid, mode='bilinear', 
                                        padding_mode='zeros', align_corners=True)
                sampled = sampled.squeeze()
                
                # NCC in local window
                ncc = self._compute_ncc_torch(ref_tensor, sampled, self.patch_size)
                
                # Count consistent views
                consistent = (ncc > self.consistency_thresh) & valid_z.squeeze()
                consistency_volume[d_idx] += consistent.float()
        
        # Find best depth (maximum consistency)
        best_consistency, best_depth_idx = torch.max(consistency_volume, dim=0)
        
        # Get depth values
        depth_map = depths_tensor[best_depth_idx]
        
        # Convert to numpy
        depth_map = depth_map.cpu().numpy()
        confidence = best_consistency.cpu().numpy()
        
        return depth_map, confidence, ref_color
    
    def _compute_ncc_torch(self, img1: torch.Tensor, img2: torch.Tensor, 
                          patch_size: int) -> torch.Tensor:
        """Compute Normalized Cross-Correlation using convolutions"""
        kernel_size = patch_size
        padding = patch_size // 2
        
        # Create averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device)
        kernel = kernel / (kernel_size * kernel_size)
        
        # Add batch/channel dims
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
        
        # Local means
        mean1 = F.conv2d(img1, kernel, padding=padding)
        mean2 = F.conv2d(img2, kernel, padding=padding)
        
        # Local variances
        var1 = F.conv2d(img1 * img1, kernel, padding=padding) - mean1 * mean1
        var2 = F.conv2d(img2 * img2, kernel, padding=padding) - mean2 * mean2
        
        # Covariance
        cov = F.conv2d(img1 * img2, kernel, padding=padding) - mean1 * mean2
        
        # NCC
        denom = torch.sqrt(var1 * var2 + 1e-8)
        ncc = cov / denom
        
        return ncc.squeeze()
    
    def _plane_sweep_numpy(self, ref_gray: np.ndarray, ref_color: np.ndarray,
                          ref_pose: CameraPose, neighbor_indices: List[int],
                          processed: Dict, poses: Dict[int, CameraPose],
                          depths: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU fallback for plane sweep"""
        
        K = self.K_scaled
        K_inv = np.linalg.inv(K)
        
        # Pixel coordinates
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixels = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1).astype(np.float32)
        
        # Cost volume
        consistency_volume = np.zeros((len(depths), H, W), dtype=np.float32)
        
        for d_idx, depth in enumerate(depths):
            # Back-project
            rays = np.einsum('ijk,lk->ijl', pixels, K_inv)
            points_ref = rays * depth
            
            # To world
            points_world = (points_ref - ref_pose.t) @ ref_pose.R
            
            for n_idx in neighbor_indices:
                pose_n = poses[n_idx]
                gray_n = processed[n_idx]['gray']
                
                # Project to neighbor
                points_n = points_world @ pose_n.R.T + pose_n.t
                
                z = points_n[..., 2]
                valid = z > 0.1
                
                proj = points_n[..., :2] / (z[..., np.newaxis] + 1e-8)
                proj = proj @ K[:2, :2].T + K[:2, 2]
                
                # Sample
                px = proj[..., 0].astype(np.float32)
                py = proj[..., 1].astype(np.float32)
                
                sampled = cv.remap(gray_n, px, py, cv.INTER_LINEAR, borderValue=0)
                
                # Simple SSD instead of NCC for speed
                diff = (ref_gray - sampled) ** 2
                kernel = np.ones((self.patch_size, self.patch_size)) / (self.patch_size ** 2)
                ssd = cv.filter2D(diff, -1, kernel)
                
                consistent = (ssd < 0.02) & valid
                consistency_volume[d_idx] += consistent.astype(np.float32)
        
        best_depth_idx = np.argmax(consistency_volume, axis=0)
        best_consistency = np.max(consistency_volume, axis=0)
        
        depth_map = depths[best_depth_idx]
        
        return depth_map, best_consistency, ref_color
    
    def _backproject(self, depth_map: np.ndarray, confidence: np.ndarray,
                    color_map: np.ndarray, pose: CameraPose,
                    min_confidence: float) -> Tuple[np.ndarray, np.ndarray]:
        """Back-project depth map to 3D points"""
        H, W = depth_map.shape
        
        # Valid mask
        valid = (confidence >= min_confidence) & (depth_map > 0)
        
        if not np.any(valid):
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        # Pixel coordinates
        y_coords, x_coords = np.where(valid)
        depths = depth_map[valid]
        colors = color_map[y_coords, x_coords]
        
        # Back-project
        K_inv = np.linalg.inv(self.K_scaled)
        
        pixels = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1).astype(np.float32)
        rays = pixels @ K_inv.T
        points_cam = rays * depths[:, np.newaxis]
        
        # To world coordinates
        points_world = (points_cam - pose.t) @ pose.R
        
        # RGB order
        colors_rgb = colors[:, ::-1]  # BGR to RGB
        
        return points_world, colors_rgb
    
    def _filter_outliers(self, points: np.ndarray, colors: np.ndarray,
                        k: int = 20, std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Statistical outlier removal"""
        if len(points) < k + 1:
            return points, colors
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Subsample for speed if too many points
            if len(points) > 500000:
                indices = np.random.choice(len(points), 500000, replace=False)
                points_sample = points[indices]
            else:
                points_sample = points
                indices = np.arange(len(points))
            
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(points_sample)
            distances, _ = nn.kneighbors(points_sample)
            
            mean_dist = np.mean(distances[:, 1:], axis=1)
            threshold = np.mean(mean_dist) + std_ratio * np.std(mean_dist)
            
            inlier_mask = mean_dist < threshold
            
            return points[indices[inlier_mask]], colors[indices[inlier_mask]]
            
        except ImportError:
            # Fallback: simple distance-based filtering
            centroid = np.median(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            threshold = np.percentile(distances, 95)
            mask = distances < threshold
            return points[mask], colors[mask]
    
    def _voxel_down_sample(self, points: np.ndarray, colors: np.ndarray,
                         voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """Voxel grid downsampling"""
        if len(points) == 0:
            return points, colors
        
        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Create unique key for each voxel
        voxel_keys = (voxel_indices[:, 0].astype(np.int64) * 1000000000 + 
                     voxel_indices[:, 1].astype(np.int64) * 1000000 + 
                     voxel_indices[:, 2].astype(np.int64))
        
        # Get unique voxels
        _, unique_indices = np.unique(voxel_keys, return_index=True)
        
        return points[unique_indices], colors[unique_indices]


def create_combined_dense_cloud(camera: Camera, images: List[dict], 
                                poses: Dict[int, CameraPose],
                                use_stereo: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function to create dense cloud using stereo method
    """
    if use_stereo:
        recon = DenseStereoReconstructor(camera)
        return recon.reconstruct(images, poses)
    else:
        return np.array([]), np.array([])
