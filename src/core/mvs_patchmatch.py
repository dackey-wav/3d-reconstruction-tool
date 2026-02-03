"""
PatchMatch Multi-View Stereo

Based on "PatchMatch Stereo - Stereo Matching with Slanted Support Windows"
and "Massively Parallel Multiview Stereopsis by Surface Normal Diffusion"

Key ideas:
1. Random initialization of depth & normal hypotheses
2. Spatial propagation (neighbors likely have similar depth)
3. View propagation (project hypothesis to other views)
4. Refinement with random perturbations
5. Multi-view consistency filtering
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


@dataclass
class DepthNormalMap:
    """Depth and normal map for a view"""
    depth: np.ndarray      # (H, W)
    normal: np.ndarray     # (H, W, 3)
    confidence: np.ndarray # (H, W)


class PatchMatchMVS:
    """
    GPU-accelerated PatchMatch Multi-View Stereo
    """
    
    def __init__(self, camera: Camera,
                 scale: float = 0.25,
                 patch_size: int = 11,
                 num_iterations: int = 3,
                 num_samples: int = 8,
                 min_views: int = 3,
                 depth_min: float = 0.1,
                 depth_max: float = 100.0):
        
        self.camera = camera
        self.scale = scale
        self.patch_size = patch_size
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.min_views = min_views
        self.depth_min = depth_min
        self.depth_max = depth_max
        
        if HAS_TORCH and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"PatchMatch MVS using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("PatchMatch MVS using CPU")
        
        # Scaled intrinsics
        self.K_scaled = camera.K.copy()
        self.K_scaled[:2] *= scale
    
    def reconstruct(self, images: List[dict], 
                   poses: Dict[int, CameraPose],
                   sparse_points: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full MVS reconstruction pipeline
        """
        print("\n" + "=" * 60)
        print("PATCHMATCH MULTI-VIEW STEREO")
        print(f"  Scale: {self.scale}x, Patch: {self.patch_size}, Iters: {self.num_iterations}")
        print("=" * 60)
        
        t0 = time.time()
        
        cam_indices = sorted(poses.keys())
        n_cams = len(cam_indices)
        
        if n_cams < 3:
            print("Need at least 3 cameras")
            return np.array([]), np.array([])
        
        # Estimate depth range from camera positions or sparse points
        self._estimate_depth_range(poses, sparse_points)
        print(f"  Depth range: [{self.depth_min:.2f}, {self.depth_max:.2f}]")
        
        # Prepare images
        print("\nPreparing images...")
        proc_images = self._prepare_images(images, cam_indices)
        
        # Compute depth maps for each view
        print(f"\nComputing depth maps for {n_cams} views...")
        depth_maps = {}
        
        for i, ref_idx in enumerate(cam_indices):
            t1 = time.time()
            
            # Find source views (neighbors)
            src_indices = self._select_source_views(ref_idx, cam_indices, poses, k=4)
            
            if len(src_indices) < 2:
                print(f"  [{i+1}/{n_cams}] Cam {ref_idx}: skipped (not enough neighbors)")
                continue
            
            # Run PatchMatch
            depth_normal = self._patchmatch_cuda(
                ref_idx, src_indices, proc_images, poses
            )
            
            depth_maps[ref_idx] = depth_normal
            
            valid_pixels = np.sum(depth_normal.confidence >= self.min_views)
            elapsed = time.time() - t1
            print(f"  [{i+1}/{n_cams}] Cam {ref_idx}: {valid_pixels:,} valid pixels ({elapsed:.1f}s)")
        
        # Depth map fusion
        print("\nFusing depth maps...")
        points, colors = self._fuse_depth_maps(depth_maps, proc_images, poses)
        
        print(f"  Raw points: {len(points):,}")
        
        # Filter and downsample
        if len(points) > 0:
            points, colors = self._filter_points(points, colors)
            print(f"  After filtering: {len(points):,}")
        
        total_time = time.time() - t0
        print(f"\nPatchMatch MVS completed in {total_time:.1f}s")
        
        return points, colors
    
    def _estimate_depth_range(self, poses: Dict[int, CameraPose], 
                              sparse_points: np.ndarray = None):
        """Estimate scene depth range"""
        centers = np.array([poses[i].center for i in poses])
        
        if sparse_points is not None and len(sparse_points) > 0:
            # Use sparse points to estimate depth
            all_depths = []
            for idx in poses:
                pose = poses[idx]
                pts_cam = pose.transform_points(sparse_points)
                depths = pts_cam[:, 2]
                valid = depths > 0
                if np.any(valid):
                    all_depths.extend(depths[valid])
            
            if all_depths:
                self.depth_min = max(0.1, np.percentile(all_depths, 1))
                self.depth_max = np.percentile(all_depths, 99) * 1.5
                return
        
        # Fallback: estimate from camera positions
        scene_scale = np.percentile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 90)
        self.depth_min = max(0.1, scene_scale * 0.05)
        self.depth_max = scene_scale * 10.0
    
    def _prepare_images(self, images: List[dict], indices: List[int]) -> Dict:
        """Prepare scaled images"""
        processed = {}
        
        for idx in indices:
            img = images[idx]['image']
            h, w = img.shape[:2]
            new_h, new_w = int(h * self.scale), int(w * self.scale)
            
            img_scaled = cv.resize(img, (new_w, new_h))
            gray = cv.cvtColor(img_scaled, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # Compute gradients for edge-aware processing
            grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
            grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
            
            processed[idx] = {
                'color': img_scaled,
                'gray': gray,
                'grad_x': grad_x,
                'grad_y': grad_y,
                'shape': (new_h, new_w)
            }
        
        return processed
    
    def _select_source_views(self, ref_idx: int, all_indices: List[int],
                            poses: Dict[int, CameraPose], k: int = 4) -> List[int]:
        """Select best source views based on baseline and angle"""
        ref_center = poses[ref_idx].center
        ref_dir = poses[ref_idx].R[2, :]  # View direction
        
        scores = []
        for idx in all_indices:
            if idx == ref_idx:
                continue
            
            src_center = poses[idx].center
            src_dir = poses[idx].R[2, :]
            
            # Baseline
            baseline = np.linalg.norm(src_center - ref_center)
            
            # Angle between views (prefer ~15-30 degrees)
            cos_angle = np.dot(ref_dir, src_dir)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            # Score: prefer moderate baseline and angle
            if 5 < angle < 60:
                score = baseline * (1 - abs(angle - 20) / 60)
            else:
                score = 0
            
            scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:k]]
    
    def _patchmatch_cuda(self, ref_idx: int, src_indices: List[int],
                        images: Dict, poses: Dict[int, CameraPose]) -> DepthNormalMap:
        """
        PatchMatch stereo on GPU
        """
        ref_data = images[ref_idx]
        H, W = ref_data['shape']
        ref_pose = poses[ref_idx]
        
        # Convert to tensors
        ref_gray = torch.from_numpy(ref_data['gray']).float().to(self.device)
        
        K = torch.from_numpy(self.K_scaled.astype(np.float32)).to(self.device)
        K_inv = torch.inverse(K)
        
        # Reference camera
        R_ref = torch.from_numpy(ref_pose.R.astype(np.float32)).to(self.device)
        t_ref = torch.from_numpy(ref_pose.t.astype(np.float32)).to(self.device)
        
        # Source cameras
        src_grays = []
        src_Rs = []
        src_ts = []
        for idx in src_indices:
            src_grays.append(
                torch.from_numpy(images[idx]['gray']).float().to(self.device)
            )
            src_Rs.append(
                torch.from_numpy(poses[idx].R.astype(np.float32)).to(self.device)
            )
            src_ts.append(
                torch.from_numpy(poses[idx].t.astype(np.float32)).to(self.device)
            )
        
        # Pixel grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # === INITIALIZATION ===
        # Random depth initialization (log-uniform)
        log_depth_min = np.log(self.depth_min)
        log_depth_max = np.log(self.depth_max)
        depth = torch.exp(
            torch.rand(H, W, device=self.device) * (log_depth_max - log_depth_min) + log_depth_min
        )
        
        # Random normal initialization (facing camera)
        normal = torch.zeros(H, W, 3, device=self.device)
        normal[:, :, 2] = -1  # Pointing towards camera
        
        # Add random perturbation to normals
        normal[:, :, 0] = torch.randn(H, W, device=self.device) * 0.3
        normal[:, :, 1] = torch.randn(H, W, device=self.device) * 0.3
        normal = F.normalize(normal, dim=-1)
        
        # Best cost so far
        best_cost = torch.full((H, W), float('inf'), device=self.device)
        
        # === PATCHMATCH ITERATIONS ===
        for iteration in range(self.num_iterations):
            # Propagation directions alternate
            if iteration % 2 == 0:
                row_range = range(H)
                col_range = range(W)
            else:
                row_range = range(H - 1, -1, -1)
                col_range = range(W - 1, -1, -1)
            
            # Spatial propagation (vectorized)
            depth, normal, best_cost = self._spatial_propagation(
                depth, normal, best_cost, ref_gray, src_grays,
                K, K_inv, R_ref, t_ref, src_Rs, src_ts,
                x_grid, y_grid, forward=(iteration % 2 == 0)
            )
            
            # Random refinement
            depth, normal, best_cost = self._random_refinement(
                depth, normal, best_cost, ref_gray, src_grays,
                K, K_inv, R_ref, t_ref, src_Rs, src_ts,
                x_grid, y_grid, iteration
            )
        
        # Compute final confidence (number of consistent views)
        confidence = self._compute_confidence(
            depth, normal, ref_gray, src_grays,
            K, K_inv, R_ref, t_ref, src_Rs, src_ts,
            x_grid, y_grid
        )
        
        return DepthNormalMap(
            depth=depth.cpu().numpy(),
            normal=normal.cpu().numpy(),
            confidence=confidence.cpu().numpy()
        )
    
    def _compute_patch_cost(self, ref_gray: torch.Tensor,
                           depth: torch.Tensor, normal: torch.Tensor,
                           src_grays: List[torch.Tensor],
                           K: torch.Tensor, K_inv: torch.Tensor,
                           R_ref: torch.Tensor, t_ref: torch.Tensor,
                           src_Rs: List[torch.Tensor], src_ts: List[torch.Tensor],
                           x_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute matching cost using bilateral weighted NCC
        """
        H, W = ref_gray.shape
        half_patch = self.patch_size // 2
        
        # Aggregate cost over all source views
        total_cost = torch.zeros(H, W, device=self.device)
        valid_count = torch.zeros(H, W, device=self.device)
        
        # Back-project reference pixels to 3D
        ones = torch.ones(H, W, device=self.device)
        pixels_ref = torch.stack([x_grid, y_grid, ones], dim=-1)  # (H, W, 3)
        rays = torch.matmul(pixels_ref, K_inv.T)
        points_3d = rays * depth.unsqueeze(-1)  # (H, W, 3)
        
        # Transform to world coordinates
        points_world = torch.matmul(points_3d - t_ref, R_ref)  # (H, W, 3)
        
        for src_idx, (src_gray, R_src, t_src) in enumerate(zip(src_grays, src_Rs, src_ts)):
            # Project to source view
            points_src = torch.matmul(points_world, R_src.T) + t_src
            z_src = points_src[:, :, 2]
            
            # Check depth validity
            valid_depth = z_src > 0.1
            
            # Perspective projection
            proj = points_src[:, :, :2] / (z_src.unsqueeze(-1) + 1e-8)
            proj = torch.matmul(proj, K[:2, :2].T) + K[:2, 2]
            
            # Check if projection is in bounds
            valid_x = (proj[:, :, 0] >= half_patch) & (proj[:, :, 0] < W - half_patch)
            valid_y = (proj[:, :, 1] >= half_patch) & (proj[:, :, 1] < H - half_patch)
            valid = valid_depth & valid_x & valid_y
            
            # Sample source image
            proj_norm = proj.clone()
            proj_norm[:, :, 0] = 2.0 * proj[:, :, 0] / (W - 1) - 1.0
            proj_norm[:, :, 1] = 2.0 * proj[:, :, 1] / (H - 1) - 1.0
            
            src_sampled = F.grid_sample(
                src_gray.unsqueeze(0).unsqueeze(0),
                proj_norm.unsqueeze(0),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze()
            
            # Compute NCC cost
            cost = self._ncc_cost(ref_gray, src_sampled)
            
            # Accumulate (only for valid projections)
            total_cost += torch.where(valid, cost, torch.zeros_like(cost))
            valid_count += valid.float()
        
        # Average cost
        avg_cost = total_cost / (valid_count + 1e-8)
        avg_cost = torch.where(valid_count >= 2, avg_cost, torch.full_like(avg_cost, float('inf')))
        
        return avg_cost
    
    def _ncc_cost(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute NCC cost (1 - NCC, so lower is better)"""
        kernel_size = self.patch_size
        padding = kernel_size // 2
        
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device)
        kernel = kernel / (kernel_size * kernel_size)
        
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
        
        mean1 = F.conv2d(img1, kernel, padding=padding)
        mean2 = F.conv2d(img2, kernel, padding=padding)
        
        var1 = F.conv2d(img1 * img1, kernel, padding=padding) - mean1 * mean1
        var2 = F.conv2d(img2 * img2, kernel, padding=padding) - mean2 * mean2
        cov = F.conv2d(img1 * img2, kernel, padding=padding) - mean1 * mean2
        
        ncc = cov / (torch.sqrt(var1 * var2) + 1e-8)
        cost = 1.0 - ncc.squeeze()
        
        return cost
    
    def _spatial_propagation(self, depth, normal, best_cost, ref_gray, src_grays,
                            K, K_inv, R_ref, t_ref, src_Rs, src_ts,
                            x_grid, y_grid, forward=True):
        """Propagate hypotheses from neighbors"""
        H, W = depth.shape
        
        # Neighbor offsets
        if forward:
            offsets = [(-1, 0), (0, -1)]  # Top and left
        else:
            offsets = [(1, 0), (0, 1)]    # Bottom and right
        
        for dy, dx in offsets:
            # Shift depth and normal maps
            if dy != 0:
                if dy > 0:
                    depth_neighbor = F.pad(depth[:-dy, :], (0, 0, dy, 0), value=self.depth_min)
                    normal_neighbor = F.pad(normal[:-dy, :, :], (0, 0, 0, 0, dy, 0))
                else:
                    depth_neighbor = F.pad(depth[-dy:, :], (0, 0, 0, -dy), value=self.depth_min)
                    normal_neighbor = F.pad(normal[-dy:, :, :], (0, 0, 0, 0, 0, -dy))
            else:
                if dx > 0:
                    depth_neighbor = F.pad(depth[:, :-dx], (dx, 0), value=self.depth_min)
                    normal_neighbor = F.pad(normal[:, :-dx, :], (0, 0, dx, 0))
                else:
                    depth_neighbor = F.pad(depth[:, -dx:], (0, -dx), value=self.depth_min)
                    normal_neighbor = F.pad(normal[:, -dx:, :], (0, 0, 0, -dx))
            
            # Compute cost for neighbor hypothesis
            neighbor_cost = self._compute_patch_cost(
                ref_gray, depth_neighbor, normal_neighbor,
                src_grays, K, K_inv, R_ref, t_ref, src_Rs, src_ts,
                x_grid, y_grid
            )
            
            # Update where neighbor is better
            better = neighbor_cost < best_cost
            depth = torch.where(better, depth_neighbor, depth)
            normal = torch.where(better.unsqueeze(-1), normal_neighbor, normal)
            best_cost = torch.where(better, neighbor_cost, best_cost)
        
        return depth, normal, best_cost
    
    def _random_refinement(self, depth, normal, best_cost, ref_gray, src_grays,
                          K, K_inv, R_ref, t_ref, src_Rs, src_ts,
                          x_grid, y_grid, iteration):
        """Refine hypotheses with random perturbations"""
        H, W = depth.shape
        
        # Decreasing search range
        depth_range = (self.depth_max - self.depth_min) * (0.5 ** iteration)
        normal_range = 0.5 * (0.5 ** iteration)
        
        for _ in range(self.num_samples):
            # Random depth perturbation
            depth_delta = (torch.rand(H, W, device=self.device) * 2 - 1) * depth_range
            depth_new = torch.clamp(depth + depth_delta, self.depth_min, self.depth_max)
            
            # Random normal perturbation
            normal_delta = torch.randn(H, W, 3, device=self.device) * normal_range
            normal_new = F.normalize(normal + normal_delta, dim=-1)
            
            # Compute cost
            new_cost = self._compute_patch_cost(
                ref_gray, depth_new, normal_new,
                src_grays, K, K_inv, R_ref, t_ref, src_Rs, src_ts,
                x_grid, y_grid
            )
            
            # Update where new is better
            better = new_cost < best_cost
            depth = torch.where(better, depth_new, depth)
            normal = torch.where(better.unsqueeze(-1), normal_new, normal)
            best_cost = torch.where(better, new_cost, best_cost)
        
        return depth, normal, best_cost
    
    def _compute_confidence(self, depth, normal, ref_gray, src_grays,
                           K, K_inv, R_ref, t_ref, src_Rs, src_ts,
                           x_grid, y_grid) -> torch.Tensor:
        """Count number of views with consistent depth"""
        H, W = depth.shape
        confidence = torch.zeros(H, W, device=self.device)
        
        ones = torch.ones(H, W, device=self.device)
        pixels_ref = torch.stack([x_grid, y_grid, ones], dim=-1)
        rays = torch.matmul(pixels_ref, K_inv.T)
        points_3d = rays * depth.unsqueeze(-1)
        points_world = torch.matmul(points_3d - t_ref, R_ref)
        
        ncc_threshold = 0.6
        
        for src_gray, R_src, t_src in zip(src_grays, src_Rs, src_ts):
            points_src = torch.matmul(points_world, R_src.T) + t_src
            z_src = points_src[:, :, 2]
            valid_depth = z_src > 0.1
            
            proj = points_src[:, :, :2] / (z_src.unsqueeze(-1) + 1e-8)
            proj = torch.matmul(proj, K[:2, :2].T) + K[:2, 2]
            
            valid_x = (proj[:, :, 0] >= 0) & (proj[:, :, 0] < W)
            valid_y = (proj[:, :, 1] >= 0) & (proj[:, :, 1] < H)
            valid = valid_depth & valid_x & valid_y
            
            proj_norm = proj.clone()
            proj_norm[:, :, 0] = 2.0 * proj[:, :, 0] / (W - 1) - 1.0
            proj_norm[:, :, 1] = 2.0 * proj[:, :, 1] / (H - 1) - 1.0
            
            src_sampled = F.grid_sample(
                src_gray.unsqueeze(0).unsqueeze(0),
                proj_norm.unsqueeze(0),
                mode='bilinear', padding_mode='zeros', align_corners=True
            ).squeeze()
            
            ncc = 1.0 - self._ncc_cost(ref_gray, src_sampled)
            consistent = valid & (ncc > ncc_threshold)
            confidence += consistent.float()
        
        return confidence
    
    def _fuse_depth_maps(self, depth_maps: Dict[int, DepthNormalMap],
                        images: Dict, poses: Dict[int, CameraPose]) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse depth maps into a single point cloud"""
        all_points = []
        all_colors = []
        
        K_inv = np.linalg.inv(self.K_scaled)
        
        for idx, dm in depth_maps.items():
            valid = dm.confidence >= self.min_views
            
            if not np.any(valid):
                continue
            
            H, W = dm.depth.shape
            y_coords, x_coords = np.where(valid)
            depths = dm.depth[valid]
            colors = images[idx]['color'][y_coords, x_coords]
            
            # Back-project
            pixels = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
            rays = pixels @ K_inv.T
            points_cam = rays * depths[:, np.newaxis]
            
            # To world
            pose = poses[idx]
            points_world = (points_cam - pose.t) @ pose.R
            
            all_points.append(points_world)
            all_colors.append(colors[:, ::-1])  # BGR to RGB
        
        if not all_points:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        return np.vstack(all_points), np.vstack(all_colors)
    
    def _filter_points(self, points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter outliers and downsample"""
        # Statistical outlier removal
        centroid = np.median(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        threshold = np.percentile(distances, 95)
        mask = distances < threshold
        points, colors = points[mask], colors[mask]
        
        # Voxel downsample
        voxel_size = 0.01
        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        voxel_keys = (voxel_indices[:, 0] * 1000000000 + 
                     voxel_indices[:, 1] * 1000000 + 
                     voxel_indices[:, 2])
        _, unique_idx = np.unique(voxel_keys, return_index=True)
        
        return points[unique_idx], colors[unique_idx]