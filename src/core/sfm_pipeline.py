"""
Structure from Motion Pipeline

Incremental SfM:
1. Initialize from best image pair
2. Incrementally add images using PnP
3. Triangulate new points
4. Bundle adjustment for refinement
"""
import numpy as np
import cv2 as cv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import glob
import time

from .camera import Camera, CameraPose, load_calibration
from .features import FeatureExtractor, FeatureMatcher, ImageFeatures, FeatureMatch
from .geometry import (
    triangulate_points, 
    compute_essential_matrix, 
    decompose_essential,
    compute_reprojection_error
)


class SfMPipeline:
    """
    Incremental Structure from Motion pipeline
    """
    
    def __init__(self, calibration_path: str, fast_mode: bool = False, neural_mode: bool = False):
        self.camera = load_calibration(calibration_path)
        self.fast_mode = fast_mode
        self.neural_mode = neural_mode
        
        # Neural matcher (LightGlue + SuperPoint)
        if neural_mode:
            try:
                from .neural_matcher import NeuralMatcher
                self.neural_matcher = NeuralMatcher()
                print("Using LightGlue neural matcher")
                self.extractor = None
                self.matcher = None
            except ImportError as e:
                print(f"WARNING: Could not load neural matcher: {e}")
                print("Falling back to SIFT")
                self.neural_mode = False
                neural_mode = False
        
        # Classical SIFT matcher
        if not neural_mode:
            if fast_mode:
                self.extractor = FeatureExtractor(n_features=3000)
                self.matcher = FeatureMatcher(ratio_threshold=0.8)
            else:
                self.extractor = FeatureExtractor(n_features=8000)
                self.matcher = FeatureMatcher(ratio_threshold=0.75)
            self.neural_matcher = None
        
        self.image_scale = 0.5 if fast_mode else 1.0
        
        # Reconstruction state
        self.images: List[dict] = []
        self.features: List[ImageFeatures] = []
        self.neural_features: List = []
        self.poses: Dict[int, CameraPose] = {}
        self.points_3d: Dict[int, np.ndarray] = {}
        self.point_colors: Dict[int, np.ndarray] = {}
        self.observations: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.observation_index: Dict[Tuple[int, int], int] = {}
        
        self.match_cache: Dict[Tuple[int, int], List[FeatureMatch]] = {}
    
    def _get_keypoint_pt(self, img_idx: int, kp_idx: int) -> Tuple[float, float]:
        """
        Get keypoint coordinates - handles both neural and classical modes
        """
        if self.neural_mode:
            # Neural features store keypoints as np.ndarray directly
            return tuple(self.neural_features[img_idx].keypoints[kp_idx])
        else:
            # Classical SIFT stores cv.KeyPoint objects
            return self.features[img_idx].keypoints[kp_idx].pt
    
    def load_images(self, image_dir: str, max_images: int = None) -> List[dict]:
        """Load images from directory"""
        image_dir = Path(image_dir)
        
        extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(str(image_dir / ext)))
        
        image_paths = sorted(set(image_paths))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        images = []
        for path in image_paths:
            img = cv.imread(path)
            if img is None:
                print(f"  Warning: failed to load {path}")
                continue
            
            if self.image_scale < 1.0:
                h, w = img.shape[:2]
                new_w = int(w * self.image_scale)
                new_h = int(h * self.image_scale)
                img = cv.resize(img, (new_w, new_h))
            
            img = cv.undistort(img, self.camera.K * self.image_scale, self.camera.dist)
            
            images.append({
                'path': path,
                'image': img,
                'gray': cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            })
        
        mode_str = " (FAST MODE)" if self.fast_mode else ""
        print(f"Loaded {len(images)} images{mode_str}")
        return images
    
    def extract_features(self) -> List[ImageFeatures]:
        """Extract features from all images"""
        print("\nExtracting features...")
        features = []
        
        if self.neural_mode:
            from .neural_matcher import convert_neural_to_cv_keypoints
            self.neural_features = []
            for i, img_data in enumerate(self.images):
                neural_feat = self.neural_matcher.extract(img_data['image'])
                self.neural_features.append(neural_feat)
                
                # Create ImageFeatures wrapper for compatibility
                kp = convert_neural_to_cv_keypoints(neural_feat)
                desc = neural_feat.descriptors.T if neural_feat.descriptors is not None else None
                feat = ImageFeatures(keypoints=kp, descriptors=desc, image_shape=img_data['image'].shape[:2])
                features.append(feat)
                
                if (i + 1) % 20 == 0 or i == len(self.images) - 1:
                    print(f"  Processed {i+1}/{len(self.images)} images (SuperPoint)")
        else:
            for i, img_data in enumerate(self.images):
                feat = self.extractor.extract(img_data['image'])
                features.append(feat)
                if (i + 1) % 20 == 0 or i == len(self.images) - 1:
                    print(f"  Processed {i+1}/{len(self.images)} images (SIFT)")
        
        total_kp = sum(len(f) for f in features)
        print(f"  Total: {total_kp:,} keypoints")
        
        return features
    
    def match_image_pairs(self, window_size: int = 10) -> Dict[Tuple[int, int], List[FeatureMatch]]:
        """Match features between image pairs"""
        print(f"\nMatching features (window={window_size})...")
        n_images = len(self.features)
        matches = {}
        
        pairs_to_match = set()
        
        for i in range(n_images):
            for j in range(i + 1, min(i + window_size + 1, n_images)):
                pairs_to_match.add((i, j))
        
        loop_window = min(15, n_images // 3)
        for i in range(loop_window):
            for j in range(n_images - loop_window, n_images):
                if i < j:
                    pairs_to_match.add((i, j))
        
        for i in range(n_images):
            for offset in [5, 10, 15, 20, 25, 30]:
                j = i + offset
                if j < n_images:
                    pairs_to_match.add((i, j))
        
        pairs_to_match = sorted(pairs_to_match)
        print(f"  Phase 1: Matching {len(pairs_to_match)} pairs...")
        
        matched_count = 0
        min_matches_threshold = 15
        
        for idx, (i, j) in enumerate(pairs_to_match):
            if self.neural_mode:
                match_list, F = self.neural_matcher.match_pair_geometric(
                    self.neural_features[i], 
                    self.neural_features[j],
                    min_matches=min_matches_threshold
                )
                # Convert NeuralMatch to FeatureMatch for compatibility
                match_list = [FeatureMatch(idx1=m.idx1, idx2=m.idx2, distance=m.distance) 
                             for m in match_list]
            else:
                match_list, F = self.matcher.match_pair_geometric(
                    self.features[i], self.features[j], min_matches=min_matches_threshold
                )
            
            if len(match_list) >= min_matches_threshold:
                matches[(i, j)] = match_list
                matched_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"    [{idx+1}/{len(pairs_to_match)}] {matched_count} pairs matched")
        
        print(f"  Phase 1 result: {matched_count} valid pairs")
        
        components = self._get_components(matches, n_images)
        
        if len(components) > 1:
            print(f"\n  Phase 2: Bridging {len(components)} components...")
            bridge_pairs = self._find_bridge_pairs(components, n_images)
            
            bridge_matched = 0
            for i, j in bridge_pairs:
                if (i, j) in matches:
                    continue
                
                if self.neural_mode:
                    match_list, F = self.neural_matcher.match_pair_geometric(
                        self.neural_features[i],
                        self.neural_features[j],
                        min_matches=12
                    )
                    match_list = [FeatureMatch(idx1=m.idx1, idx2=m.idx2, distance=m.distance) 
                                 for m in match_list]
                else:
                    match_list, F = self.matcher.match_pair_geometric(
                        self.features[i], self.features[j], min_matches=12
                    )
                
                if len(match_list) >= 12:
                    matches[(i, j)] = match_list
                    bridge_matched += 1
            
            print(f"    Found {bridge_matched} bridge connections")
        
        print(f"  Total: {len(matches)} valid pairs")
        self._analyze_connectivity(matches, n_images)
        
        return matches
    
    def _get_components(self, matches: dict, n_images: int) -> List[List[int]]:
        """Get connected components from match graph"""
        adj = defaultdict(set)
        for (i, j) in matches.keys():
            adj[i].add(j)
            adj[j].add(i)
        
        visited = set()
        components = []
        
        for start in range(n_images):
            if start in visited or start not in adj:
                continue
            
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            if component:
                components.append(sorted(component))
        
        return components
    
    def _find_bridge_pairs(self, components: List[List[int]], n_images: int) -> List[Tuple[int, int]]:
        """Find pairs that could bridge different components"""
        bridge_pairs = []
        
        components = sorted(components, key=len, reverse=True)
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                samples1 = [comp1[0], comp1[-1], comp1[len(comp1)//2]] if len(comp1) > 2 else comp1
                samples2 = [comp2[0], comp2[-1], comp2[len(comp2)//2]] if len(comp2) > 2 else comp2
                
                for img1 in samples1:
                    for img2 in samples2:
                        if img1 < img2:
                            bridge_pairs.append((img1, img2))
                        else:
                            bridge_pairs.append((img2, img1))
        
        return list(set(bridge_pairs))
    
    def _analyze_connectivity(self, matches: dict, n_images: int):
        """Analyze and report graph connectivity"""
        components = self._get_components(matches, n_images)
        
        connected = set()
        for comp in components:
            connected.update(comp)
        
        print(f"\n  Connectivity analysis:")
        print(f"    Connected images: {len(connected)}/{n_images}")
        print(f"    Components: {len(components)}")
        
        if components:
            components_sorted = sorted(components, key=len, reverse=True)
            
            for idx, comp in enumerate(components_sorted[:5]):
                print(f"    Component {idx+1}: {len(comp)} images (#{min(comp)}-#{max(comp)})")
            
            if len(components_sorted) > 5:
                print(f"    ... and {len(components_sorted) - 5} more small components")
        
        adj = defaultdict(set)
        for (i, j) in matches.keys():
            adj[i].add(j)
            adj[j].add(i)
        
        isolated = [i for i in range(n_images) if i not in adj]
        if isolated:
            print(f"    Isolated images (no matches): {isolated}")
        
        if len(components) > 1:
            print(f"\n  ⚠ WARNING: Graph is fragmented into {len(components)} parts!")
    
    def find_best_initial_pair(self) -> Optional[dict]:
        """Find best pair for initialization"""
        print("\nFinding best initial pair...")
        
        components = self._get_components(self.match_cache, len(self.features))
        if not components:
            print("  No connected components found!")
            return None
        
        largest_component = max(components, key=len)
        print(f"  Searching in largest component ({len(largest_component)} images)...")
        
        candidates = []
        min_matches_for_init = 50
        
        for (i, j), matches in self.match_cache.items():
            if i not in largest_component or j not in largest_component:
                continue
                
            if len(matches) < min_matches_for_init:
                continue
            
            # Get points using helper method
            pts1 = np.array([self._get_keypoint_pt(i, m.idx1) for m in matches])
            pts2 = np.array([self._get_keypoint_pt(j, m.idx2) for m in matches])
            
            F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1.0, 0.999)
            if F is None:
                continue
            
            inlier_mask = mask.ravel() == 1
            pts1_in = pts1[inlier_mask]
            pts2_in = pts2[inlier_mask]
            matches_in = [m for m, inl in zip(matches, inlier_mask) if inl]
            
            if len(matches_in) < min_matches_for_init:
                continue
            
            E = compute_essential_matrix(self.camera, F)
            R, t, _ = decompose_essential(E, self.camera, pts1_in, pts2_in)
            
            pose1 = CameraPose.identity()
            pose2 = CameraPose(R=R, t=t.ravel())
            
            sample_idx = np.linspace(0, len(pts1_in)-1, min(50, len(pts1_in)), dtype=int)
            sample_pts1 = pts1_in[sample_idx]
            sample_pts2 = pts2_in[sample_idx]
            
            points_3d, valid = triangulate_points(
                self.camera, pose1, pose2, sample_pts1, sample_pts2
            )
            
            valid_count = np.sum(valid)
            if valid_count < 20:
                continue
            
            C1 = pose1.center
            C2 = pose2.center
            
            parallax_angles = []
            for pt in points_3d[valid]:
                ray1 = pt - C1
                ray2 = pt - C2
                cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2) + 1e-8)
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                parallax_angles.append(angle_deg)
            
            median_parallax = np.median(parallax_angles)
            
            if median_parallax < 1.5 or median_parallax > 40:
                continue
            
            score = len(matches_in) * (valid_count / len(sample_pts1))
            if 3 < median_parallax < 20:
                score *= 1.5
            
            candidates.append({
                'pair': (i, j),
                'matches': matches_in,
                'R': R,
                't': t,
                'parallax': median_parallax,
                'score': score,
                'pts1': pts1_in,
                'pts2': pts2_in,
                'valid_ratio': valid_count / len(sample_pts1)
            })
        
        if not candidates:
            print("  No valid initial pair found!")
            return None
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        print("  Top candidates:")
        for c in candidates[:3]:
            print(f"    ({c['pair'][0]}, {c['pair'][1]}): {len(c['matches'])} matches, "
                  f"parallax={c['parallax']:.1f}°, valid={c['valid_ratio']:.0%}")
        
        best = candidates[0]
        print(f"\n  Selected: ({best['pair'][0]}, {best['pair'][1]})")
        
        return best
    
    def initialize(self, init_data: dict) -> bool:
        """Initialize reconstruction from best pair"""
        i, j = init_data['pair']
        R, t = init_data['R'], init_data['t']
        matches = init_data['matches']
        pts1, pts2 = init_data['pts1'], init_data['pts2']
        
        self.poses[i] = CameraPose.identity()
        self.poses[j] = CameraPose(R=R, t=t.ravel())
        
        points_3d, valid_mask = triangulate_points(
            self.camera, self.poses[i], self.poses[j], pts1, pts2
        )
        
        point_id = 0
        for idx, match in enumerate(matches):
            if not valid_mask[idx]:
                continue
            
            self.points_3d[point_id] = points_3d[idx]
            self._add_observation(point_id, i, match.idx1)
            self._add_observation(point_id, j, match.idx2)
            
            # Get color using helper
            pt = self._get_keypoint_pt(i, match.idx1)
            x, y = int(pt[0]), int(pt[1])
            img = self.images[i]['image']
            h, w = img.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                self.point_colors[point_id] = img[y, x][::-1]
            else:
                self.point_colors[point_id] = np.array([127, 127, 127])
            
            point_id += 1
        
        print(f"  Initialized with {len(self.points_3d)} 3D points")
        return len(self.points_3d) > 0
    
    def _add_observation(self, point_id: int, img_idx: int, kp_idx: int):
        """Add observation linking 3D point to 2D keypoint"""
        self.observations[point_id].append((img_idx, kp_idx))
        self.observation_index[(img_idx, kp_idx)] = point_id
    
    def find_next_image(self, failed: Set[int]) -> Optional[int]:
        """Find image with most 2D-3D correspondences"""
        reconstructed = set(self.poses.keys())
        candidates = []
        
        for img_idx in range(len(self.features)):
            if img_idx in reconstructed or img_idx in failed:
                continue
            
            count = 0
            for other_idx in reconstructed:
                key = (min(img_idx, other_idx), max(img_idx, other_idx))
                if key not in self.match_cache:
                    continue
                
                matches = self.match_cache[key]
                for m in matches:
                    if key[0] == img_idx:
                        other_kp = m.idx2
                    else:
                        other_kp = m.idx1
                    
                    if (other_idx, other_kp) in self.observation_index:
                        count += 1
            
            if count >= 12:
                candidates.append((img_idx, count))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def register_image(self, img_idx: int) -> bool:
        """Register new image using PnP"""
        points_3d = []
        points_2d = []
        point_ids = []
        
        reconstructed = set(self.poses.keys())
        
        for other_idx in reconstructed:
            key = (min(img_idx, other_idx), max(img_idx, other_idx))
            if key not in self.match_cache:
                continue
            
            matches = self.match_cache[key]
            
            for m in matches:
                if key[0] == img_idx:
                    my_kp = m.idx1
                    other_kp = m.idx2
                else:
                    my_kp = m.idx2
                    other_kp = m.idx1
                
                if (other_idx, other_kp) not in self.observation_index:
                    continue
                
                point_id = self.observation_index[(other_idx, other_kp)]
                
                if point_id in point_ids:
                    continue
                
                # Use helper method for keypoint coordinates
                kp_pt = self._get_keypoint_pt(img_idx, my_kp)
                
                points_2d.append(kp_pt)
                points_3d.append(self.points_3d[point_id])
                point_ids.append(point_id)
        
        if len(points_3d) < 6:
            return False
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        methods = [
            (cv.SOLVEPNP_ITERATIVE, 8.0),
            (cv.SOLVEPNP_EPNP, 10.0),
            (cv.SOLVEPNP_P3P, 12.0),
        ]
        
        best_inliers = None
        best_rvec = None
        best_tvec = None
        best_count = 0
        
        for method, reproj_thresh in methods:
            try:
                success, rvec, tvec, inliers = cv.solvePnPRansac(
                    points_3d, points_2d, self.camera.K, None,
                    iterationsCount=5000,
                    reprojectionError=reproj_thresh,
                    confidence=0.99,
                    flags=method
                )
                
                if success and inliers is not None:
                    n_inliers = len(inliers)
                    if n_inliers > best_count:
                        best_count = n_inliers
                        best_inliers = inliers
                        best_rvec = rvec
                        best_tvec = tvec
            except:
                continue
        
        if best_inliers is None or best_count < 6:
            return False
        
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        inlier_mask[best_inliers.ravel()] = True
        
        if np.sum(inlier_mask) >= 6:
            try:
                success, rvec, tvec = cv.solvePnP(
                    points_3d[inlier_mask], 
                    points_2d[inlier_mask], 
                    self.camera.K, None,
                    rvec=best_rvec, tvec=best_tvec,
                    useExtrinsicGuess=True,
                    flags=cv.SOLVEPNP_ITERATIVE
                )
                if success:
                    best_rvec = rvec
                    best_tvec = tvec
            except:
                pass
        
        R, _ = cv.Rodrigues(best_rvec)
        self.poses[img_idx] = CameraPose(R=R, t=best_tvec.ravel())
        
        # Add observations for inliers
        for idx, pid in enumerate(point_ids):
            if inlier_mask[idx]:
                for other_idx in reconstructed:
                    key = (min(img_idx, other_idx), max(img_idx, other_idx))
                    if key not in self.match_cache:
                        continue
                    for m in self.match_cache[key]:
                        if key[0] == img_idx:
                            my_kp = m.idx1
                        else:
                            my_kp = m.idx2
                        
                        kp_pt = self._get_keypoint_pt(img_idx, my_kp)
                        
                        if np.allclose(kp_pt, points_2d[idx], atol=0.1):
                            if (img_idx, my_kp) not in self.observation_index:
                                self._add_observation(pid, img_idx, my_kp)
                            break
        
        print(f"  Registered image {img_idx}: {best_count}/{len(points_3d)} inliers")
        return True
    
    def triangulate_new_points(self, img_idx: int) -> int:
        """Triangulate new points with registered image"""
        new_count = 0
        current_id = max(self.points_3d.keys()) + 1 if self.points_3d else 0
        
        for other_idx in self.poses.keys():
            if other_idx == img_idx:
                continue
            
            key = (min(img_idx, other_idx), max(img_idx, other_idx))
            if key not in self.match_cache:
                continue
            
            matches = self.match_cache[key]
            
            new_matches = []
            for m in matches:
                if key[0] == img_idx:
                    my_kp, other_kp = m.idx1, m.idx2
                else:
                    my_kp, other_kp = m.idx2, m.idx1
                
                if (img_idx, my_kp) not in self.observation_index and \
                   (other_idx, other_kp) not in self.observation_index:
                    new_matches.append((m, my_kp, other_kp))
            
            if len(new_matches) < 5:
                continue
            
            # Use helper for points
            pts_img = np.array([self._get_keypoint_pt(img_idx, m[1]) for m in new_matches])
            pts_other = np.array([self._get_keypoint_pt(other_idx, m[2]) for m in new_matches])
            
            points_3d, valid = triangulate_points(
                self.camera, self.poses[img_idx], self.poses[other_idx], pts_img, pts_other
            )
            
            for idx, (m, my_kp, other_kp) in enumerate(new_matches):
                if not valid[idx]:
                    continue
                
                self.points_3d[current_id] = points_3d[idx]
                self._add_observation(current_id, img_idx, my_kp)
                self._add_observation(current_id, other_idx, other_kp)
                
                pt = self._get_keypoint_pt(img_idx, my_kp)
                x, y = int(pt[0]), int(pt[1])
                img = self.images[img_idx]['image']
                h, w = img.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    self.point_colors[current_id] = img[y, x][::-1]
                else:
                    self.point_colors[current_id] = np.array([127, 127, 127])
                
                current_id += 1
                new_count += 1
        
        return new_count
    
    def bundle_adjustment_light(self):
        """Simplified Bundle Adjustment"""
        print("\n  Running Bundle Adjustment...")
        
        if len(self.poses) < 3 or len(self.points_3d) < 50:
            print("    Skipped (not enough data)")
            return
        
        cam_indices = sorted(self.poses.keys())
        point_indices = sorted(self.points_3d.keys())
        
        obs_data = []
        for pid in point_indices:
            for (img_idx, kp_idx) in self.observations[pid]:
                if img_idx in self.poses:
                    pt = self._get_keypoint_pt(img_idx, kp_idx)
                    obs_data.append((img_idx, pid, pt[0], pt[1]))
        
        if len(obs_data) < 100:
            print("    Skipped (not enough observations)")
            return
        
        total_error = 0
        for cam_idx, pid, u, v in obs_data:
            pt3d = self.points_3d[pid]
            pose = self.poses[cam_idx]
            pt_cam = pose.transform_points(pt3d.reshape(1, 3))[0]
            if pt_cam[2] > 0:
                proj = self.camera.project(pt_cam.reshape(1, 3))[0]
                total_error += np.sqrt((proj[0] - u)**2 + (proj[1] - v)**2)
        
        initial_error = total_error / len(obs_data)
        print(f"    Initial mean reprojection error: {initial_error:.2f} px")
        
        fixed_cam = cam_indices[0]
        
        for iteration in range(3):
            improved = 0
            
            for cam_idx in cam_indices:
                if cam_idx == fixed_cam:
                    continue
                
                cam_obs = [(pid, u, v) for c, pid, u, v in obs_data if c == cam_idx]
                
                if len(cam_obs) < 6:
                    continue
                
                pts_3d = np.array([self.points_3d[pid] for pid, u, v in cam_obs], dtype=np.float32)
                pts_2d = np.array([[u, v] for pid, u, v in cam_obs], dtype=np.float32)
                
                pose = self.poses[cam_idx]
                rvec, _ = cv.Rodrigues(pose.R)
                tvec = pose.t.reshape(3, 1)
                
                try:
                    success, rvec_new, tvec_new = cv.solvePnP(
                        pts_3d, pts_2d, self.camera.K, None,
                        rvec=rvec, tvec=tvec,
                        useExtrinsicGuess=True,
                        flags=cv.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        R_new, _ = cv.Rodrigues(rvec_new)
                        self.poses[cam_idx] = CameraPose(R=R_new, t=tvec_new.ravel())
                        improved += 1
                except:
                    pass
            
            total_error = 0
            for cam_idx, pid, u, v in obs_data:
                pt3d = self.points_3d[pid]
                pose = self.poses[cam_idx]
                pt_cam = pose.transform_points(pt3d.reshape(1, 3))[0]
                if pt_cam[2] > 0:
                    proj = self.camera.project(pt_cam.reshape(1, 3))[0]
                    total_error += np.sqrt((proj[0] - u)**2 + (proj[1] - v)**2)
            
            current_error = total_error / len(obs_data)
            print(f"    Iteration {iteration+1}: error = {current_error:.2f} px, refined {improved} cameras")
        
        print(f"    Final mean reprojection error: {current_error:.2f} px")
    
    def try_recover_images(self, failed: Set[int]) -> int:
        """Try to recover failed images"""
        recovered = 0
        
        for img_idx in list(failed):
            if self.register_image(img_idx):
                failed.remove(img_idx)
                new_pts = self.triangulate_new_points(img_idx)
                print(f"    Recovered image {img_idx}, +{new_pts} points")
                recovered += 1
        
        return recovered
    
    def reconstruct(self, image_dir: str, max_images: int = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Run full SfM pipeline"""
        t0 = time.time()
        
        print("=" * 60)
        print("LOADING IMAGES")
        print("=" * 60)
        self.images = self.load_images(image_dir, max_images)
        
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images")
        
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION")
        print("=" * 60)
        self.features = self.extract_features()
        
        print("\n" + "=" * 60)
        print("FEATURE MATCHING")
        print("=" * 60)
        window_size = min(12, len(self.images) // 3 + 4)
        self.match_cache = self.match_image_pairs(window_size=window_size)
        
        if not self.match_cache:
            raise ValueError("No valid image pairs found!")
        
        print("\n" + "=" * 60)
        print("INITIALIZATION")
        print("=" * 60)
        init_data = self.find_best_initial_pair()
        if init_data is None:
            raise ValueError("Could not find good initial pair")
        
        if not self.initialize(init_data):
            raise ValueError("Initialization failed")
        
        print("\n" + "=" * 60)
        print("INCREMENTAL RECONSTRUCTION")
        print("=" * 60)
        
        failed: Set[int] = set()
        last_ba_count = 2
        
        while True:
            next_img = self.find_next_image(failed)
            
            if next_img is None:
                if failed:
                    print(f"\n  Attempting to recover {len(failed)} failed images...")
                    recovered = self.try_recover_images(failed)
                    if recovered > 0:
                        continue
                break
            
            print(f"\n→ Adding image {next_img} ({len(self.poses)+1}/{len(self.images)})...")
            
            if not self.register_image(next_img):
                failed.add(next_img)
                print(f"    Failed to register")
                continue
            
            new_pts = self.triangulate_new_points(next_img)
            print(f"    +{new_pts} new 3D points (total: {len(self.points_3d)})")
            
            if len(self.poses) >= last_ba_count + 5:
                self.bundle_adjustment_light()
                last_ba_count = len(self.poses)
        
        self.bundle_adjustment_light()
        
        if failed:
            print(f"\n  Final recovery attempt for {len(failed)} images...")
            self.try_recover_images(failed)
        
        self._normalize_reconstruction()
        
        points = np.array([self.points_3d[i] for i in sorted(self.points_3d.keys())])
        colors = np.array([self.point_colors.get(i, [127, 127, 127]) 
                         for i in sorted(self.points_3d.keys())])
        
        print("\n" + "=" * 60)
        print("RECONSTRUCTION COMPLETE")
        print("=" * 60)
        print(f"  Cameras: {len(self.poses)}/{len(self.images)}")
        print(f"  3D points: {len(points)}")
        print(f"  Failed images: {len(failed)}")
        print(f"  Time: {time.time() - t0:.1f}s")
        
        if failed:
            print(f"  Failed indices: {sorted(failed)[:20]}{'...' if len(failed) > 20 else ''}")
        
        return points, colors, self.poses
    
    def _normalize_reconstruction(self):
        """Center and scale the reconstruction"""
        if not self.points_3d:
            return
        
        points = np.array(list(self.points_3d.values()))
        centroid = np.median(points, axis=0)
        
        for pid in self.points_3d:
            self.points_3d[pid] -= centroid
        
        for img_idx in self.poses:
            R, t = self.poses[img_idx].R, self.poses[img_idx].t
            C = -R.T @ t
            C_new = C - centroid
            t_new = -R @ C_new
            self.poses[img_idx] = CameraPose(R=R, t=t_new)
        
        points_centered = np.array(list(self.points_3d.values()))
        scale = np.percentile(np.linalg.norm(points_centered, axis=1), 90)
        
        if scale > 0:
            target_scale = 10.0
            factor = target_scale / scale
            
            for pid in self.points_3d:
                self.points_3d[pid] *= factor
            
            for img_idx in self.poses:
                R, t = self.poses[img_idx].R, self.poses[img_idx].t
                self.poses[img_idx] = CameraPose(R=R, t=t * factor)
    
    def save_ply(self, output_path: str):
        """Save point cloud to PLY file"""
        points = np.array([self.points_3d[i] for i in sorted(self.points_3d.keys())])
        colors = np.array([self.point_colors.get(i, [127, 127, 127]) 
                         for i in sorted(self.points_3d.keys())])
        
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
        
        print(f"Saved {len(points)} points to {output_path}")