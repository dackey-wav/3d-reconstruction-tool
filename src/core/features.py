"""
Feature detection and matching

Uses SIFT for robust feature detection.
Custom implementation of ratio test matching.
"""
import numpy as np
import cv2 as cv
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ImageFeatures:
    """Features extracted from one image"""
    keypoints: List[cv.KeyPoint]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]
    
    @property
    def points(self) -> np.ndarray:
        """Nx2 array of keypoint coordinates"""
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float32)
    
    def __len__(self):
        return len(self.keypoints)


@dataclass
class FeatureMatch:
    """Match between two images"""
    idx1: int  # keypoint index in image 1
    idx2: int  # keypoint index in image 2
    distance: float


class FeatureExtractor:
    """
    SIFT-based feature extractor
    
    Why SIFT:
    - Scale and rotation invariant
    - Robust to illumination changes
    - High quality descriptors (128-dim)
    """
    
    def __init__(self, n_features: int = 10000):
        """
        Args:
            n_features: maximum number of features to detect
        """
        self.detector = cv.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=0.03,  # Снижен для большего количества фич
            edgeThreshold=15,        # Увеличен для лучших фич на краях
            sigma=1.6
        )
    
    def extract(self, image: np.ndarray) -> ImageFeatures:
        """
        Extract features from an image
        
        Args:
            image: BGR or grayscale image
            
        Returns:
            ImageFeatures object
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for better contrast (helps with indoor scenes)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Detect and compute
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            descriptors = np.array([])
        
        return ImageFeatures(
            keypoints=list(keypoints),
            descriptors=descriptors,
            image_shape=gray.shape
        )


class FeatureMatcher:
    """
    Feature matcher with Lowe's ratio test
    
    Uses FLANN for fast approximate nearest neighbor search.
    """
    
    def __init__(self, ratio_threshold: float = 0.8):
        """
        Args:
            ratio_threshold: Lowe's ratio test threshold (0.7-0.8 typical)
        """
        self.ratio_threshold = ratio_threshold
        
        # FLANN parameters for SIFT (float descriptors)
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=150)  # Увеличено для лучшего качества
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)
    
    def match(self, features1: ImageFeatures, 
              features2: ImageFeatures,
              cross_check: bool = True) -> List[FeatureMatch]:
        """
        Match features between two images
        
        Args:
            features1: features from first image
            features2: features from second image
            cross_check: if True, only keep symmetric matches
            
        Returns:
            List of FeatureMatch objects
        """
        if len(features1.descriptors) < 2 or len(features2.descriptors) < 2:
            return []
        
        # Convert to float32 for FLANN
        desc1 = features1.descriptors.astype(np.float32)
        desc2 = features2.descriptors.astype(np.float32)
        
        # Forward matching with ratio test
        try:
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv.error:
            return []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(FeatureMatch(
                        idx1=m.queryIdx,
                        idx2=m.trainIdx,
                        distance=m.distance
                    ))
        
        if not cross_check:
            return good_matches
        
        # Cross-check: match in reverse direction
        try:
            raw_matches_rev = self.matcher.knnMatch(desc2, desc1, k=2)
        except cv.error:
            return good_matches
        
        # Build reverse match set
        reverse_matches = set()
        for match_pair in raw_matches_rev:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    reverse_matches.add((m.trainIdx, m.queryIdx))
        
        # Keep only symmetric matches
        symmetric_matches = [
            m for m in good_matches 
            if (m.idx1, m.idx2) in reverse_matches
        ]
        
        return symmetric_matches
    
    def match_pair_geometric(self, features1: ImageFeatures,
                            features2: ImageFeatures,
                            min_matches: int = 20) -> Tuple[List[FeatureMatch], np.ndarray]:
        """
        Match features and filter with geometric verification (RANSAC)
        
        Returns:
            matches: geometrically verified matches
            F: fundamental matrix (or None if failed)
        """
        raw_matches = self.match(features1, features2, cross_check=True)
        
        if len(raw_matches) < min_matches:
            return [], None
        
        # Get point coordinates
        pts1 = np.array([features1.keypoints[m.idx1].pt for m in raw_matches])
        pts2 = np.array([features2.keypoints[m.idx2].pt for m in raw_matches])
        
        # RANSAC with fundamental matrix
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 
                                        ransacReprojThreshold=2.0,  # Увеличен
                                        confidence=0.999)
        
        if F is None or mask is None:
            return [], None
        
        # Filter matches by inlier mask
        inlier_matches = [m for m, is_inlier in zip(raw_matches, mask.ravel()) if is_inlier]
        
        return inlier_matches, F