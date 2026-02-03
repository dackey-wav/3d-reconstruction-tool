"""
Neural Feature Matching using LightGlue + SuperPoint

Provides significantly better matching than SIFT for challenging cases:
- Large viewpoint changes
- Repetitive textures
- Low texture regions
"""
import numpy as np
import cv2 as cv
import torch
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass


class NeuralMatch(NamedTuple):
    """Match between two keypoints (compatible with FeatureMatch)"""
    idx1: int
    idx2: int
    distance: float = 0.0


@dataclass
class NeuralFeatures:
    """Features extracted by SuperPoint"""
    keypoints: np.ndarray      # (N, 2) - x, y coordinates
    descriptors: np.ndarray    # (256, N) - descriptors
    scores: np.ndarray         # (N,) - keypoint scores
    image_size: Tuple[int, int]  # (H, W)


class NeuralMatcher:
    """
    Neural feature extractor and matcher using SuperPoint + LightGlue
    """
    
    def __init__(self, max_keypoints: int = 2048):
        self.max_keypoints = max_keypoints
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Neural matcher using: {self.device}")
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load SuperPoint and LightGlue models"""
        try:
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import load_image, rbd
            
            print("Loading SuperPoint + LightGlue models...")
            
            self.extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
            
            print("Models loaded!")
            
        except ImportError as e:
            raise ImportError(
                f"LightGlue not installed. Run:\n"
                f"  pip install git+https://github.com/cvg/LightGlue.git\n"
                f"Original error: {e}"
            )
    
    def extract(self, image: np.ndarray) -> NeuralFeatures:
        """Extract SuperPoint features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(gray).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        
        # Extract features
        with torch.no_grad():
            feats = self.extractor.extract(img_tensor)
        
        # Convert to numpy
        keypoints = feats['keypoints'][0].cpu().numpy()      # (N, 2)
        descriptors = feats['descriptors'][0].cpu().numpy()  # (N, 256) or (256, N)
        scores = feats['keypoint_scores'][0].cpu().numpy()   # (N,)
        
        # Ensure descriptors are (256, N) format
        if descriptors.shape[0] != 256 and len(descriptors.shape) == 2:
            descriptors = descriptors.T
        
        return NeuralFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            image_size=(h, w)
        )
    
    def match(self, feats1: NeuralFeatures, feats2: NeuralFeatures) -> List[NeuralMatch]:
        """Match features between two images using LightGlue"""
        
        # Handle empty features
        if len(feats1.keypoints) == 0 or len(feats2.keypoints) == 0:
            return []
        
        # Prepare tensors for LightGlue
        h1, w1 = feats1.image_size
        h2, w2 = feats2.image_size
        
        # Ensure correct descriptor format (N, 256)
        desc1 = feats1.descriptors
        desc2 = feats2.descriptors
        
        if desc1.shape[0] == 256:
            desc1 = desc1.T
        if desc2.shape[0] == 256:
            desc2 = desc2.T
        
        data1 = {
            'keypoints': torch.from_numpy(feats1.keypoints).float().unsqueeze(0).to(self.device),
            'descriptors': torch.from_numpy(desc1).float().unsqueeze(0).to(self.device),
            'keypoint_scores': torch.from_numpy(feats1.scores).float().unsqueeze(0).to(self.device),
            'image_size': torch.tensor([[h1, w1]]).to(self.device)
        }
        
        data2 = {
            'keypoints': torch.from_numpy(feats2.keypoints).float().unsqueeze(0).to(self.device),
            'descriptors': torch.from_numpy(desc2).float().unsqueeze(0).to(self.device),
            'keypoint_scores': torch.from_numpy(feats2.scores).float().unsqueeze(0).to(self.device),
            'image_size': torch.tensor([[h2, w2]]).to(self.device)
        }
        
        # Run matching
        with torch.no_grad():
            result = self.matcher({'image0': data1, 'image1': data2})
        
        # Extract matches
        matches_tensor = result['matches0'][0].cpu().numpy()  # (N,) indices or -1 for no match
        
        matches = []
        for idx1, idx2 in enumerate(matches_tensor):
            if idx2 >= 0:  # Valid match
                matches.append(NeuralMatch(idx1=idx1, idx2=int(idx2), distance=0.0))
        
        return matches
    
    def match_pair_geometric(self, feats1: NeuralFeatures, feats2: NeuralFeatures,
                            min_matches: int = 15) -> Tuple[List[NeuralMatch], Optional[np.ndarray]]:
        """
        Match features with geometric verification (Fundamental matrix)
        Returns matches and fundamental matrix
        """
        # Get raw matches
        matches = self.match(feats1, feats2)
        
        if len(matches) < min_matches:
            return [], None
        
        # Extract matched points
        pts1 = np.array([feats1.keypoints[m.idx1] for m in matches], dtype=np.float64)
        pts2 = np.array([feats2.keypoints[m.idx2] for m in matches], dtype=np.float64)
        
        # Validate points array
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            return [], None
        
        # Check for valid coordinates
        if np.any(np.isnan(pts1)) or np.any(np.isnan(pts2)):
            return [], None
        
        if np.any(np.isinf(pts1)) or np.any(np.isinf(pts2)):
            return [], None
        
        # Ensure contiguous arrays with correct shape
        pts1 = np.ascontiguousarray(pts1.reshape(-1, 2))
        pts2 = np.ascontiguousarray(pts2.reshape(-1, 2))
        
        try:
            # Geometric verification with RANSAC
            F, mask = cv.findFundamentalMat(
                pts1, pts2, 
                cv.FM_RANSAC, 
                ransacReprojThreshold=2.0, 
                confidence=0.999
            )
            
            if F is None or mask is None:
                return [], None
            
            # Filter matches by inlier mask
            inlier_mask = mask.ravel() == 1
            filtered_matches = [m for m, is_inlier in zip(matches, inlier_mask) if is_inlier]
            
            if len(filtered_matches) < min_matches:
                return [], None
            
            return filtered_matches, F
            
        except cv.error as e:
            # OpenCV error - return empty
            print(f"    Warning: geometric verification failed: {e}")
            return [], None


def convert_neural_to_cv_keypoints(feats: NeuralFeatures) -> List[cv.KeyPoint]:
    """Convert NeuralFeatures keypoints to OpenCV KeyPoint format"""
    keypoints = []
    for i, (x, y) in enumerate(feats.keypoints):
        kp = cv.KeyPoint(
            x=float(x), 
            y=float(y), 
            size=10.0,  # Default size
            angle=-1,
            response=float(feats.scores[i]) if i < len(feats.scores) else 1.0,
            octave=0,
            class_id=-1
        )
        keypoints.append(kp)
    return keypoints
