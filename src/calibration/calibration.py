"""
Improved camera calibration with automatic outlier rejection
"""
import numpy as np
import cv2 as cv
import glob
import os

def calibrate_camera_robust(image_dir, output_dir, checkerboard_size=(9, 6), 
                           max_error_threshold=2.0):
    """
    Robust camera calibration with iterative outlier rejection
    
    Args:
        image_dir: directory with chessboard images
        output_dir: where to save results
        checkerboard_size: (width, height) in internal corners
        max_error_threshold: reject images with error > this (pixels)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    CHECKERBOARD = checkerboard_size
    
    # Prepare object points (3D coordinates of chessboard corners)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    
    # Find all images (avoid duplicates)
    image_paths = set(
        glob.glob(os.path.join(image_dir, '*.JPG')) +
        glob.glob(os.path.join(image_dir, '*.jpg')) +
        glob.glob(os.path.join(image_dir, '*.jpeg')) +
        glob.glob(os.path.join(image_dir, '*.JPEG')) +
        glob.glob(os.path.join(image_dir, '*.png'))
    )
    images = sorted(image_paths)
    
    print(f"Found {len(images)} images")
    
    # Detect corners in all images
    all_data = []  # [(fname, objp, corners, img_shape), ...]
    
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"✗ Failed to load: {fname}")
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Improve contrast
        gray = cv.equalizeHist(gray)
        
        flags = (cv.CALIB_CB_ADAPTIVE_THRESH +
                cv.CALIB_CB_NORMALIZE_IMAGE +
                cv.CALIB_CB_FAST_CHECK)
        
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, flags)
        
        if ret:
            # Refine corners to sub-pixel accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            all_data.append((fname, objp, corners2, gray.shape[::-1]))
            print(f"✓ Found corners in: {os.path.basename(fname)}")
        else:
            print(f"✗ No corners in: {os.path.basename(fname)}")
    
    if len(all_data) < 10:
        print(f"\n⚠ ERROR: Only {len(all_data)} images with detected corners!")
        print("Need at least 10 images for reliable calibration.")
        return None
    
    print(f"\n{'='*60}")
    print("ITERATIVE CALIBRATION WITH OUTLIER REJECTION")
    print(f"{'='*60}")
    
    # Initial calibration with all images
    objpoints = [d[1] for d in all_data]
    imgpoints = [d[2] for d in all_data]
    img_size = all_data[0][3]
    
    # Use simpler distortion model (k1, k2, p1, p2) - fix k3 to 0
    # This prevents overfitting with unstable high-order coefficients
    flags = cv.CALIB_FIX_K3
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, flags=flags
    )
    
    print(f"\nInitial calibration RMS error: {ret:.4f}")
    
    # Compute per-image errors
    errors = []
    for i in range(len(all_data)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        errors.append((error, i, all_data[i][0]))
    
    # Remove worst outliers iteratively
    iteration = 0
    while True:
        max_error = max(e[0] for e in errors)
        
        if max_error < max_error_threshold or len(errors) < 10:
            break
        
        iteration += 1
        errors_sorted = sorted(errors, key=lambda x: x[0], reverse=True)
        worst_error, worst_idx, worst_fname = errors_sorted[0]
        
        print(f"\nIteration {iteration}: Removing {os.path.basename(worst_fname)} (error: {worst_error:.4f})")
        
        # Remove worst image
        all_data = [d for i, d in enumerate(all_data) if i != worst_idx]
        
        # Re-calibrate with same flags
        objpoints = [d[1] for d in all_data]
        imgpoints = [d[2] for d in all_data]
        
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, img_size, None, None, flags=flags
        )
        
        # Recompute errors
        errors = []
        for i in range(len(all_data)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            errors.append((error, i, all_data[i][0]))
        
        mean_error = np.mean([e[0] for e in errors])
        print(f"  New RMS: {ret:.4f}, Mean error: {mean_error:.4f}, Images remaining: {len(all_data)}")
    
    # Final results
    mean_error = np.mean([e[0] for e in errors])
    
    print(f"\n{'='*60}")
    print("FINAL CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"RMS error: {ret:.4f}")
    print(f"Mean reprojection error: {mean_error:.4f}")
    print(f"Images used: {len(all_data)}")
    print(f"\nCamera Matrix:\n{mtx}")
    print(f"\nDistortion Coefficients:\n{dist}")
    
    # Save results
    calibration_file = os.path.join(output_dir, 'calibration_data.npz')
    np.savez(calibration_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    
    txt_file = os.path.join(output_dir, 'calibration_data.txt')
    with open(txt_file, 'w') as f:
        f.write("CAMERA CALIBRATION RESULTS\n\n")
        f.write(f"RMS Re-projection Error: {ret}\n")
        f.write(f"Mean Re-projection Error: {mean_error}\n")
        f.write(f"Images used: {len(all_data)}\n\n")
        f.write("Camera Matrix:\n")
        f.write(np.array2string(mtx, separator=', ') + "\n\n")
        f.write("Distortion coefficients:\n")
        f.write(np.array2string(dist, separator=', ') + "\n")
    
    print(f"\n✓ Calibration saved to {calibration_file}")
    print(f"✓ Text report saved to {txt_file}")
    
    # Show per-image errors
    print(f"\nPer-Image Errors:")
    for error, idx, fname in sorted(errors):
        status = "✓" if error < 0.5 else "⚠" if error < 1.0 else "✗"
        print(f"  {status} [{idx:2d}] {os.path.basename(fname):30s}  {error:.4f}")
    
    return mtx, dist


if __name__ == '__main__':
    calibrate_camera_robust(
        image_dir='data/calibration_images/chessboard',
        output_dir='src/calibration/calibration_results',
        checkerboard_size=(9, 6),
        max_error_threshold=0.5  # Strict threshold for good calibration
    )