"""Interactive calibration validation tool"""
import numpy as np
import cv2 as cv
import glob
import os

def validate_calibration():
    """
    Validate calibration by showing reprojection errors per image
    Helps identify bad images to exclude
    """
    calibration_file = 'src/calibration/calibration_results/calibration_data.npz'
    
    if not os.path.exists(calibration_file):
        print("Error: calibration_data.npz not found!")
        print("Run calibration.py first")
        return
    
    data = np.load(calibration_file)
    mtx = data['mtx']
    dist = data['dist']
    rvecs = data['rvecs']
    tvecs = data['tvecs']
    
    print("\n" + "="*60)
    print("CALIBRATION VALIDATION")
    print("="*60)
    print(f"\nCamera Matrix:\n{mtx}\n")
    print(f"Distortion Coefficients:\n{dist}\n")
    
    # Recompute objpoints and imgpoints from images
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    CHECKERBOARD = (9, 6)
    
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    
    objpoints = []
    imgpoints = []
    image_names = []
    
    # FIX: Use set to avoid duplicates
    image_paths = set(
        glob.glob('data/calibration_images/chessboard/*.JPG') +
        glob.glob('data/calibration_images/chessboard/*.jpg') +
        glob.glob('data/calibration_images/chessboard/*.jpeg') +
        glob.glob('data/calibration_images/chessboard/*.JPEG')
    )
    images = sorted(image_paths)
    
    print(f"Found {len(images)} unique images\n")
    
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        
        flags = (cv.CALIB_CB_ADAPTIVE_THRESH +
                cv.CALIB_CB_NORMALIZE_IMAGE +
                cv.CALIB_CB_FAST_CHECK)
        
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, flags)
        
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            image_names.append(os.path.basename(fname))
    
    print(f"Successfully found corners in {len(objpoints)} images")
    print(f"Calibration was done with {len(rvecs)} images")
    
    # FIX: Check if counts match
    if len(objpoints) != len(rvecs):
        print(f"\n⚠ ERROR: Mismatch between images found now ({len(objpoints)}) "
              f"and calibration data ({len(rvecs)})")
        print("This means calibration data is outdated or images were added/removed.")
        print("Solution: Re-run calibration.py")
        return None
    
    # Compute per-image reprojection errors
    print("\nPer-Image Reprojection Errors:")
    print("-" * 60)
    
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        errors.append((error, image_names[i], i))
        
        status = "✓ GOOD" if error < 0.5 else "⚠ WARNING" if error < 1.0 else "✗ BAD"
        print(f"[{i:2d}] {image_names[i]:30s}  Error: {error:.4f}  {status}")
    
    # Sort by error
    errors.sort(reverse=True)
    
    print("\n" + "="*60)
    print("WORST IMAGES (consider removing):")
    print("="*60)
    for error, name, idx in errors[:5]:
        print(f"  [{idx}] {name}: {error:.4f}")
    
    mean_error = np.mean([e[0] for e in errors])
    print(f"\nMean reprojection error: {mean_error:.4f}")
    print(f"Target: < 0.5 pixels (excellent), < 1.0 (acceptable)")
    
    # Check if there are BAD images
    bad_images = [(e, n, i) for e, n, i in errors if e > 1.0]
    if bad_images:
        print(f"\n⚠ WARNING: {len(bad_images)} images with error > 1.0 pixels!")
        print("\nRecommendations:")
        print("  1. Remove these bad images from data/calibration_images/chessboard/:")
        for error, name, idx in bad_images:
            print(f"     - {name} (error: {error:.2f})")
        print("  2. Re-run calibration.py")
        print("  3. Check lighting conditions (no shadows/glare on chessboard)")
        print("  4. Ensure chessboard is perfectly flat (mount on rigid cardboard)")
    else:
        print("\n✓ All images have acceptable reprojection error!")
    
    return errors

if __name__ == '__main__':
    validate_calibration()