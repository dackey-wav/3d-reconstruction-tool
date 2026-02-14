"""
COLMAP Integration for high-quality reconstruction
"""
import subprocess
import shutil
from pathlib import Path
import os
import sys


def find_colmap():
    """Find COLMAP executable"""
    if 'COLMAP_PATH' in os.environ:
        colmap = Path(os.environ['COLMAP_PATH'])
        if colmap.exists():
            return str(colmap)
    
    common_paths = [
        r"C:\COLMAP\COLMAP.bat",
        r"C:\Program Files\COLMAP\COLMAP.bat",
        "/usr/local/bin/colmap",
        "/usr/bin/colmap",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return shutil.which("colmap")


def run_colmap_reconstruction(image_dir: str, output_dir: str, 
                              use_gpu: bool = True,
                              quality: str = "medium",
                              dense: bool = True) -> bool:
    """
    Run COLMAP reconstruction pipeline
    """
    colmap = find_colmap()
    if not colmap:
        print("COLMAP NOT FOUND! Download from: https://github.com/colmap/colmap/releases")
        return False
    
    # Quality presets
    quality_settings = {
        'low': {
            'max_image_size': 1000,
            'max_num_features': 4096,
            'SiftExtraction.num_threads': 4,
        },
        'medium': {
            'max_image_size': 1600,
            'max_num_features': 8192,
            'SiftExtraction.num_threads': -1,
        },
        'high': {
            'max_image_size': 3200,
            'max_num_features': 16384,
            'SiftExtraction.num_threads': -1,
        },
    }
    
    settings = quality_settings.get(quality, quality_settings['medium'])
    
    print("=" * 60)
    print("COLMAP RECONSTRUCTION")
    print("=" * 60)
    print(f"COLMAP: {colmap}")
    print(f"Images: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Quality: {quality}")
    print(f"GPU: {use_gpu}")
    print()
    
    output_path = Path(output_dir)
    
    # Create clean image directory with only images (no subfolders)
    clean_images_path = output_path / "images"
    database_path = output_path / "database.db"
    sparse_path = output_path / "sparse"
    dense_path = output_path / "dense"
    
    # Clean previous run
    if output_path.exists():
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    clean_images_path.mkdir(exist_ok=True)
    sparse_path.mkdir(exist_ok=True)
    
    # Copy only image files to clean directory
    print("Copying images to clean directory...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_source = Path(image_dir)
    
    copied = 0
    for img_file in image_source.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            shutil.copy2(img_file, clean_images_path / img_file.name)
            copied += 1
    
    print(f"  Copied {copied} images")
    
    if copied == 0:
        print("ERROR: No images found!")
        return False
    
    try:
        # Step 1: Feature extraction
        print("\n[1/5] Extracting features...")
        feature_cmd = [
            colmap, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(clean_images_path),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--SiftExtraction.max_num_features", str(settings['max_num_features']),
        ]
        if not use_gpu:
            feature_cmd.extend(["--SiftExtraction.use_gpu", "0"])
        subprocess.run(feature_cmd, check=True)
        
        # Step 2: Feature matching
        print("\n[2/5] Matching features...")
        match_cmd = [
            colmap, "exhaustive_matcher",
            "--database_path", str(database_path),
        ]
        if not use_gpu:
            match_cmd.extend(["--SiftMatching.use_gpu", "0"])
        subprocess.run(match_cmd, check=True)
        
        # Step 3: Sparse reconstruction
        print("\n[3/5] Sparse reconstruction...")
        subprocess.run([
            colmap, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(clean_images_path),
            "--output_path", str(sparse_path),
        ], check=True)
        
        # Find the reconstruction
        sparse_model = None
        for subdir in sparse_path.iterdir():
            if subdir.is_dir():
                sparse_model = subdir
                break
        
        if not sparse_model:
            print("ERROR: No sparse model created!")
            return False
        
        print(f"  Found model: {sparse_model}")
        
        # Export sparse to PLY
        sparse_ply = output_path / "sparse.ply"
        subprocess.run([
            colmap, "model_converter",
            "--input_path", str(sparse_model),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY",
        ], check=True)
        print(f"  Sparse cloud: {sparse_ply}")
        
        if not dense:
            print("\n" + "=" * 60)
            print("SPARSE RECONSTRUCTION COMPLETE!")
            print("=" * 60)
            print(f"\nResults in: {output_path}")
            print(f"  sparse.ply - Open in MeshLab")
            return True
        
        # Step 4: Dense reconstruction
        print("\n[4/5] Preparing for dense reconstruction...")
        dense_path.mkdir(exist_ok=True)
        
        # Image undistorter
        subprocess.run([
            colmap, "image_undistorter",
            "--image_path", str(clean_images_path),
            "--input_path", str(sparse_model),
            "--output_path", str(dense_path),
            "--output_type", "COLMAP",
            "--max_image_size", str(settings['max_image_size']),
        ], check=True)
        
        # Patch match stereo
        print("\n  Running PatchMatch stereo (this may take a while)...")
        subprocess.run([
            colmap, "patch_match_stereo",
            "--workspace_path", str(dense_path),
            "--PatchMatchStereo.geom_consistency", "true",
        ], check=True)
        
        # Fusion
        print("\n[5/5] Fusing depth maps...")
        fused_path = output_path / "dense.ply"
        subprocess.run([
            colmap, "stereo_fusion",
            "--workspace_path", str(dense_path),
            "--output_path", str(fused_path),
            "--StereoFusion.min_num_pixels", "3",
        ], check=True)
        
        print("\n" + "=" * 60)
        print("RECONSTRUCTION COMPLETE!")
        print("=" * 60)
        print(f"\nResults in: {output_path}")
        print(f"  sparse.ply - Sparse point cloud")
        print(f"  dense.ply  - Dense point cloud")
        
        # Count points
        if fused_path.exists():
            with open(fused_path, 'rb') as f:
                header = f.read(2000).decode('utf-8', errors='ignore')
                for line in header.split('\n'):
                    if line.startswith("element vertex"):
                        n_pts = int(line.split()[-1])
                        print(f"\n  Dense points: {n_pts:,}")
                        break
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed!")
        print(f"  {' '.join(str(x) for x in e.cmd[:3])}...")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run COLMAP reconstruction')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--sparse-only', action='store_true', help='Skip dense reconstruction')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    image_dir = project_root / 'data' / 'samples' / args.dataset
    output_dir = image_dir / 'colmap_reconstruction'
    
    if not image_dir.exists():
        print(f"ERROR: Not found: {image_dir}")
        sys.exit(1)
    
    success = run_colmap_reconstruction(
        str(image_dir),
        str(output_dir),
        use_gpu=not args.no_gpu,
        quality=args.quality,
        dense=not args.sparse_only
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()