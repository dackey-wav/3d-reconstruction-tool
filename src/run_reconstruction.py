"""
Main entry point for 3D reconstruction

Usage:
    python -m src.run_reconstruction <dataset_name> [options]
    
Examples:
    python -m src.run_reconstruction kitchen --neural --mvs    # Best quality
    python -m src.run_reconstruction room --neural --stereo    # Faster
    python -m src.run_reconstruction table --neural --fast     # Quick test
"""
import argparse
from pathlib import Path
import sys

src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core import SfMPipeline, load_calibration
from core.utils import save_ply, save_cameras_ply


def main():
    parser = argparse.ArgumentParser(description='3D Reconstruction from Images')
    parser.add_argument('dataset', help='Dataset name (subfolder in data/samples/)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--dense', action='store_true',
                        help='Dense SIFT matching (slow, CPU)')
    parser.add_argument('--stereo', action='store_true',
                        help='Plane sweep stereo (fast, basic quality)')
    parser.add_argument('--mvs', action='store_true',
                        help='PatchMatch MVS (best quality, GPU accelerated)')
    parser.add_argument('--combined', action='store_true',
                        help='Combined stereo + dense (deprecated)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: sparse only, reduced resolution')
    parser.add_argument('--neural', action='store_true',
                        help='Use LightGlue neural matcher (recommended)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    calibration_path = project_root / 'src' / 'calibration' / 'calibration_results' / 'calibration_data.npz'
    image_dir = project_root / 'data' / 'samples' / args.dataset
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = image_dir / 'reconstruction'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check paths
    if not calibration_path.exists():
        print(f"ERROR: Calibration file not found: {calibration_path}")
        print("Run camera calibration first!")
        sys.exit(1)
    
    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 60)
    print("3D RECONSTRUCTION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Images: {image_dir}")
    print(f"Output: {output_dir}")
    
    mode_parts = []
    if args.fast:
        mode_parts.append("FAST")
    if args.neural:
        mode_parts.append("LightGlue")
    else:
        mode_parts.append("SIFT")
    
    if args.mvs:
        mode_parts.append("PatchMatch MVS")
    elif args.stereo:
        mode_parts.append("Plane Sweep")
    elif args.dense:
        mode_parts.append("Dense SIFT")
    else:
        mode_parts.append("Sparse only")
    
    print(f"Mode: {' + '.join(mode_parts)}")
    print()
    
    # Run sparse SfM
    pipeline = SfMPipeline(str(calibration_path), fast_mode=args.fast, neural_mode=args.neural)
    points, colors, poses = pipeline.reconstruct(str(image_dir), args.max_images)
    
    # Save sparse result
    sparse_output = output_dir / 'sparse.ply'
    save_ply(points, colors, str(sparse_output))
    
    # Save camera positions
    cameras_output = output_dir / 'cameras.ply'
    save_cameras_ply(poses, str(cameras_output))
    
    # Convert poses to CameraPose objects
    from core.camera import CameraPose
    pose_dict = {}
    for idx, pose in poses.items():
        if hasattr(pose, 'R'):
            pose_dict[idx] = pose
        else:
            R, t = pose
            pose_dict[idx] = CameraPose(R=R, t=t.ravel())
    
    # Dense reconstruction
    run_dense = (args.mvs or args.stereo or args.dense) and not args.fast
    
    if run_dense and len(poses) >= 3:
        camera = load_calibration(str(calibration_path))
        
        if args.mvs:
            # PatchMatch MVS - best quality
            print("\n" + "=" * 60)
            print("Starting PatchMatch MVS...")
            print("=" * 60)
            
            from core.mvs_patchmatch import PatchMatchMVS
            mvs = PatchMatchMVS(
                camera, 
                scale=0.25,           # 1/4 resolution for speed
                num_iterations=3,     # PatchMatch iterations
                min_views=3           # Minimum consistent views
            )
            dense_points, dense_colors = mvs.reconstruct(
                pipeline.images, 
                pose_dict, 
                sparse_points=points
            )
            
            if len(dense_points) > 0:
                dense_output = output_dir / 'dense_mvs.ply'
                save_ply(dense_points, dense_colors, str(dense_output))
                print(f"Saved {len(dense_points):,} points to {dense_output}")
        
        elif args.stereo:
            # Plane sweep stereo - faster but basic
            from core.dense_stereo import DenseStereoReconstructor
            stereo = DenseStereoReconstructor(camera, scale=0.25)
            dense_points, dense_colors = stereo.reconstruct(
                pipeline.images, pose_dict, max_pairs=30
            )
            
            if len(dense_points) > 0:
                dense_output = output_dir / 'dense_stereo.ply'
                save_ply(dense_points, dense_colors, str(dense_output))
                print(f"Saved {len(dense_points):,} points to {dense_output}")
        
        elif args.dense:
            # Dense SIFT matching - slow CPU method
            print("\nWarning: --dense uses slow CPU matching. Consider --mvs instead.")
            from core.dense_reconstruction import DenseReconstructor
            dense_recon = DenseReconstructor(camera)
            dense_points, dense_colors = dense_recon.reconstruct(
                pipeline.images, pose_dict, window=8
            )
            
            if len(dense_points) > 0:
                dense_output = output_dir / 'dense.ply'
                save_ply(dense_points, dense_colors, str(dense_output))
    
    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"  - sparse.ply: {len(points):,} points")
    if args.mvs:
        print(f"  - dense_mvs.ply: PatchMatch MVS cloud")
    elif args.stereo:
        print(f"  - dense_stereo.ply: Plane sweep cloud")
    elif args.dense:
        print(f"  - dense.ply: Dense SIFT cloud")
    print(f"  - cameras.ply: {len(poses)} camera positions")


if __name__ == '__main__':
    main()