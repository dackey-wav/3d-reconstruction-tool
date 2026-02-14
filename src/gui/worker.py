import sys
from PyQt5.QtCore import QObject, QProcess, pyqtSignal


class ReconstructionWorker(QObject):
    """
    Runs reconstruction scripts (SfM or COLMAP) in a separate process
    to keep the GUI responsive.
    """
    log_updated = pyqtSignal(str)      # Complete log line
    status_updated = pyqtSignal(str)   # Short status for progress bar
    finished = pyqtSignal(bool)        # Success/Fail

    def __init__(self):
        super().__init__()
        self.process = None

    def stop(self):
        if self.process and self.process.state() == QProcess.Running:
            self.process.kill()

    def run_reconstruction(self, options):
        """
        Args:
            options: dict with keys:
                - dataset (str)
                - method (str): 'colmap_dense', 'colmap_sparse', 
                                'custom_neural_stereo', 'custom_neural_mvs', etc.
                - quality (str, optional): 'low'/'medium'/'high' for COLMAP
                - use_gpu (bool, optional): GPU flag for COLMAP
                - sparse_only (bool, optional): skip dense for COLMAP
                - max_images (int, optional): limit images for custom pipeline
                - neural (bool, optional): use LightGlue
                - stereo (bool, optional): plane sweep
                - mvs (bool, optional): PatchMatch MVS
        """
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_output)
        self.process.finished.connect(self._handle_finished)

        # Handle both old-style (str) and new-style (dict) calls
        if isinstance(options, str):
            # Legacy: options is dataset name, treat as neural+stereo
            options = {
                'dataset': options,
                'method': 'custom_neural_stereo',
                'neural': True,
                'stereo': True,
            }

        dataset = options.get('dataset', '')
        method = options.get('method', '')

        cmd = [sys.executable, "-m"]

        if method == 'colmap_dense':
            quality = options.get('quality', 'medium')
            args = ["src.tools.run_colmap", dataset, "--quality", quality]
            if not options.get('use_gpu', True):
                args.append("--no-gpu")
            cmd.extend(args)
            self.status_updated.emit("Starting COLMAP Dense Reconstruction...")

        elif method == 'colmap_sparse':
            quality = options.get('quality', 'medium')
            args = ["src.tools.run_colmap", dataset, "--quality", quality, "--sparse-only"]
            if not options.get('use_gpu', True):
                args.append("--no-gpu")
            cmd.extend(args)
            self.status_updated.emit("Starting COLMAP Sparse SfM...")

        else:
            # Custom pipeline
            args = ["src.run_reconstruction", dataset]

            if options.get('neural', False):
                args.append("--neural")

            if options.get('mvs', False):
                args.append("--mvs")
            elif options.get('stereo', False):
                args.append("--stereo")
            elif options.get('dense', False):
                args.append("--dense")
            else:
                args.append("--fast")

            max_images = options.get('max_images')
            if max_images is not None:
                args.extend(["--max-images", str(max_images)])

            cmd.extend(args)
            self.status_updated.emit("Starting Custom Reconstruction...")

        print(f"Executing: {' '.join(cmd)}")
        self.process.start(cmd[0], cmd[1:])

    def _handle_output(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode("utf8", errors='ignore')

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Send raw log
            self.log_updated.emit(line)

            # Parse for short status updates
            lower_line = line.lower()
            if "extracting features" in lower_line:
                self.status_updated.emit("Step 1/4: Extracting Features...")
            elif "matching features" in lower_line:
                self.status_updated.emit("Step 2/4: Matching Features...")
            elif "reconstruction" in lower_line and "sparse" in lower_line:
                self.status_updated.emit("Step 3/4: Sparse Reconstruction...")
            elif "dense" in lower_line or "stereo" in lower_line:
                self.status_updated.emit("Step 4/4: Dense Reconstruction...")
            elif "saved" in lower_line and ".ply" in lower_line:
                self.status_updated.emit("Saving results...")

    def _handle_finished(self, exit_code, exit_status):
        success = (exit_code == 0)
        self.finished.emit(success)