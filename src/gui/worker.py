from PyQt5.QtCore import QObject, pyqtSignal, QProcess
import sys

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

    def run_reconstruction(self, dataset, method='neural'):
        """
        Args:
            method: 'neural', 'colmap_sparse', 'colmap_dense'
        """
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_output)
        self.process.finished.connect(self._handle_finished)
        
        # Build command based on method
        cmd = [sys.executable, "-m"]
        
        if method == 'colmap_dense':
            # Run COLMAP tool
            cmd.extend(["src.tools.run_colmap", dataset, "--quality", "medium"])
            self.status_updated.emit("Starting COLMAP Dense Reconstruction...")
            
        elif method == 'colmap_sparse':
             # Run COLMAP Sparse
            cmd.extend(["src.tools.run_colmap", dataset, "--sparse-only"])
            self.status_updated.emit("Starting COLMAP Sparse SfM...")
            
        elif method == 'neural':
            # Run our pipeline
            cmd.extend(["src.run_reconstruction", dataset, "--neural", "--stereo"])
            self.status_updated.emit("Starting Neural Reconstruction...")
            
        else:
            self.log_updated.emit("Unknown method")
            self.finished.emit(False)
            return

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
                 self.status_updated.emit("Saving Results...")

    def _handle_finished(self, exit_code, exit_status):
        success = (exit_code == 0) and (exit_status == QProcess.NormalExit)
        self.finished.emit(success)