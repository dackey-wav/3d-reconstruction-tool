import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QLabel, QComboBox, QPushButton, QProgressBar,
    QMessageBox, QGroupBox, QHBoxLayout, QSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Import our components
from .viewer import ViewerTab
from .worker import ReconstructionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Reconstruction")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self._get_style())
        
        # Central Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create Tabs
        self.tab_create = self._create_creator_tab()
        self.tab_viewer = ViewerTab()
        
        self.tabs.addTab(self.tab_create, "Create Model")
        self.tabs.addTab(self.tab_viewer, "View Model")
        
        # Worker logic
        self.worker = ReconstructionWorker()
        self.worker.log_updated.connect(lambda text: print(f"[LOG] {text}"))
        self.worker.status_updated.connect(self._update_status)
        self.worker.finished.connect(self._reconstruction_finished)
        
        # Populate datasets
        self._load_datasets()
        
        # Connect method change to update options visibility
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        self._on_method_changed()

    def _create_creator_tab(self):
        """Builds the Create/Run tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        
        # === Dataset Selection ===
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QHBoxLayout(dataset_group)
        dataset_layout.setContentsMargins(15, 25, 15, 15)
        
        dataset_layout.addWidget(QLabel("Select dataset:"))
        self.combo_dataset = QComboBox()
        self.combo_dataset.setMinimumWidth(200)
        dataset_layout.addWidget(self.combo_dataset)
        dataset_layout.addStretch()
        
        layout.addWidget(dataset_group)
        
        # === Algorithm Selection ===
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(15, 25, 15, 15)
        
        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.setMinimumWidth(300)
        
        # All available methods
        self.combo_method.addItem("COLMAP Dense (Best Quality)", "colmap_dense")
        self.combo_method.addItem("COLMAP Sparse (Fast SfM)", "colmap_sparse")
        self.combo_method.addItem("Custom Pipeline - Neural + Stereo", "custom_neural_stereo")
        self.combo_method.addItem("Custom Pipeline - Neural + MVS", "custom_neural_mvs")
        self.combo_method.addItem("Custom Pipeline - SIFT + Stereo", "custom_sift_stereo")
        self.combo_method.addItem("Custom Pipeline - SIFT + MVS", "custom_sift_mvs")
        self.combo_method.addItem("Custom Pipeline - Neural (Sparse Only)", "custom_neural_sparse")
        self.combo_method.addItem("Custom Pipeline - SIFT (Sparse Only)", "custom_sift_sparse")
        
        method_row.addWidget(self.combo_method)
        method_row.addStretch()
        algo_layout.addLayout(method_row)
        
        # === COLMAP Options ===
        self.colmap_options = QWidget()
        colmap_layout = QHBoxLayout(self.colmap_options)
        colmap_layout.setContentsMargins(0, 10, 0, 0)
        
        colmap_layout.addWidget(QLabel("Quality:"))
        self.combo_colmap_quality = QComboBox()
        self.combo_colmap_quality.addItem("Low (Fast)", "low")
        self.combo_colmap_quality.addItem("Medium", "medium")
        self.combo_colmap_quality.addItem("High (Slow)", "high")
        self.combo_colmap_quality.setCurrentIndex(1)  # Default medium
        colmap_layout.addWidget(self.combo_colmap_quality)
        
        colmap_layout.addSpacing(20)
        self.cb_colmap_gpu = QCheckBox("Use GPU")
        self.cb_colmap_gpu.setChecked(True)
        colmap_layout.addWidget(self.cb_colmap_gpu)
        
        colmap_layout.addStretch()
        algo_layout.addWidget(self.colmap_options)
        
        # === Custom Pipeline Options ===
        self.custom_options = QWidget()
        custom_layout = QVBoxLayout(self.custom_options)
        custom_layout.setContentsMargins(0, 10, 0, 0)
        
        # Row 1: Max images
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Max images:"))
        self.spin_max_images = QSpinBox()
        self.spin_max_images.setRange(5, 200)
        self.spin_max_images.setValue(50)
        self.spin_max_images.setSpecialValueText("All")
        row1.addWidget(self.spin_max_images)
        row1.addStretch()

        custom_layout.addLayout(row1)
        
        # Row 2: Advanced options
        row2 = QHBoxLayout()
        self.cb_full_match = QCheckBox("Full Pairwise Matching (slow)")
        self.cb_full_match.setChecked(False)
        row2.addWidget(self.cb_full_match)
        
        row2.addSpacing(20)
        self.cb_check_quality = QCheckBox("Image Quality Check")
        self.cb_check_quality.setChecked(False)
        row2.addWidget(self.cb_check_quality)
        
        row2.addStretch()
        custom_layout.addLayout(row2)
        
        algo_layout.addWidget(self.custom_options)
        
        layout.addWidget(algo_group)
        
        # === Progress Section ===
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(15, 25, 15, 15)
        
        self.label_status = QLabel("Ready")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14px; color: #aaa;")
        progress_layout.addWidget(self.label_status)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        
        # === Run Button ===
        self.btn_run = QPushButton("START RECONSTRUCTION")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.start_reconstruction)
        layout.addWidget(self.btn_run)
        
        return tab
    
    def _on_method_changed(self):
        """Show/hide options based on selected method"""
        method = self.combo_method.currentData()

        is_colmap = method.startswith("colmap")
        self.colmap_options.setVisible(is_colmap)
        self.custom_options.setVisible(not is_colmap)

    def _load_datasets(self):
        """Finds folders in data/samples"""
        root = Path(__file__).parent.parent.parent / "data" / "samples"
        if root.exists():
            folders = []
            for f in root.iterdir():
                if f.is_dir():
                    # Count images
                    images = list(f.glob("*.jpg")) + list(f.glob("*.JPG")) + \
                             list(f.glob("*.png")) + list(f.glob("*.PNG"))
                    folders.append((f.name, len(images)))
            
            for name, count in sorted(folders):
                self.combo_dataset.addItem(f"{name} ({count} images)", name)
        else:
            print("Error: data/samples not found")

    def start_reconstruction(self):
        dataset = self.combo_dataset.currentData()
        method = self.combo_method.currentData()
        method_name = self.combo_method.currentText()

        if not dataset:
            return

        # Build options dict based on method
        options = {
            'dataset': dataset,
            'method': method,
        }

        if method.startswith("colmap"):
            options['quality'] = self.combo_colmap_quality.currentData()
            options['use_gpu'] = self.cb_colmap_gpu.isChecked()
            options['sparse_only'] = (method == "colmap_sparse")
        else:
            max_val = self.spin_max_images.value()
            options['max_images'] = max_val if max_val > 5 else None
            options['neural'] = 'neural' in method
            options['stereo'] = 'stereo' in method
            options['mvs'] = 'mvs' in method
            options['dense'] = 'sift' in method and 'sparse' not in method and 'stereo' not in method and 'mvs' not in method

        # UI Updates
        self.btn_run.setEnabled(False)
        self.btn_run.setText("PROCESSING...")
        self.btn_run.setStyleSheet("""
            QPushButton { 
                background-color: #555; 
                color: #aaa; 
                border: none; 
                border-radius: 5px; 
                font-size: 14px;
                font-weight: bold;
            }
        """)

        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.label_status.setText(f"Running: {method_name}...")

        # Run
        self.worker.run_reconstruction(options)

    def _update_status(self, text):
        self.label_status.setText(text)

    def _reconstruction_finished(self, success):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("START RECONSTRUCTION")
        self.btn_run.setStyleSheet("""
            QPushButton { 
                background-color: #0066cc; 
                color: white; 
                border: none; 
                border-radius: 5px; 
                font-size: 14px; 
                font-weight: bold; 
            }
            QPushButton:hover { background-color: #0077ee; }
            QPushButton:pressed { background-color: #0055aa; }
        """)
        
        self.progress_bar.setRange(0, 100)
        
        if success:
            self.progress_bar.setValue(100)
            self.label_status.setText("Complete")
            
            reply = QMessageBox.question(
                self, "Complete", 
                "Reconstruction finished. Open Viewer?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.tabs.setCurrentIndex(1)
                dataset = self.combo_dataset.currentData()
                base = Path(__file__).parent.parent.parent / "data" / "samples" / dataset
                
                # Try to find result files
                possible_files = [
                    base / "colmap_reconstruction" / "dense" / "fused.ply",
                    base / "colmap_reconstruction" / "fused.ply",
                    base / "colmap_reconstruction" / "sparse.ply",
                    base / "reconstruction" / "dense_stereo.ply",
                    base / "reconstruction" / "dense_mvs.ply",
                    base / "reconstruction" / "sparse.ply",
                    base / "point_cloud.ply",
                ]
                for p in possible_files:
                    if p.exists():
                        self.tab_viewer.load_ply_file(str(p))
                        break
        else:
            self.progress_bar.setValue(0)
            self.label_status.setText("Failed")
            QMessageBox.critical(self, "Error", "Reconstruction failed. Check console for details.")

    def _get_style(self):
        return """
            QMainWindow { background-color: #1e1e1e; color: #f0f0f0; }
            
            QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; }
            QTabBar::tab { 
                background: #2d2d2d; 
                color: #999; 
                padding: 10px 25px; 
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px; 
            }
            QTabBar::tab:selected { 
                background: #3c3c3c; 
                color: white; 
                border-bottom: 2px solid #0066cc; 
            }
            
            QWidget { color: #f0f0f0; font-family: "Segoe UI", Arial, sans-serif; }
            
            QGroupBox { 
                border: 1px solid #3e3e42; 
                border-radius: 5px; 
                margin-top: 20px; 
                font-weight: bold; 
                font-size: 13px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
            }
            
            QComboBox { 
                background: #333; 
                padding: 6px 10px; 
                border: 1px solid #555; 
                border-radius: 4px; 
                min-height: 20px;
                color: #f0f0f0;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #f0f0f0;
                border: 1px solid #555;
                selection-background-color: #0066cc;
            }
            
            QSpinBox {
                background: #333;
                padding: 4px 8px;
                border: 1px solid #555;
                border-radius: 4px;
                color: #f0f0f0;
            }
            
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #555;
                background: #333;
            }
            QCheckBox::indicator:checked {
                background: #0066cc;
                border-color: #0066cc;
            }
            
            QProgressBar { 
                border: none; 
                background: #2d2d2d; 
                border-radius: 4px; 
            }
            QProgressBar::chunk { 
                background-color: #0066cc; 
                border-radius: 4px; 
            }
            
            QPushButton { 
                background-color: #0066cc; 
                color: white; 
                border: none; 
                border-radius: 5px; 
                font-size: 14px; 
                font-weight: bold; 
            }
            QPushButton:hover { background-color: #0077ee; }
            QPushButton:pressed { background-color: #0055aa; }
            
            QLabel { color: #ccc; }
        """


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()