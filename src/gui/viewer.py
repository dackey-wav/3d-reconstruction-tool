"""
PyQt5 3D Point Cloud Viewer

Simple GUI to visualize PLY reconstruction results.
Works with both COLMAP and our pipeline outputs.
"""
import sys
import struct
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
    QLabel, QSlider, QGroupBox, QSplitter, QListWidget, QListWidgetItem, 
    QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt

try:
    import OpenGL.GL as gl_raw
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

try:
    import pyqtgraph.opengl as gl
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


def load_ply(filepath: str) -> tuple:
    """Load PLY file (ASCII or binary), return (points, colors)"""
    
    with open(filepath, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Parse header
        n_vertices = 0
        is_binary = False
        is_little_endian = True
        properties = []
        
        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('format binary_little_endian'):
                is_binary = True
                is_little_endian = True
            elif line.startswith('format binary_big_endian'):
                is_binary = True
                is_little_endian = False
            elif line.startswith('format ascii'):
                is_binary = False
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_type, prop_name))
        
        print(f"PLY: {n_vertices} vertices, binary={is_binary}, properties={len(properties)}")
        
        # Property type sizes
        prop_sizes = {
            'float': ('f', 4), 'float32': ('f', 4),
            'double': ('d', 8), 'float64': ('d', 8),
            'uchar': ('B', 1), 'uint8': ('B', 1),
            'char': ('b', 1), 'int8': ('b', 1),
            'ushort': ('H', 2), 'uint16': ('H', 2),
            'short': ('h', 2), 'int16': ('h', 2),
            'uint': ('I', 4), 'uint32': ('I', 4),
            'int': ('i', 4), 'int32': ('i', 4),
        }
        
        # Build format string and find property indices
        endian = '<' if is_little_endian else '>'
        fmt_parts = []
        prop_indices = {}
        
        for i, (prop_type, prop_name) in enumerate(properties):
            if prop_type in prop_sizes:
                fmt_char, _ = prop_sizes[prop_type]
                fmt_parts.append(fmt_char)
                prop_indices[prop_name] = i
        
        fmt = endian + ''.join(fmt_parts)
        vertex_size = struct.calcsize(fmt) if fmt_parts else 0
        
        # Find coordinate and color indices
        x_idx = prop_indices.get('x', 0)
        y_idx = prop_indices.get('y', 1)
        z_idx = prop_indices.get('z', 2)
        r_idx = prop_indices.get('red', None)
        g_idx = prop_indices.get('green', None)
        b_idx = prop_indices.get('blue', None)
        has_color = r_idx is not None
        
        points = []
        colors = []
        
        if is_binary and vertex_size > 0:
            # Binary format
            data = f.read(vertex_size * n_vertices)
            
            for i in range(n_vertices):
                offset = i * vertex_size
                if offset + vertex_size > len(data):
                    break
                try:
                    vertex = struct.unpack(fmt, data[offset:offset + vertex_size])
                    x, y, z = vertex[x_idx], vertex[y_idx], vertex[z_idx]
                    
                    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                        continue
                    
                    points.append([x, y, z])
                    
                    if has_color:
                        r = int(vertex[r_idx])
                        g = int(vertex[g_idx])
                        b = int(vertex[b_idx])
                        colors.append([r, g, b, 255])
                    else:
                        colors.append([180, 180, 180, 255])
                        
                except struct.error:
                    continue
                    
                if i % 100000 == 0 and i > 0:
                    print(f"  Loaded {i:,} / {n_vertices:,} points...")
        else:
            # ASCII format
            for i in range(n_vertices):
                line = f.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split()
                
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                        
                        if len(parts) >= 6:
                            r = int(float(parts[3]))
                            g = int(float(parts[4]))
                            b = int(float(parts[5]))
                            colors.append([r, g, b, 255])
                        else:
                            colors.append([180, 180, 180, 255])
                    except (ValueError, IndexError):
                        continue
    
    print(f"  Loaded {len(points):,} points total")
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


class ViewerTab(QWidget):
    """3D Viewer Widget"""
    
    def __init__(self):
        super().__init__()
        self.points = None
        self.colors = None
        self.scatter = None
        
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # --- Left Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(280)
        
        # Files
        file_group = QGroupBox("Assets")
        file_layout = QVBoxLayout(file_group)
        
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open File")
        self.btn_open.clicked.connect(self.open_file)
        self.btn_folder = QPushButton("Scan Folder")
        self.btn_folder.clicked.connect(self.open_folder)
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_folder)
        file_layout.addLayout(btn_layout)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list)
        left_layout.addWidget(file_group)
        
        # Visuals
        view_group = QGroupBox("Rendering")
        view_layout = QVBoxLayout(view_group)
        
        view_layout.addWidget(QLabel("Point Size:"))
        self.slider_size = QSlider(Qt.Horizontal)
        self.slider_size.setRange(1, 15)
        self.slider_size.setValue(2)
        self.slider_size.valueChanged.connect(self.update_visuals)
        view_layout.addWidget(self.slider_size)
        
        view_layout.addWidget(QLabel("Density:"))
        self.slider_density = QSlider(Qt.Horizontal)
        self.slider_density.setRange(1, 100)
        self.slider_density.setValue(100)
        self.slider_density.valueChanged.connect(self.update_visuals)
        view_layout.addWidget(self.slider_density)

        self.cb_smooth = QCheckBox("High Quality Points")
        self.cb_smooth.setChecked(True)
        self.cb_smooth.toggled.connect(self.update_visuals)
        view_layout.addWidget(self.cb_smooth)
        
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_view)
        view_layout.addWidget(self.btn_reset)
        
        left_layout.addWidget(view_group)
        
        self.label_info = QLabel("No geometry loaded")
        self.label_info.setStyleSheet("color: #888;")
        self.label_info.setWordWrap(True)
        left_layout.addWidget(self.label_info)
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        # --- Right 3D View ---
        if HAS_PYQTGRAPH:
            self.view3d = gl.GLViewWidget()
            self.view3d.setCameraPosition(distance=25)
            self.view3d.setBackgroundColor(30, 30, 30)
            
            grid = gl.GLGridItem()
            grid.scale(2, 2, 1)
            grid.setColor((80, 80, 80, 60))
            self.view3d.addItem(grid)
            
            splitter.addWidget(self.view3d)
        else:
            splitter.addWidget(QLabel("Error: pyqtgraph not installed"))
            
        splitter.setSizes([280, 920])

    def _apply_styles(self):
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 20px;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            
            QListWidget {
                background-color: #252526;
                color: #f0f0f0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
            }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background-color: #0078d4; color: white; }
            QListWidget::item:hover { background-color: #3e3e42; }
            
            QPushButton {
                background-color: #3e3e42;
                color: #f0f0f0;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #4e4e52; }
        """)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PLY", "", "PLY (*.ply)")
        if path:
            self.load_ply_file(path)

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            p = Path(d)
            files = sorted(list(p.glob("**/*.ply")))
            self.file_list.clear()
            for f in files:
                item = QListWidgetItem(f.name)
                item.setData(Qt.UserRole, str(f))
                self.file_list.addItem(item)

    def on_file_selected(self, item):
        self.load_ply_file(item.data(Qt.UserRole))

    def load_ply_file(self, path):
        if not HAS_PYQTGRAPH:
            return
            
        try:
            self.points, self.colors = load_ply(path)
        except Exception as e:
            print(f"Error loading PLY: {e}")
            return

        if len(self.points) == 0:
            return
        
        # Auto-adjust density for large files
        if len(self.points) > 1000000:
            self.slider_density.setValue(30)
        elif len(self.points) > 500000:
            self.slider_density.setValue(50)
        
        self.label_info.setText(f"{Path(path).name}\n{len(self.points):,} vertices")
        self.update_visuals()
        self.reset_view()

    def update_visuals(self):
        if self.points is None or len(self.points) == 0:
            return
            
        if not HAS_PYQTGRAPH:
            return
            
        # Remove old scatter
        if self.scatter is not None:
            try:
                self.view3d.removeItem(self.scatter)
            except:
                pass
            self.scatter = None
        
        # Subsample based on density slider
        density = self.slider_density.value() / 100.0
        n_display = max(100, int(len(self.points) * density))
        
        if n_display < len(self.points):
            indices = np.random.choice(len(self.points), n_display, replace=False)
            pts = self.points[indices]
            cols = self.colors[indices]
        else:
            pts = self.points
            cols = self.colors
        
        # Center points
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        # Normalize colors to 0-1
        cols_float = cols.astype(np.float32) / 255.0
        
        # Create scatter plot
        self.scatter = gl.GLScatterPlotItem(
            pos=pts_centered,
            color=cols_float,
            size=self.slider_size.value(),
            pxMode=True
        )
        
        self.scatter.setGLOptions('translucent')
        
        # Enable point smoothing if available
        if HAS_OPENGL and self.cb_smooth.isChecked():
            try:
                gl_raw.glEnable(gl_raw.GL_POINT_SMOOTH)
            except:
                pass
        
        self.view3d.addItem(self.scatter)

    def reset_view(self):
        if HAS_PYQTGRAPH:
            self.view3d.setCameraPosition(distance=30, elevation=30, azimuth=45)