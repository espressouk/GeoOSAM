import sys
import os
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module, compose
from shapely.geometry import shape
from rasterio.features import shapes
import rasterio
import cv2
import numpy as np
import torch
import datetime
import pathlib
import platform
import subprocess
import urllib.request
from qgis.PyQt.QtCore import QVariant, Qt, QThread, pyqtSignal
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsWkbTypes,
    QgsPointXY,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsFillSymbol,
    QgsField,
    QgsVectorFileWriter
)
from qgis.gui import QgsRubberBand, QgsMapTool
from qgis.PyQt import QtWidgets, QtCore, QtGui

# fmt: off
plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(plugin_dir)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Ultralytics MobileSAM setup
MOBILESAM_AVAILABLE = False

try:
    from ultralytics import SAM
    test_model = SAM('mobile_sam.pt') # sam2.1_b.pt
    MOBILESAM_AVAILABLE = True
    print("✅ Ultralytics MobileSAM available")

    class UltralyticsPredictor:
        def __init__(self, model):
            self.model = model
            self.features = None

        def set_image(self, image):
            self.image = image

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
            try:
                if point_coords is not None:
                    if len(point_coords) > 0:
                        point = point_coords[0]
                        x, y = int(point[0]), int(point[1])
                        results = self.model.predict(
                            source=self.image,
                            points=[[x, y]],
                            labels=[1],
                            verbose=False
                        )
                    else:
                        return self._empty_result()

                elif box is not None:
                    if len(box) > 0:
                        bbox = box[0]
                        x1, y1, x2, y2 = int(bbox[0]), int(
                            bbox[1]), int(bbox[2]), int(bbox[3])
                        results = self.model.predict(
                            source=self.image,
                            bboxes=[[x1, y1, x2, y2]],
                            verbose=False
                        )
                    else:
                        return self._empty_result()
                else:
                    results = self.model.predict(
                        source=self.image, verbose=False)

                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks_tensor = result.masks.data
                        if len(masks_tensor) > 0:
                            mask_tensor = masks_tensor[0]
                            if hasattr(mask_tensor, 'cpu'):
                                mask = mask_tensor.cpu().numpy()
                            else:
                                mask = mask_tensor.numpy()

                            if mask.max() <= 1.0:
                                mask = (mask * 255).astype(np.uint8)
                            else:
                                mask = mask.astype(np.uint8)

                            num_pixels = np.sum(mask > 0)
                            if num_pixels > 0:
                                return [mask], [1.0], None
                            else:
                                return self._empty_result()

                return self._empty_result()

            except Exception as e:
                print(f"MobileSAM prediction error: {e}")
                return self._empty_result()

        def _empty_result(self):
            empty_mask = np.zeros(
                (self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            return [empty_mask], [0.0], None

except ImportError:
    print("⚠️ Ultralytics not available - install with: /usr/bin/python3 -m pip install --user ultralytics")
    MOBILESAM_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Ultralytics MobileSAM failed: {e}")
    MOBILESAM_AVAILABLE = False

if MOBILESAM_AVAILABLE:
    print("   Using fast Ultralytics MobileSAM")
else:
    print("   Falling back to SAM2")

"""
GeoOSAM Control Panel - Enhanced SAM segmentation for QGIS
Copyright (C) 2025 by Ofer Butbega
"""

# Global threading configuration
_THREADS_CONFIGURED = False


def setup_pytorch_performance():
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED:
        return torch.get_num_threads()

    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    optimal_threads = max(4, int(num_cores * 0.75)) if num_cores >= 16 else \
        max(4, num_cores - 2) if num_cores >= 8 else \
        max(1, num_cores - 1)

    torch.set_num_interop_threads(min(4, optimal_threads // 2))
    torch.set_num_threads(optimal_threads)

    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)

    _THREADS_CONFIGURED = True
    return optimal_threads


def auto_download_checkpoint():
    """Download SAM2 checkpoint if missing"""
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(plugin_dir, "sam2", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "sam2.1_hiera_tiny.pt")
    download_script = os.path.join(
        checkpoint_dir, "download_sam2_checkpoints.sh")

    if os.path.exists(checkpoint_path):
        print(f"✅ SAM2 checkpoint found")
        return True

    print(f"🔍 SAM2 checkpoint not found, downloading...")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Try bash script first (Linux/Mac)
    if platform.system() in ['Linux', 'Darwin'] and os.path.exists(download_script):
        try:
            result = subprocess.run(['bash', download_script], cwd=checkpoint_dir,
                                    capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(checkpoint_path):
                print("✅ Checkpoint downloaded via script")
                return True
        except Exception as e:
            print(f"⚠️ Script failed: {e}")

    # Python fallback
    try:
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt"
        urllib.request.urlretrieve(url, checkpoint_path)
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1000000:
            print("✅ Checkpoint downloaded via Python")
            return True
        else:
            print("❌ Download verification failed")
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def show_checkpoint_dialog(parent=None):
    """Show download dialog for SAM2 checkpoint"""
    from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog
    from qgis.PyQt.QtCore import Qt

    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle("SAM2 Model Download")
    msg.setText("GeoOSAM requires the SAM2 model checkpoint (~160MB).")
    msg.setInformativeText("Would you like to download it now?")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg.setDefaultButton(QMessageBox.Yes)

    if msg.exec_() == QMessageBox.Yes:
        progress = QProgressDialog(
            "Downloading SAM2 model...", "Cancel", 0, 0, parent)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            success = auto_download_checkpoint()
            progress.close()
            if success:
                QMessageBox.information(
                    parent, "Success", "✅ SAM2 model downloaded successfully!")
                return True
            else:
                QMessageBox.critical(
                    parent, "Download Failed", "❌ Failed to download SAM2 model.")
                return False
        except Exception as e:
            progress.close()
            QMessageBox.critical(parent, "Error", f"Download error: {e}")
            return False
    return False


def detect_best_device():
    """Detect best available device and model"""
    cores = None
    try:
        if torch.cuda.is_available() and not os.getenv("GEOOSAM_FORCE_CPU"):
            gpu_props = torch.cuda.get_device_properties(0)
            if gpu_props.total_memory / 1024**3 >= 4:  # 4GB minimum
                device = "cuda"
                model_choice = "SAM2"
                print(
                    f"🎮 GPU detected: {torch.cuda.get_device_name(0)} - using SAM2")
                return device, model_choice, cores

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device, model_choice = "mps", "SAM2"
            print("🍎 Apple Silicon GPU detected - using SAM2")
            return device, model_choice, cores

        else:
            device = "cpu"
            model_choice = "MobileSAM" if MOBILESAM_AVAILABLE else "SAM2"
            cores = setup_pytorch_performance()
            print(f"💻 CPU detected - using {model_choice} ({cores} cores)")
            return device, model_choice, cores

    except Exception as e:
        print(f"⚠️ Device detection failed: {e}, falling back to CPU")
        device, model_choice = "cpu", "SAM2"
        cores = setup_pytorch_performance()
        return device, model_choice, cores


class OptimizedSAM2Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, predictor, arr, mode, model_choice="SAM2", point_coords=None,
                 point_labels=None, box=None, mask_transform=None, debug_info=None, device="cpu"):
        super().__init__()
        self.predictor = predictor
        self.arr = arr
        self.mode = mode
        self.model_choice = model_choice
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.box = box
        self.mask_transform = mask_transform
        self.debug_info = debug_info or {}
        self.device = device

    def run(self):
        try:
            self.progress.emit(f"🖼️ Setting image for {self.model_choice}...")
            self.predictor.set_image(self.arr)

            self.progress.emit(f"🧠 Running {self.model_choice} inference...")

            with torch.no_grad():
                if self.mode == "point":
                    masks, scores, logits = self.predictor.predict(
                        point_coords=self.point_coords,
                        point_labels=self.point_labels,
                        multimask_output=False
                    )
                elif self.mode == "bbox":
                    masks, scores, logits = self.predictor.predict(
                        box=self.box,
                        multimask_output=False
                    )
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

            self.progress.emit("⚡ Processing mask...")

            mask = masks[0]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            elif torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()

            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            result = {
                'mask': mask,
                'scores': scores,
                'logits': logits,
                'mask_transform': self.mask_transform,
                'debug_info': {**self.debug_info, 'model': self.model_choice}
            }

            self.finished.emit(result)

        except Exception as e:
            import traceback
            error_msg = f"{self.model_choice} inference failed: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class EnhancedPointClickTool(QgsMapTool):
    def __init__(self, canvas, cb):
        super().__init__(canvas)
        self.canvas = canvas
        self.cb = cb
        self.setCursor(QtCore.Qt.CrossCursor)

        self.point_rubber = QgsRubberBand(canvas, QgsWkbTypes.PointGeometry)
        self.point_rubber.setColor(QtCore.Qt.red)
        self.point_rubber.setIcon(QgsRubberBand.ICON_CIRCLE)
        self.point_rubber.setIconSize(12)
        self.point_rubber.setWidth(4)

    def canvasReleaseEvent(self, e):
        map_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        self.point_rubber.addPoint(map_point, True)
        self.canvas.refresh()
        self.cb(map_point)

    def deactivate(self):
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        super().deactivate()

    def clear_feedback(self):
        self.point_rubber.reset(QgsWkbTypes.PointGeometry)
        self.canvas.refresh()


class EnhancedBBoxClickTool(QgsMapTool):
    def __init__(self, canvas, cb):
        super().__init__(canvas)
        self.canvas = canvas
        self.cb = cb
        self.setCursor(QtCore.Qt.CrossCursor)
        self.start_point = None
        self.is_dragging = False

        self.bbox_rubber = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.bbox_rubber.setColor(QtCore.Qt.blue)
        self.bbox_rubber.setFillColor(QtGui.QColor(0, 0, 255, 60))
        self.bbox_rubber.setWidth(2)

    def canvasPressEvent(self, e):
        self.start_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
        self.is_dragging = True
        self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)

    def canvasMoveEvent(self, e):
        if self.is_dragging and self.start_point:
            current_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
            rect = QgsRectangle(self.start_point, current_point)
            self.bbox_rubber.setToGeometry(QgsGeometry.fromRect(rect), None)
            self.canvas.refresh()

    def canvasReleaseEvent(self, e):
        if self.is_dragging and self.start_point:
            end_point = self.canvas.getCoordinateTransform().toMapCoordinates(e.pos())
            rect = QgsRectangle(self.start_point, end_point)
            if rect.width() > 10 and rect.height() > 10:
                self.cb(rect)
            else:
                self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)
                self.canvas.refresh()
        self.is_dragging = False
        self.start_point = None

    def deactivate(self):
        self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)
        self.is_dragging = False
        self.start_point = None
        super().deactivate()

    def clear_feedback(self):
        self.bbox_rubber.reset(QgsWkbTypes.PolygonGeometry)
        self.canvas.refresh()


class GeoOSAMControlPanel(QtWidgets.QDockWidget):
    """Enhanced SAM segmentation control panel for QGIS"""

    DEFAULT_CLASSES = {
        'Agriculture' : {'color': '255,215,0',   'description': 'Farmland and crops'},
        'Buildings'   : {'color': '220,20,60',   'description': 'Residential & commercial structures'},
        'Commercial'  : {'color': '135,206,250', 'description': 'Shopping and business districts'},
        'Industrial'  : {'color': '128,0,128',   'description': 'Factories and warehouses'},
        'Other'       : {'color': '148,0,211',   'description': 'Unclassified objects'},
        'Parking'     : {'color': '255,140,0',   'description': 'Parking lots and areas'},
        'Residential' : {'color': '255,105,180', 'description': 'Housing areas'},
        'Roads'       : {'color': '105,105,105', 'description': 'Streets, highways, and pathways'},
        'Vessels'     : {'color': '0,206,209',   'description': 'Boats, ship'},
        'Vehicle'     : {'color': '255,69,0',    'description': 'Cars, trucks, and buses'},
        'Vegetation'  : {'color': '34,139,34',   'description': 'Trees, grass, and parks'},
        'Water'       : {'color': '30,144,255',  'description': 'Rivers, lakes, and ponds'}
    }


    EXTRA_COLORS = [
        '50,205,50', '255,20,147', '255,165,0', '186,85,211', '0,128,128',
        '255,192,203', '165,42,42', '0,250,154', '255,0,255', '127,255,212'
    ]

    def __init__(self, iface, parent=None):
        super().__init__("", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()

        # Initialize device and model
        self.device, self.model_choice, self.num_cores = detect_best_device()
        self._init_sam_model()

        # Setup docking
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFloatable)

        # State variables
        self.point = None
        self.bbox = None
        self.current_mode = None
        self.result_layers = {}
        self.segment_counts = {}
        self.current_class = None
        self.classes = self.DEFAULT_CLASSES.copy()
        self.worker = None
        self.original_raster_layer = None
        self.keep_raster_selected = True

        # Output management
        self.shapefile_save_dir = None
        self.mask_save_dir = None
        self.save_debug_masks = False

        # Undo functionality
        self.undo_stack = []

        # Initialize
        self._init_save_directories()
        self.pointTool = EnhancedPointClickTool(self.canvas, self._point_done)
        self.bboxTool = EnhancedBBoxClickTool(self.canvas, self._bbox_done)
        self.original_map_tool = None
        self._setup_ui()

    def _init_sam_model(self):
        """Initialize the selected SAM model"""
        plugin_dir = os.path.dirname(os.path.abspath(__file__))

        if self.model_choice == "MobileSAM":
            self._init_mobilesam_model()
        else:
            self._init_sam2_model(plugin_dir)

    def _init_sam2_model(self, plugin_dir):
        """Initialize SAM2 model"""
        checkpoint_path = os.path.join(
            plugin_dir, "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")

        if not os.path.exists(checkpoint_path):
            if not auto_download_checkpoint():
                if not show_checkpoint_dialog(self):
                    raise Exception(
                        "SAM2 checkpoint required but not available")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        try:
            with initialize_config_module(config_module="sam2.configs"):
                sam_model = build_sam2(
                    "sam2.1/sam2.1_hiera_t", checkpoint_path, device=self.device)

                if self.device == "cuda":
                    sam_model = sam_model.cuda()

                sam_model.eval()
                if self.device == "cpu":
                    try:
                        sam_model = torch.jit.optimize_for_inference(sam_model)
                    except:
                        pass

                self.predictor = SAM2ImagePredictor(sam_model)
                print(f"✅ SAM2 model loaded on {self.device}")

        except Exception as e:
            print(f"❌ Failed to load SAM2: {e}")
            raise

    def _init_mobilesam_model(self):
        """Initialize Ultralytics MobileSAM model"""
        try:
            from ultralytics import SAM
            mobile_sam = SAM('mobile_sam.pt')
            self.predictor = UltralyticsPredictor(mobile_sam)
            print(f"✅ Ultralytics MobileSAM loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load MobileSAM: {e}, falling back to SAM2")
            self.model_choice = "SAM2"
            self._init_sam2_model(os.path.dirname(os.path.abspath(__file__)))

    def _init_save_directories(self):
        """Initialize output directories"""
        self.shapefile_save_dir = pathlib.Path.home() / "GeoOSAM_shapefiles"
        self.mask_save_dir = pathlib.Path.home() / "GeoOSAM_masks"
        self.shapefile_save_dir.mkdir(exist_ok=True)

    def _setup_ui(self):
        """Setup the control panel UI"""
        main_widget = QtWidgets.QWidget()
        self.setWidget(main_widget)
        layout = QtWidgets.QVBoxLayout(main_widget)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title and performance info
        device_icon = "🎮" if "cuda" in self.device else "🍎" if "mps" in self.device else "💻"
        title = QtWidgets.QLabel("GeoOSAM Control Panel")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #2E86AB; padding: 5px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        perf_info = f"{device_icon} {self.device.upper()} ({self.model_choice})"
        if self.num_cores:
            perf_info += f" ({self.num_cores} cores)"

        perf_label = QtWidgets.QLabel(perf_info)
        perf_label.setStyleSheet(
            "font-size: 15px; color: #666; text-align: center; padding: 2px;")
        perf_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(perf_label)

        # Setup sections
        self._add_separator(layout)
        self._setup_output_settings_section(layout)
        self._add_separator(layout)
        self._setup_class_section(layout)
        self._add_separator(layout)
        self._setup_mode_section(layout)
        self._add_separator(layout)
        self._setup_status_section(layout)
        self._add_separator(layout)
        self._setup_control_section(layout)

        layout.addStretch()
        self.setMinimumWidth(300)
        self.setMaximumWidth(380)

    def _add_separator(self, layout):
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(separator)

    def _setup_output_settings_section(self, layout):
        output_header = QtWidgets.QLabel("📁 Output Settings")
        output_header.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #2E86AB;")
        layout.addWidget(output_header)

        folder_layout = QtWidgets.QHBoxLayout()
        self.outputFolderLabel = QtWidgets.QLabel("📂 Default folders")
        self.outputFolderLabel.setStyleSheet("font-size: 15px; color: #666;")
        folder_layout.addWidget(self.outputFolderLabel)

        self.selectFolderBtn = QtWidgets.QPushButton("📁 Choose")
        self.selectFolderBtn.clicked.connect(self._select_output_folder)
        self.selectFolderBtn.setStyleSheet("font-size: 9px; padding: 2px 6px;")
        folder_layout.addWidget(self.selectFolderBtn)
        layout.addLayout(folder_layout)

        debug_layout = QtWidgets.QHBoxLayout()
        self.saveDebugCheckbox = QtWidgets.QCheckBox("💾 Save debug masks")
        self.saveDebugCheckbox.setChecked(False)
        self.saveDebugCheckbox.setStyleSheet("font-size: 15px;")
        self.saveDebugCheckbox.toggled.connect(self._on_debug_toggle)
        debug_layout.addWidget(self.saveDebugCheckbox)
        layout.addLayout(debug_layout)

    def _setup_class_section(self, layout):
        class_header = QtWidgets.QLabel("📋 Class Selection")
        class_header.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #2E86AB;")
        layout.addWidget(class_header)

        self.classComboBox = QtWidgets.QComboBox()
        self.classComboBox.addItem("-- Select Class --", None)
        for class_name, class_info in self.classes.items():
            self.classComboBox.addItem(f"{class_name}", class_name)
        self.classComboBox.currentTextChanged.connect(self._on_class_changed)
        self.classComboBox.setStyleSheet("padding: 4px; font-size: 11px;")
        layout.addWidget(self.classComboBox)

        self.currentClassLabel = QtWidgets.QLabel("Current Class: None")
        self.currentClassLabel.setStyleSheet(
            "font-weight: bold; padding: 8px; border: 2px solid gray; "
            "background-color: rgba(200,200,200,50); border-radius: 4px; font-size: 11px;")
        self.currentClassLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.currentClassLabel)

        class_btn_layout = QtWidgets.QHBoxLayout()
        self.addClassBtn = QtWidgets.QPushButton("➕ Add")
        self.addClassBtn.clicked.connect(self._add_new_class)
        self.addClassBtn.setStyleSheet("font-size: 15px; padding: 4px;")
        class_btn_layout.addWidget(self.addClassBtn)

        self.editClassBtn = QtWidgets.QPushButton("✏️ Edit")
        self.editClassBtn.clicked.connect(self._edit_classes)
        self.editClassBtn.setStyleSheet("font-size: 15px; padding: 4px;")
        class_btn_layout.addWidget(self.editClassBtn)
        layout.addLayout(class_btn_layout)

    def _setup_mode_section(self, layout):
        mode_header = QtWidgets.QLabel("🎯 Segmentation Mode")
        mode_header.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #2E86AB;")
        layout.addWidget(mode_header)

        mode_layout = QtWidgets.QVBoxLayout()
        self.pointModeBtn = QtWidgets.QPushButton("🎯 Point Mode")
        self.pointModeBtn.clicked.connect(self._activate_point_tool)
        self.pointModeBtn.setStyleSheet(
            "padding: 8px; font-size: 17px; font-weight: bold; "
            "background-color: #4CAF50; color: white; border-radius: 4px;")
        mode_layout.addWidget(self.pointModeBtn)

        self.bboxModeBtn = QtWidgets.QPushButton("📦 BBox Mode")
        self.bboxModeBtn.clicked.connect(self._activate_bbox_tool)
        self.bboxModeBtn.setStyleSheet(
            "padding: 8px; font-size: 17px; font-weight: bold; "
            "background-color: #2196F3; color: white; border-radius: 4px;")
        self.bboxModeBtn.setVisible(False)  # Hidden for now
        mode_layout.addWidget(self.bboxModeBtn)

        layout.addLayout(mode_layout)

        self.currentModeLabel = QtWidgets.QLabel("Mode: None")
        self.currentModeLabel.setStyleSheet(
            "padding: 4px; background-color: #f0f0f0; border-radius: 3px; font-size: 15px;")
        self.currentModeLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.currentModeLabel)

    def _setup_status_section(self, layout):
        status_header = QtWidgets.QLabel("📊 Status")
        status_header.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #2E86AB;")
        layout.addWidget(status_header)

        self.statusLabel = QtWidgets.QLabel("Ready to segment")
        self.statusLabel.setStyleSheet(
            "padding: 6px; background-color: #e8f5e8; border-left: 4px solid #4CAF50; "
            "font-size: 15px; border-radius: 3px;")
        self.statusLabel.setWordWrap(True)
        layout.addWidget(self.statusLabel)

        self.statsLabel = QtWidgets.QLabel("Total Segments: 0")
        self.statsLabel.setStyleSheet(
            "font-size: 15px; color: #666; padding: 2px;")
        layout.addWidget(self.statsLabel)

    def _setup_control_section(self, layout):
        control_header = QtWidgets.QLabel("⚙️ Controls")
        control_header.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #2E86AB;")
        layout.addWidget(control_header)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.setVisible(False)
        self.progressBar.setStyleSheet(
            "QProgressBar { border: 1px solid #ccc; border-radius: 3px; }")
        layout.addWidget(self.progressBar)

        control_layout = QtWidgets.QVBoxLayout()

        self.cancelBtn = QtWidgets.QPushButton("❌ Cancel Processing")
        self.cancelBtn.clicked.connect(self._cancel_segmentation)
        self.cancelBtn.setVisible(False)
        self.cancelBtn.setStyleSheet(
            "padding: 6px; font-size: 15px; background-color: #f44336; color: white; border-radius: 3px;")
        control_layout.addWidget(self.cancelBtn)

        self.undoBtn = QtWidgets.QPushButton("↶ Undo Last Polygon")
        self.undoBtn.clicked.connect(self._undo_last_polygon)
        self.undoBtn.setEnabled(False)
        self.undoBtn.setStyleSheet(
            "padding: 6px; font-size: 15px; background-color: #FF5722; color: white; border-radius: 3px;")
        control_layout.addWidget(self.undoBtn)

        self.exportBtn = QtWidgets.QPushButton("💾 Export All")
        self.exportBtn.clicked.connect(self._export_all_classes)
        self.exportBtn.setStyleSheet(
            "padding: 6px; font-size: 15px; background-color: #9C27B0; color: white; border-radius: 3px;")
        control_layout.addWidget(self.exportBtn)

        layout.addLayout(control_layout)

    def _select_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Shapefiles", str(self.shapefile_save_dir.parent))

        if folder:
            self.shapefile_save_dir = pathlib.Path(
                folder) / "GeoOSAM_shapefiles"
            self.mask_save_dir = pathlib.Path(folder) / "GeoOSAM_masks"
            self.shapefile_save_dir.mkdir(exist_ok=True)
            if self.save_debug_masks:
                self.mask_save_dir.mkdir(exist_ok=True)

            short_path = "..." + str(self.shapefile_save_dir)[-30:] if len(
                str(self.shapefile_save_dir)) > 35 else str(self.shapefile_save_dir)
            self.outputFolderLabel.setText(f"📂 {short_path}")
            self._update_status(
                f"📁 Output folder: {self.shapefile_save_dir}", "info")

    def _on_debug_toggle(self, checked):
        self.save_debug_masks = checked
        if checked:
            self.mask_save_dir.mkdir(exist_ok=True)
            self._update_status("💾 Debug masks will be saved", "info")
        else:
            self._update_status("🚫 Debug masks disabled", "info")

    def _on_class_changed(self):
        selected_data = self.classComboBox.currentData()
        if selected_data:
            self.current_class = selected_data
            class_info = self.classes[selected_data]
            self.currentClassLabel.setText(f"Current: {selected_data}")

            color = class_info['color']
            try:
                r, g, b = [int(c.strip()) for c in color.split(',')]
                self.currentClassLabel.setStyleSheet(
                    f"font-weight: bold; padding: 8px; margin: 2px; "
                    f"border: 3px solid rgb({r},{g},{b}); "
                    f"background-color: rgba({r},{g},{b}, 30); "
                    f"color: rgb({max(0, r-50)},{max(0, g-50)},{max(0, b-50)}); "
                    f"border-radius: 4px; font-size: 11px;")
            except:
                self.currentClassLabel.setStyleSheet(
                    f"font-weight: bold; padding: 8px; border: 2px solid rgb({color}); "
                    f"background-color: rgba({color}, 50); font-size: 11px;")

            self._activate_point_tool()
        else:
            self.current_class = None
            self.currentClassLabel.setText("Current: None")
            self.currentClassLabel.setStyleSheet(
                "font-weight: bold; padding: 8px; border: 2px solid gray; "
                "background-color: rgba(200,200,200,50); border-radius: 4px; font-size: 11px;")

    def _add_new_class(self):
        class_name, ok = QtWidgets.QInputDialog.getText(
            self, 'Add Class', 'Enter class name:')
        if ok and class_name and class_name not in self.classes:
            used_colors = [info['color'] for info in self.classes.values()]
            available_colors = [
                c for c in self.EXTRA_COLORS if c not in used_colors]

            if available_colors:
                color = available_colors[0]
            else:
                import random
                color = f"{random.randint(100,255)},{random.randint(100,255)},{random.randint(100,255)}"

            description = f'Custom class: {class_name}'
            self.classes[class_name] = {
                'color': color, 'description': description}
            self.classComboBox.addItem(class_name, class_name)
            self._update_status(
                f"Added class: {class_name} (RGB:{color})", "info")

    def _edit_classes(self):
        class_list = list(self.classes.keys())
        if not class_list:
            self._update_status("No classes to edit", "warning")
            return

        class_name, ok = QtWidgets.QInputDialog.getItem(
            self, 'Edit Classes', 'Select class to edit:', class_list, 0, False)

        if ok and class_name:
            current_info = self.classes[class_name]
            new_name, ok2 = QtWidgets.QInputDialog.getText(
                self, 'Edit Class Name', f'Edit name for {class_name}:', text=class_name)

            if ok2 and new_name:
                current_color = current_info['color']
                new_color, ok3 = QtWidgets.QInputDialog.getText(
                    self, 'Edit Color', f'Edit color for {new_name} (R,G,B):', text=current_color)

                if ok3 and new_color:
                    try:
                        parts = [int(p.strip()) for p in new_color.split(',')]
                        if len(parts) == 3 and all(0 <= p <= 255 for p in parts):
                            if new_name != class_name:
                                del self.classes[class_name]

                            self.classes[new_name] = {
                                'color': new_color,
                                'description': current_info.get('description', f'Class: {new_name}')
                            }
                            self._refresh_class_combo()
                            self._update_status(
                                f"Updated {new_name} with RGB({new_color})", "info")
                        else:
                            self._update_status(
                                "Invalid color format! Use R,G,B (0-255)", "error")
                    except ValueError:
                        self._update_status(
                            "Invalid color format! Use R,G,B (0-255)", "error")

    def _refresh_class_combo(self):
        current_class = self.current_class
        self.classComboBox.clear()
        self.classComboBox.addItem("-- Select Class --", None)

        for class_name, class_info in self.classes.items():
            self.classComboBox.addItem(class_name, class_name)

        if current_class and current_class in self.classes:
            index = self.classComboBox.findData(current_class)
            if index >= 0:
                self.classComboBox.setCurrentIndex(index)

    def _validate_class_selection(self):
        if not self.current_class:
            self._update_status("Please select a class first!", "warning")
            return False

        current_layer = self.iface.activeLayer()
        if not isinstance(current_layer, QgsRasterLayer):
            raster_layers = [layer for layer in QgsProject.instance().mapLayers().values()
                             if isinstance(layer, QgsRasterLayer)]
            if raster_layers:
                self.original_raster_layer = raster_layers[0]
                self.iface.setActiveLayer(self.original_raster_layer)
            else:
                self._update_status(
                    "Please select a raster layer first!", "warning")
                return False
        return True

    def _activate_point_tool(self):
        if not self._validate_class_selection():
            return

        self.current_mode = 'point'
        self.original_map_tool = self.canvas.mapTool()

        self.currentModeLabel.setText("Mode: 🎯 Point Active")
        self.currentModeLabel.setStyleSheet(
            "padding: 4px; background-color: #4CAF50; color: white; border-radius: 3px; font-size: 15px;")

        self._update_status(
            f"Point mode active for [{self.current_class}]. Click on map to segment.", "processing")
        self.canvas.setMapTool(self.pointTool)

    def _activate_bbox_tool(self):
        if not self._validate_class_selection():
            return

        self.current_mode = 'bbox'
        self.original_map_tool = self.canvas.mapTool()

        self.currentModeLabel.setText("Mode: 📦 BBox Active")
        self.currentModeLabel.setStyleSheet(
            "padding: 4px; background-color: #2196F3; color: white; border-radius: 3px; font-size: 15px;")

        self._update_status(
            f"BBox mode active for [{self.current_class}]. Click and drag to segment.", "processing")
        self.canvas.setMapTool(self.bboxTool)

    def _point_done(self, pt):
        self.point = pt
        self.bbox = None
        self._update_status(
            f"Processing point at ({pt.x():.1f}, {pt.y():.1f})...", "processing")
        self._run_segmentation()

    def _bbox_done(self, rect):
        self.bbox = rect
        self.point = None
        self._update_status(
            f"Processing bbox ({rect.width():.1f}×{rect.height():.1f})...", "processing")
        self._run_segmentation()

    def _run_segmentation(self):
        if not self.current_class:
            self._update_status("No class selected", "error")
            return

        rlayer = self.iface.activeLayer()
        if not isinstance(rlayer, QgsRasterLayer):
            self._update_status("Select a raster layer first", "error")
            return

        if self.point is None and self.bbox is None:
            self._update_status("No selection found", "error")
            return

        import time
        start_time = time.time()

        self._set_ui_enabled(False)
        self._update_status(
            "🚀 Preparing optimized image data...", "processing")

        try:
            result = self._prepare_optimized_segmentation_data(rlayer)
            if result is None:
                self._set_ui_enabled(True)
                return

            arr, mask_transform, debug_info, input_coords, input_labels, input_box = result
            prep_time = time.time() - start_time

        except Exception as e:
            self._update_status(f"Error preparing data: {e}", "error")
            self._set_ui_enabled(True)
            return

        mode = "point" if self.point is not None else "bbox"

        self.worker = OptimizedSAM2Worker(
            predictor=self.predictor,
            arr=arr,
            mode=mode,
            model_choice=self.model_choice,
            point_coords=input_coords,
            point_labels=input_labels,
            box=input_box,
            mask_transform=mask_transform,
            debug_info={**debug_info, 'prep_time': prep_time},
            device=self.device
        )

        self.worker.finished.connect(self._on_segmentation_finished)
        self.worker.error.connect(self._on_segmentation_error)
        self.worker.progress.connect(self._on_segmentation_progress)
        self.worker.start()

    def _prepare_optimized_segmentation_data(self, rlayer):
        rpath = rlayer.source()
        adaptive_crop_size = self._get_adaptive_crop_size()

        with rasterio.open(rpath) as src:
            if self.point is not None:  # POINT MODE
                row, col = src.index(self.point.x(), self.point.y())
                center_pixel_x, center_pixel_y = col, row

                crop_size = adaptive_crop_size
                half_size = crop_size // 2

                x_min = max(0, center_pixel_x - half_size)
                y_min = max(0, center_pixel_y - half_size)
                x_max = min(src.width, center_pixel_x + half_size)
                y_max = min(src.height, center_pixel_y + half_size)

                if x_max <= x_min or y_max <= y_min:
                    self._update_status("Invalid crop area for point", "error")
                    return None

                window = rasterio.windows.Window(
                    x_min, y_min, x_max - x_min, y_max - y_min)

                try:
                    arr = src.read([1, 2, 3], window=window,
                                   out_dtype=np.uint8)
                    if arr.size == 0:
                        self._update_status("Empty crop area", "error")
                        return None
                except Exception as e:
                    self._update_status(f"Error reading raster: {e}", "error")
                    return None

                arr = np.moveaxis(arr, 0, -1)

                if arr.max() > arr.min():
                    arr_min, arr_max = arr.min(), arr.max()
                    arr = ((arr.astype(np.float32) - arr_min) /
                           (arr_max - arr_min) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)

                relative_x = center_pixel_x - x_min
                relative_y = center_pixel_y - y_min
                relative_x = max(0, min(arr.shape[1] - 1, relative_x))
                relative_y = max(0, min(arr.shape[0] - 1, relative_y))

                input_coords = np.array([[relative_x, relative_y]])
                input_labels = np.array([1])
                input_box = None
                mask_transform = src.window_transform(window)

                debug_info = {
                    'mode': 'POINT',
                    'class': self.current_class,
                    'actual_crop': f"{arr.shape[1]}x{arr.shape[0]}",
                    'device': self.device
                }

            else:  # BBOX MODE
                try:
                    window = rasterio.windows.from_bounds(
                        self.bbox.xMinimum(), self.bbox.yMinimum(),
                        self.bbox.xMaximum(), self.bbox.yMaximum(),
                        src.transform
                    )
                except Exception as e:
                    self._update_status(
                        f"Error creating bbox window: {e}", "error")
                    return None

                if window.width <= 0 or window.height <= 0:
                    self._update_status("Invalid bbox dimensions", "error")
                    return None

                try:
                    arr = src.read([1, 2, 3], window=window,
                                   out_dtype=np.uint8)
                    if arr.size == 0:
                        self._update_status("Empty bbox crop area", "error")
                        return None
                except Exception as e:
                    self._update_status(
                        f"Error reading bbox raster: {e}", "error")
                    return None

                arr = np.moveaxis(arr, 0, -1)
                if arr.max() > arr.min():
                    arr_min, arr_max = arr.min(), arr.max()
                    arr = ((arr.astype(np.float32) - arr_min) /
                           (arr_max - arr_min) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)

                x0, y0 = 0, 0
                x1, y1 = arr.shape[1] - 1, arr.shape[0] - 1
                input_box = np.array([[x0, y0, x1, y1]])
                input_coords = None
                input_labels = None
                mask_transform = src.window_transform(window)

                debug_info = {
                    'mode': 'BBOX',
                    'class': self.current_class,
                    'crop_size': f"{arr.shape[1]}x{arr.shape[0]}",
                    'device': self.device
                }

        return arr, mask_transform, debug_info, input_coords, input_labels, input_box

    def _get_adaptive_crop_size(self):
        canvas_scale = self.canvas.scale()

        if self.device == "cuda":
            base_size = 768
        elif self.device == "mps":
            base_size = 512
        else:
            base_size = 384 if self.model_choice == "MobileSAM" else 512

        if canvas_scale > 100000:
            crop_size = min(base_size * 1.5, 1024)
        elif canvas_scale > 50000:
            crop_size = int(base_size * 1.2)
        elif canvas_scale < 1000:
            crop_size = base_size // 2
        else:
            crop_size = base_size

        crop_size = max(256, crop_size)
        return crop_size

    def _on_segmentation_finished(self, result):
        try:
            import time
            start_process_time = time.time()

            self._update_status("✨ Processing results...", "processing")

            mask = result['mask']
            mask_transform = result['mask_transform']
            debug_info = result['debug_info']

            self._process_segmentation_result(mask, mask_transform, debug_info)

            process_time = time.time() - start_process_time
            prep_time = debug_info.get('prep_time', 0)
            total_time = prep_time + process_time

            model_info = f"({debug_info.get('model', 'SAM')} on {self.device.upper()})"
            self._update_status(
                f"✅ Completed in {total_time:.1f}s {model_info}! Click again to add more.", "info")

        except Exception as e:
            self._update_status(f"Error processing results: {e}", "error")
        finally:
            self._set_ui_enabled(True)
            if hasattr(self, 'worker') and self.worker:
                self.worker.deleteLater()
                self.worker = None

    def _process_segmentation_result(self, mask, mask_transform, debug_info):
        # Save mask image for traceability (ONLY if debug enabled)
        filename = None
        if self.save_debug_masks:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            class_prefix = f"{self.current_class}_" if self.current_class else ""

            if self.current_mode == 'point':
                filename = f"mask_{class_prefix}point_{self.point.x():.1f}_{self.point.y():.1f}_{timestamp}.png"
            else:
                filename = f"mask_{class_prefix}bbox_{self.bbox.width():.1f}x{self.bbox.height():.1f}_{timestamp}.png"

            filename = "".join(
                c for c in filename if c.isalnum() or c in "._-")
            mask_path = self.mask_save_dir / filename

            try:
                cv2.imwrite(str(mask_path), mask)
            except Exception as e:
                self._update_status(
                    f"Failed to save debug mask: {e}", "warning")
                filename = "save_failed"

        # Threshold mask to binary and clean it up
        try:
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Morphological operations to clean up the mask
            open_kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

            close_kernel = np.ones((7, 7), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

            # Remove small objects
            nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8)
            min_size = 20
            cleaned = np.zeros(binary.shape, dtype=np.uint8)
            for i in range(1, nlabels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    cleaned[labels == i] = 255
            binary = cleaned

        except Exception as e:
            self._update_status(
                f"Error thresholding/refining mask: {e}", "error")
            return

        # Convert mask to features
        feats = []
        try:
            for geom, _ in shapes(binary, mask=binary > 0, transform=mask_transform):
                shp_geom = shape(geom)
                if not shp_geom.is_valid:
                    shp_geom = shp_geom.buffer(0)
                if shp_geom.is_empty:
                    continue

                if hasattr(shp_geom, 'exterior'):
                    coords = list(shp_geom.exterior.coords)
                    qgs_points = []
                    for coord in coords:
                        if len(coord) >= 2:
                            qgs_points.append(QgsPointXY(coord[0], coord[1]))
                    if len(qgs_points) >= 3:
                        qgs_geom = QgsGeometry.fromPolygonXY([qgs_points])
                    else:
                        continue
                else:
                    try:
                        wkt_str = shp_geom.wkt
                        qgs_geom = QgsGeometry.fromWkt(wkt_str)
                    except Exception as e:
                        continue

                if not qgs_geom.isNull() and not qgs_geom.isEmpty():
                    f = QgsFeature()
                    f.setGeometry(qgs_geom)
                    feats.append(f)

        except Exception as e:
            self._update_status(
                f"Error processing geometries: {str(e)}", "error")
            return

        if not feats:
            self._update_status("No segments found", "warning")
            return

        # Store current raster layer reference
        current_layer = self.iface.activeLayer()
        if isinstance(current_layer, QgsRasterLayer):
            self.original_raster_layer = current_layer

        result_layer = self._get_or_create_class_layer(self.current_class)

        # Prepare attributes
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        crop_info = debug_info.get(
            'crop_size', 'unknown') if self.current_mode == 'bbox' else debug_info.get('actual_crop', 'unknown')
        class_color = self.classes.get(
            self.current_class, {}).get('color', '128,128,128')
        canvas_scale = self.canvas.scale()

        # Set attributes for features
        for feat in feats:
            self.segment_counts[self.current_class] += 1
            feat.setAttributes([
                self.segment_counts[self.current_class],
                self.current_class,
                class_color,
                self.current_mode,
                timestamp_str,
                filename or "debug_disabled",
                crop_info,
                canvas_scale
            ])

        try:
            # Add features and track for undo
            result_layer.startEditing()
            success = result_layer.dataProvider().addFeatures(feats)
            result_layer.commitChanges()

            if success:
                # Track for undo functionality
                all_features = list(result_layer.getFeatures())
                new_feature_ids = [f.id() for f in all_features[-len(feats):]]
                self.undo_stack.append((self.current_class, new_feature_ids))
                self.undoBtn.setEnabled(True)

            result_layer.updateExtents()
            result_layer.triggerRepaint()

            # Keep raster layer selected
            if self.keep_raster_selected and self.original_raster_layer:
                self.iface.setActiveLayer(self.original_raster_layer)

        except Exception as e:
            self._update_status(f"Error adding features: {e}", "error")
            return

        # Update layer name
        total_features = result_layer.featureCount()
        color_info = f" [RGB:{class_color}]"
        new_layer_name = f"SAM_{self.current_class} ({total_features} parts){color_info}"
        result_layer.setName(new_layer_name)

        # Clear visual feedback
        if self.current_mode == 'point':
            self.pointTool.clear_feedback()
        elif self.current_mode == 'bbox':
            self.bboxTool.clear_feedback()

        # Update status and stats
        undo_hint = " (↶ Undo available)" if len(feats) > 0 else ""
        self._update_status(
            f"✅ Added {len(feats)} [{self.current_class}] polygons!{undo_hint}", "info")
        self._update_stats()

    def _get_or_create_class_layer(self, class_name):
        if class_name in self.result_layers and self.result_layers[class_name]:
            return self.result_layers[class_name]

        rlayer = self.iface.activeLayer()
        if isinstance(rlayer, QgsRasterLayer):
            self.original_raster_layer = rlayer

        class_info = self.classes.get(class_name, {'color': '128,128,128'})
        color = class_info['color']

        layer_name = f"SAM_{class_name}_{datetime.datetime.now():%H%M%S} [RGB:{color}]"
        layer = QgsVectorLayer(
            f"Polygon?crs={rlayer.crs().authid()}", layer_name, "memory")

        layer.dataProvider().addAttributes([
            QgsField("segment_id", QVariant.Int),
            QgsField("class_name", QVariant.String),
            QgsField("class_color", QVariant.String),
            QgsField("method", QVariant.String),
            QgsField("timestamp", QVariant.String),
            QgsField("mask_file", QVariant.String),
            QgsField("crop_size", QVariant.String),
            QgsField("canvas_scale", QVariant.Double)
        ])
        layer.updateFields()

        self._apply_class_style(layer, class_name)

        QgsProject.instance().addMapLayer(layer)
        self.result_layers[class_name] = layer
        self.segment_counts[class_name] = 0

        if self.keep_raster_selected and self.original_raster_layer:
            self.iface.setActiveLayer(self.original_raster_layer)

        return layer

    def _apply_class_style(self, layer, class_name):
        try:
            class_info = self.classes.get(class_name, {'color': '128,128,128'})
            color = class_info['color']

            try:
                r, g, b = [int(c.strip()) for c in color.split(',')]
            except:
                r, g, b = 128, 128, 128

            symbol = QgsFillSymbol.createSimple({
                'color': f'{r},{g},{b},180',
                'outline_color': f'{r},{g},{b},255',
                'outline_width': '1.5',
                'outline_style': 'solid'
            })

            layer.renderer().setSymbol(symbol)
            layer.setOpacity(0.85)
            layer.triggerRepaint()

        except Exception as e:
            print(f"Color application failed for {class_name}: {e}")

    def _undo_last_polygon(self):
        if not self.undo_stack:
            self._update_status("No polygons to undo", "warning")
            return

        class_name, feature_ids = self.undo_stack.pop()

        if class_name not in self.result_layers:
            self._update_status(f"Class layer {class_name} not found", "error")
            return

        layer = self.result_layers[class_name]

        try:
            layer.startEditing()
            removed_count = 0
            for feature_id in feature_ids:
                if layer.deleteFeature(feature_id):
                    removed_count += 1

            layer.commitChanges()
            layer.updateExtents()
            layer.triggerRepaint()

            if class_name in self.segment_counts:
                self.segment_counts[class_name] = max(
                    0, self.segment_counts[class_name] - removed_count)

            total_features = layer.featureCount()
            class_color = self.classes.get(
                class_name, {}).get('color', '128,128,128')
            color_info = f" [RGB:{class_color}]"
            new_layer_name = f"SAM_{class_name} ({total_features} parts){color_info}"
            layer.setName(new_layer_name)

            self._update_stats()
            self._update_status(
                f"↶ Undid {removed_count} polygons from [{class_name}]", "info")

            if not self.undo_stack:
                self.undoBtn.setEnabled(False)

        except Exception as e:
            self._update_status(f"Failed to undo: {e}", "error")
            if layer.isEditable():
                layer.rollBack()

    def _export_all_classes(self):
        if not self.result_layers:
            self._update_status("No segments to export!", "warning")
            return

        exported_count = 0
        for class_name, layer in self.result_layers.items():
            if layer and layer.featureCount() > 0:
                if self._export_layer_to_shapefile(layer, class_name):
                    exported_count += 1

        if exported_count > 0:
            self._update_status(
                f"💾 Exported {exported_count} class(es) to {self.shapefile_save_dir}", "info")
        else:
            self._update_status("No segments found to export!", "warning")

    def _export_layer_to_shapefile(self, layer, class_name):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shapefile_name = f"SAM_{class_name}_{timestamp}.shp"
            shapefile_path = str(self.shapefile_save_dir / shapefile_name)

            error = QgsVectorFileWriter.writeAsVectorFormat(
                layer, shapefile_path, "utf-8", layer.crs(), "ESRI Shapefile")

            if error[0] == QgsVectorFileWriter.NoError:
                print(f"💾 Exported {class_name}: {shapefile_path}")
                return True
            else:
                print(f"❌ Export failed for {class_name}: {error}")
                return False

        except Exception as e:
            print(f"❌ Export error for {class_name}: {e}")
            return False

    def _update_stats(self):
        total_segments = sum(layer.featureCount()
                             for layer in self.result_layers.values() if layer)
        total_classes = len(
            [l for l in self.result_layers.values() if l and l.featureCount() > 0])
        self.statsLabel.setText(
            f"Total Segments: {total_segments} | Classes: {total_classes}")

    def _on_segmentation_progress(self, message):
        self._update_status(message, "processing")

    def _on_segmentation_error(self, error_message):
        self._update_status(f"❌ {error_message}", "error")
        self._set_ui_enabled(True)
        if hasattr(self, 'worker') and self.worker:
            self.worker.deleteLater()
            self.worker = None

    def _cancel_segmentation(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
            self._update_status("Segmentation cancelled", "warning")
            self._set_ui_enabled(True)

    def _set_ui_enabled(self, enabled):
        self.pointModeBtn.setEnabled(enabled)
        self.bboxModeBtn.setEnabled(enabled)
        self.classComboBox.setEnabled(enabled)
        self.addClassBtn.setEnabled(enabled)
        self.editClassBtn.setEnabled(enabled)
        self.exportBtn.setEnabled(enabled)
        self.selectFolderBtn.setEnabled(True)
        self.saveDebugCheckbox.setEnabled(True)

        if enabled and self.undo_stack:
            self.undoBtn.setEnabled(True)
        elif not enabled:
            pass  # Keep undo available during processing
        else:
            self.undoBtn.setEnabled(False)

        if hasattr(self, 'progressBar'):
            self.progressBar.setVisible(not enabled)
        if hasattr(self, 'cancelBtn'):
            self.cancelBtn.setVisible(not enabled)

        if not enabled:
            self.setCursor(Qt.WaitCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _update_status(self, message, status_type="info"):
        colors = {
            "info": "#e8f5e8; border-left: 4px solid #4CAF50",
            "warning": "#fff3cd; border-left: 4px solid #FFC107",
            "error": "#f8d7da; border-left: 4px solid #DC3545",
            "processing": "#d4edda; border-left: 4px solid #17a2b8"
        }
        color_style = colors.get(status_type, colors["info"])
        self.statusLabel.setText(message)
        self.statusLabel.setStyleSheet(
            f"padding: 6px; background-color: {color_style}; font-size: 15px; border-radius: 3px;")


class SegSamDialog(QtWidgets.QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.control_panel = None

        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("GeoOSAM Control Panel")
        label.setStyleSheet(
            "font-size: 17px; font-weight: bold; padding: 10px;")
        layout.addWidget(label)

        show_panel_btn = QtWidgets.QPushButton("Show Control Panel")
        show_panel_btn.clicked.connect(self._show_control_panel)
        layout.addWidget(show_panel_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setWindowTitle("GeoOSAM")
        self.resize(250, 120)

    def _show_control_panel(self):
        if not self.control_panel:
            self.control_panel = GeoOSAMControlPanel(self.iface)
            self.iface.addDockWidget(
                Qt.RightDockWidgetArea, self.control_panel)
        self.control_panel.show()
        self.control_panel.raise_()
        self.close()
