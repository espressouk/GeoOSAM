"""
Minimal QGIS stubs so geo_osam_dialog can be imported in a plain Python
environment (no QGIS installation required).
"""
import sys
from unittest.mock import MagicMock


class _QThread:
    def __init__(self, *a, **kw):
        pass

    def isInterruptionRequested(self):
        return False

    def currentThreadId(self):
        return 1

    def start(self):
        pass


class _QDockWidget:
    def __init__(self, *a, **kw):
        pass


class _QAbstractButton:
    def __init__(self, *a, **kw):
        pass


def _pyqtSignal(*args, **kwargs):
    class _Sig:
        def connect(self, *a, **kw):
            pass
        def emit(self, *a, **kw):
            pass
        def disconnect(self, *a, **kw):
            pass
    return _Sig()


_qtcore = MagicMock()
_qtcore.QThread = _QThread
_qtcore.pyqtSignal = _pyqtSignal
_qtcore.Qt = MagicMock()
_qtcore.QVariant = MagicMock()
_qtcore.QSettings = MagicMock()

_qtwidgets = MagicMock()
_qtwidgets.QDockWidget = _QDockWidget
_qtwidgets.QAbstractButton = _QAbstractButton

_qtgui = MagicMock()

_qgis = MagicMock()
_qgis_pyqt = MagicMock()
_qgis_pyqt.QtCore = _qtcore
_qgis_pyqt.QtWidgets = _qtwidgets
_qgis_pyqt.QtGui = _qtgui

sys.modules.update({
    "qgis":                         _qgis,
    "qgis.PyQt":                    _qgis_pyqt,
    "qgis.PyQt.QtCore":             _qtcore,
    "qgis.PyQt.QtWidgets":          _qtwidgets,
    "qgis.PyQt.QtGui":              _qtgui,
    "qgis.core":                    MagicMock(),
    "qgis.gui":                     MagicMock(),
    "hydra":                        MagicMock(),
    "hydra.core":                   MagicMock(),
    "hydra.core.global_hydra":      MagicMock(),
    "sam2":                         MagicMock(),
    "sam2.build_sam":               MagicMock(),
    "sam2.sam2_image_predictor":    MagicMock(),
})
