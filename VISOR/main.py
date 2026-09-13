import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from shape_detector.config import *
from shape_detector.preprocessor import preprocess
from shape_detector.detector import detect
from shape_detector.utils import get_dimensions, calculate_confidence, get_center

def create_template():
    size   = 300
    canvas = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(canvas, (20,  20),  (280, 280), 255, 3)
    cv2.rectangle(canvas, (60,  60),  (240, 240), 255, 3)
    cv2.rectangle(canvas, (100, 100), (200, 200), 255, 3)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)

BG      = "#0a0800"
BG2     = "#150f00"
GOLD    = "#d4af37"
GOLDDIM = "#6b5010"

STYLE_MAIN = f"""
    QMainWindow {{ background: {BG}; }}
    QWidget {{ background: {BG}; color: {GOLD}; font-family: Courier New; }}
"""
STYLE_BTN_START = f"""
    QPushButton {{
        background: transparent; color: {GOLD};
        border: 2px solid {GOLD}; border-radius: 4px;
        font-family: Courier New; font-size: 13px;
        letter-spacing: 2px; padding: 8px 20px;
    }}
    QPushButton:hover {{ background: rgba(212,175,55,0.15); }}
    QPushButton:pressed {{ background: rgba(212,175,55,0.3); }}
"""
STYLE_BTN_RESET = f"""
    QPushButton {{
        background: transparent; color: {GOLDDIM};
        border: 2px solid {GOLDDIM}; border-radius: 4px;
        font-family: Courier New; font-size: 13px;
        letter-spacing: 2px; padding: 8px 20px;
    }}
    QPushButton:hover {{ background: rgba(107,80,16,0.2); }}
    QPushButton:pressed {{ background: rgba(107,80,16,0.4); }}
"""
STYLE_BTN_STOP = f"""
    QPushButton {{
        background: rgba(212,175,55,0.2); color: {GOLD};
        border: 2px solid {GOLD}; border-radius: 4px;
        font-family: Courier New; font-size: 13px;
        letter-spacing: 2px; padding: 8px 20px;
    }}
"""
STYLE_SLIDER = f"""
    QSlider::groove:horizontal {{ height: 4px; background: {GOLDDIM}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {GOLD}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
    QSlider::sub-page:horizontal {{ background: {GOLD}; border-radius: 2px; }}
"""
STYLE_STAT     = f"QLabel {{ color: {GOLD}; font-family: Courier New; font-size: 15px; font-weight: bold; background: transparent; }}"
STYLE_STAT_LBL = f"QLabel {{ color: {GOLDDIM}; font-family: Courier New; font-size: 10px; letter-spacing: 2px; background: transparent; }}"
STYLE_DIVIDER  = f"background: {GOLDDIM};"


class VISOR(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISOR  v1.0")
        self.setStyleSheet(STYLE_MAIN)
        self.setMinimumSize(1280, 820)

        self.cap          = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.template     = create_template()
        self.running      = False   # shuru mein stopped
        self.count        = 0
        self.best_conf    = 0
        self.last_det     = 0
        self.sensitivity  = 75
        self.start_time   = time.time()
        self.fps_time     = time.time()
        self.fps          = 0
        self.frame_count  = 0
        self.last_contour = None
        self.last_score   = None
        self.last_dims    = None
        self.last_conf    = 0

        self._build_ui()

        self.qtimer = QTimer()
        self.qtimer.timeout.connect(self._update)
        self.qtimer.start(30)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(50)
        topbar.setStyleSheet(f"background: {BG2}; border-bottom: 1px solid {GOLDDIM};")
        tl = QHBoxLayout(topbar)
        tl.setContentsMargins(16, 0, 16, 0)

        logo = QLabel("VISOR")
        logo.setStyleSheet(f"color: {GOLD}; font-size: 18px; font-weight: bold; font-family: Courier New; letter-spacing: 3px; background: transparent;")
        tl.addWidget(logo)
        tl.addStretch()

        self.status_lbl = QLabel("● STOPPED")
        self.status_lbl.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 13px; letter-spacing: 2px; background: transparent;")
        tl.addWidget(self.status_lbl)
        tl.addStretch()

        self.fps_lbl = QLabel("FPS: --")
        self.fps_lbl.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 12px; background: transparent;")
        tl.addWidget(self.fps_lbl)

        self.timer_lbl = QLabel("00:00")
        self.timer_lbl.setStyleSheet(f"color: {GOLD}; font-family: Courier New; font-size: 12px; margin-left: 16px; background: transparent;")
        tl.addWidget(self.timer_lbl)
        root.addWidget(topbar)

        # Webcam feed
        self.feed_lbl = QLabel()
        self.feed_lbl.setAlignment(Qt.AlignCenter)
        self.feed_lbl.setStyleSheet(f"background: #050400;")
        self.feed_lbl.setMinimumHeight(500)
        root.addWidget(self.feed_lbl, stretch=1)

        d1 = QFrame(); d1.setFixedHeight(1); d1.setStyleSheet(STYLE_DIVIDER)
        root.addWidget(d1)

        # Stats bar
        stats_bar = QWidget()
        stats_bar.setFixedHeight(65)
        stats_bar.setStyleSheet(f"background: {BG2};")
        sl = QHBoxLayout(stats_bar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)

        self.stat_labels = {}
        for i, name in enumerate(["HEIGHT", "WIDTH", "AREA", "PERIMETER", "CONFIDENCE"]):
            if i > 0:
                div = QFrame(); div.setFixedWidth(1); div.setStyleSheet(STYLE_DIVIDER)
                sl.addWidget(div)
            box = QWidget(); box.setStyleSheet(f"background: {BG2};")
            bl  = QVBoxLayout(box)
            bl.setContentsMargins(0, 8, 0, 8); bl.setSpacing(2); bl.setAlignment(Qt.AlignCenter)
            lbl = QLabel(name); lbl.setAlignment(Qt.AlignCenter); lbl.setStyleSheet(STYLE_STAT_LBL)
            bl.addWidget(lbl)
            val = QLabel("-- px" if name != "AREA" else "--")
            val.setAlignment(Qt.AlignCenter); val.setStyleSheet(STYLE_STAT)
            bl.addWidget(val)
            self.stat_labels[name] = val
            sl.addWidget(box, stretch=1)
        root.addWidget(stats_bar)

        d2 = QFrame(); d2.setFixedHeight(1); d2.setStyleSheet(STYLE_DIVIDER)
        root.addWidget(d2)

        # Controls bar
        ctrl_bar = QWidget()
        ctrl_bar.setFixedHeight(65)
        ctrl_bar.setStyleSheet(f"background: #0f0b00;")
        cl = QHBoxLayout(ctrl_bar)
        cl.setContentsMargins(16, 8, 16, 8)
        cl.setSpacing(12)

        self.start_btn = QPushButton("▶  START")
        self.start_btn.setStyleSheet(STYLE_BTN_START)
        self.start_btn.setFixedSize(150, 46)
        self.start_btn.clicked.connect(self._toggle_start)
        cl.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹  STOP")
        self.stop_btn.setStyleSheet(STYLE_BTN_RESET)
        self.stop_btn.setFixedSize(130, 46)
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        cl.addWidget(self.stop_btn)

        reset_btn = QPushButton("↺  RESET")
        reset_btn.setStyleSheet(STYLE_BTN_RESET)
        reset_btn.setFixedSize(120, 46)
        reset_btn.clicked.connect(self._reset)
        cl.addWidget(reset_btn)

        cl.addSpacing(16)

        sens_lbl = QLabel("SENSITIVITY")
        sens_lbl.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 10px; letter-spacing: 1px; background: transparent;")
        cl.addWidget(sens_lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 100)
        self.slider.setValue(75)
        self.slider.setFixedWidth(200)
        self.slider.setStyleSheet(STYLE_SLIDER)
        self.slider.valueChanged.connect(self._sens_changed)
        cl.addWidget(self.slider)

        self.sens_val_lbl = QLabel("75%")
        self.sens_val_lbl.setStyleSheet(f"color: {GOLD}; font-family: Courier New; font-size: 12px; background: transparent;")
        cl.addWidget(self.sens_val_lbl)

        cl.addStretch()

        # Session info
        session_box = QWidget(); session_box.setStyleSheet("background: transparent;")
        sb = QHBoxLayout(session_box); sb.setSpacing(20); sb.setContentsMargins(0,0,0,0)
        self.session_labels = {}
        for key in ["COUNT", "LAST", "BEST"]:
            w = QWidget(); w.setStyleSheet("background: transparent;")
            wl = QVBoxLayout(w); wl.setSpacing(1); wl.setContentsMargins(0,0,0,0)
            k = QLabel(key)
            k.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 9px; letter-spacing: 1px; background: transparent;")
            k.setAlignment(Qt.AlignCenter)
            v = QLabel("--")
            v.setStyleSheet(f"color: {GOLD}; font-family: Courier New; font-size: 13px; font-weight: bold; background: transparent;")
            v.setAlignment(Qt.AlignCenter)
            wl.addWidget(k); wl.addWidget(v)
            self.session_labels[key] = v
            sb.addWidget(w)
        cl.addWidget(session_box)
        root.addWidget(ctrl_bar)

    def _toggle_start(self):
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_btn.setStyleSheet(STYLE_BTN_RESET)
        self.stop_btn.setStyleSheet(STYLE_BTN_STOP)
        self.status_lbl.setText("● SCANNING...")
        self.status_lbl.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 13px; letter-spacing: 2px; background: transparent;")

    def _stop(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.start_btn.setStyleSheet(STYLE_BTN_START)
        self.stop_btn.setStyleSheet(STYLE_BTN_RESET)
        self.status_lbl.setText("● STOPPED")
        self.status_lbl.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 13px; letter-spacing: 2px; background: transparent;")

    def _reset(self):
        self.count        = 0
        self.best_conf    = 0
        self.last_det     = 0
        self.last_contour = None
        self.last_score   = None
        self.last_dims    = None
        self.last_conf    = 0
        self.start_time   = time.time()
        self.running      = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.start_btn.setStyleSheet(STYLE_BTN_RESET)
        self.stop_btn.setStyleSheet(STYLE_BTN_STOP)
        self.status_lbl.setText("● SCANNING...")
        self.status_lbl.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 13px; letter-spacing: 2px; background: transparent;")
        self.session_labels["COUNT"].setText("0")
        self.session_labels["LAST"].setText("--")
        self.session_labels["BEST"].setText("--")
        for name in self.stat_labels:
            self.stat_labels[name].setText("-- px" if name != "AREA" else "--")

    def _sens_changed(self, val):
        self.sensitivity = val
        self.sens_val_lbl.setText(f"{val}%")

    def _update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # FPS
        self.frame_count += 1
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps         = self.frame_count
            self.frame_count = 0
            self.fps_time    = now
        self.fps_lbl.setText(f"FPS: {self.fps}")

        # Timer
        elapsed = int(time.time() - self.start_time)
        self.timer_lbl.setText(f"{elapsed//60:02d}:{elapsed%60:02d}")

        if self.running:
            thresh         = preprocess(frame)
            contour, score = detect(frame, thresh, self.template)

            # Stability — last contour 10 frames tak rakho
            if contour is not None:
                self.last_contour = contour
                self.last_score   = score
                self.last_det     = 0
            else:
                self.last_det += 1
                if self.last_det < 30 and self.last_contour is not None:
                    contour = self.last_contour
                    score   = self.last_score

            detected = contour is not None

            if detected:
                dims       = get_dimensions(contour)
                confidence = calculate_confidence(score)
                cx, cy     = get_center(contour)

                self.last_dims = dims
                self.last_conf = confidence

                if confidence > self.best_conf:
                    self.best_conf = confidence
                if self.last_det == 0:
                    self.count += 1

                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                pad = 8
                cv2.rectangle(frame, (x-pad, y-pad), (x+w+pad, y+h+pad), (55, 175, 212), 2)
                if cx and cy:
                    cv2.line(frame, (cx-12, cy), (cx+12, cy), (55, 175, 212), 1)
                    cv2.line(frame, (cx, cy-12), (cx, cy+12), (55, 175, 212), 1)
                cv2.putText(frame, "TARGET", (x-pad, y-pad-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (55, 175, 212), 1)

                # Update stats
                self.stat_labels["HEIGHT"].setText(f"{dims['height']} px")
                self.stat_labels["WIDTH"].setText(f"{dims['width']} px")
                self.stat_labels["AREA"].setText(f"{dims['area']} px")
                self.stat_labels["PERIMETER"].setText(f"{dims['perimeter']} px")
                self.stat_labels["CONFIDENCE"].setText(f"{confidence}%")

                self.status_lbl.setText("● DETECTED")
                self.status_lbl.setStyleSheet(f"color: #d4af37; font-family: Courier New; font-size: 13px; letter-spacing: 2px; background: transparent;")
                self.session_labels["COUNT"].setText(str(self.count))
                self.session_labels["BEST"].setText(f"{self.best_conf}%")

            else:
                # Detect nahi hua — stats clear karo
                self.last_contour = None
                self.status_lbl.setText("● SCANNING...")
                self.status_lbl.setStyleSheet(f"color: {GOLDDIM}; font-family: Courier New; font-size: 13px; letter-spacing: 2px; background: transparent;")
                self.session_labels["LAST"].setText(f"{self.last_det}s")
                self.stat_labels["HEIGHT"].setText("-- px")
                self.stat_labels["WIDTH"].setText("-- px")
                self.stat_labels["AREA"].setText("--")
                self.stat_labels["PERIMETER"].setText("-- px")
                self.stat_labels["CONFIDENCE"].setText("--%")

        # Show frame
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        img     = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        pix     = QPixmap.fromImage(img)
        self.feed_lbl.setPixmap(
            pix.scaled(self.feed_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event):
        self.qtimer.stop()
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VISOR()
    win.show()
    sys.exit(app.exec_())