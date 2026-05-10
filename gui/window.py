"""
Main PyQt5 GUI window for Eye Tracker.
Displays real-time tracking statistics, mood, and gestures with cyberpunk aesthetic.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QProgressBar, QGridLayout, QApplication, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QBrush, QImage
from PyQt5.QtCore import QPoint, QRect

import cv2
import numpy as np

from gui.theme import get_stylesheet, COLORS, get_alert_stylesheet

logger = logging.getLogger(__name__)


class EyeTrackerWindow(QMainWindow):

    export_requested = pyqtSignal()
    sound_toggled = pyqtSignal(bool)

    def __init__(self, start_minimized: bool = False):
        super().__init__()

        self.start_minimized = start_minimized
        self.sound_enabled = True

        self.setWindowTitle("Eye Tracker - Face Analytics")
        self.setGeometry(100, 100, 620, 880)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.setStyleSheet(get_stylesheet())

        self._setup_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(100)

        self.session_data = {
            'state': 'idle',
            'time_remaining': 0,
            'block_number': 0,
            'blinks_per_minute': 0.0,
            'avg_ear': 0.0,
            'on_screen_percent': 0.0,
            'drowsiness_level': 'normal',
            'drowsiness_reason': '',
            'mood': 'neutral',
            'mood_confidence': 0.0,
            'mouth_open': False,
            'smiling': False,
            'yawning': False,
            'brow_raised': False,
            'jaw_open_amount': 0.0,
            'smile_amount': 0.0,
            'smile_count': 0,
            'yawn_count': 0,
            'mouth_open_count': 0,
        }

        if self.start_minimized:
            self.showMinimized()
        else:
            self.show()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(6)

        title = QLabel("EYE TRACKER - FACE ANALYTICS")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Courier New", 14, QFont.Bold))
        main_layout.addWidget(title)

        self.camera_label = QLabel("Waiting for camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(320, 240)
        self.camera_label.setStyleSheet(
            f"background-color: #000000; border: 2px solid {COLORS['primary_neon']};"
        )
        main_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)

        # --- Stats Grid ---
        stats_frame = QFrame()
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setSpacing(6)

        row = 0
        self._add_stat_row(stats_layout, row, "STATUS:", "status_value", "IDLE", COLORS['primary_neon'],
                           "TIME LEFT:", "timer_value", "25:00", COLORS['secondary_neon'])

        row = 1
        self._add_stat_row(stats_layout, row, "BLINKS/MIN:", "blink_value", "0.0", COLORS['primary_neon'],
                           "EAR:", "ear_value", "0.50", COLORS['primary_neon'])

        row = 2
        self._add_stat_row(stats_layout, row, "FOCUS:", "gaze_value", "0%", COLORS['secondary_neon'],
                           "BLOCK:", "block_value", "0", COLORS['accent_neon'])

        main_layout.addWidget(stats_frame)

        # --- Mood & Gesture Panel ---
        mood_frame = QFrame()
        mood_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface_variant']};
                border: 2px solid {COLORS['tertiary_neon']};
                border-radius: 4px;
            }}
        """)
        mood_layout = QGridLayout(mood_frame)
        mood_layout.setContentsMargins(8, 8, 8, 8)
        mood_layout.setSpacing(4)

        mood_title = QLabel("MOOD & GESTURES")
        mood_title.setFont(QFont("Courier New", 11, QFont.Bold))
        mood_title.setStyleSheet(f"color: {COLORS['tertiary_neon']};")
        mood_title.setAlignment(Qt.AlignCenter)
        mood_layout.addWidget(mood_title, 0, 0, 1, 4)

        self.mood_label = QLabel("NEUTRAL")
        self.mood_label.setFont(QFont("Courier New", 13, QFont.Bold))
        self.mood_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.mood_label.setAlignment(Qt.AlignCenter)
        mood_layout.addWidget(self.mood_label, 1, 0, 1, 2)

        self.mood_conf_label = QLabel("0%")
        self.mood_conf_label.setFont(QFont("Courier New", 10))
        self.mood_conf_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.mood_conf_label.setAlignment(Qt.AlignCenter)
        mood_layout.addWidget(self.mood_conf_label, 1, 2, 1, 2)

        self.smile_bar = self._make_progress_bar()
        self._add_gesture_row(mood_layout, 2, "Smile:", self.smile_bar, "smile_count_label", "0")

        self.mouth_bar = self._make_progress_bar()
        self._add_gesture_row(mood_layout, 3, "Mouth:", self.mouth_bar, "mouth_count_label", "0")

        self.yawn_bar = self._make_progress_bar()
        self._add_gesture_row(mood_layout, 4, "Yawns:", self.yawn_bar, "yawn_count_label", "0")

        self.gesture_status = QLabel("")
        self.gesture_status.setFont(QFont("Courier New", 9))
        self.gesture_status.setStyleSheet(f"color: {COLORS['secondary_neon']};")
        self.gesture_status.setAlignment(Qt.AlignCenter)
        mood_layout.addWidget(self.gesture_status, 5, 0, 1, 4)

        main_layout.addWidget(mood_frame)

        # --- Alert Indicator ---
        alert_frame = QFrame()
        alert_frame.setStyleSheet(get_alert_stylesheet('normal'))
        alert_layout = QVBoxLayout(alert_frame)
        alert_layout.setContentsMargins(8, 8, 8, 8)

        alert_title = QLabel("ALERT STATUS")
        alert_title.setObjectName("stat_label")
        alert_title.setAlignment(Qt.AlignCenter)

        self.alert_indicator = QLabel("NORMAL")
        self.alert_indicator.setFont(QFont("Courier New", 12, QFont.Bold))
        self.alert_indicator.setAlignment(Qt.AlignCenter)
        self.alert_indicator.setStyleSheet(f"color: {COLORS['success']};")

        self.alert_reason = QLabel("")
        self.alert_reason.setFont(QFont("Courier New", 9))
        self.alert_reason.setAlignment(Qt.AlignCenter)
        self.alert_reason.setStyleSheet(f"color: {COLORS['text_secondary']};")

        alert_layout.addWidget(alert_title)
        alert_layout.addWidget(self.alert_indicator)
        alert_layout.addWidget(self.alert_reason)

        main_layout.addWidget(alert_frame)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.sound_btn = QPushButton("SOUND ON")
        self.sound_btn.setMaximumWidth(110)
        self.sound_btn.clicked.connect(self._toggle_sound)
        button_layout.addWidget(self.sound_btn)

        export_btn = QPushButton("EXPORT ALL")
        export_btn.setMaximumWidth(110)
        export_btn.clicked.connect(self._on_export)
        button_layout.addWidget(export_btn)

        minimize_btn = QPushButton("_")
        minimize_btn.setMaximumWidth(40)
        minimize_btn.clicked.connect(self.showMinimized)
        button_layout.addWidget(minimize_btn)

        exit_btn = QPushButton("X")
        exit_btn.setObjectName("alert_btn")
        exit_btn.setMaximumWidth(40)
        exit_btn.clicked.connect(self.close)
        button_layout.addWidget(exit_btn)

        main_layout.addLayout(button_layout)

        self.statusBar().showMessage("Ready | Waiting for camera...")
        self.statusBar().setStyleSheet(
            f"background-color: {COLORS['surface']}; color: {COLORS['text_secondary']};"
        )

    def _add_stat_row(self, layout, row, l1_text, v1_name, v1_default, v1_color,
                      l2_text, v2_name, v2_default, v2_color):
        l1 = QLabel(l1_text)
        l1.setObjectName("stat_label")
        v1 = QLabel(v1_default)
        v1.setFont(QFont("Courier New", 10, QFont.Bold))
        v1.setStyleSheet(f"color: {v1_color};")
        setattr(self, v1_name, v1)

        l2 = QLabel(l2_text)
        l2.setObjectName("stat_label")
        v2 = QLabel(v2_default)
        v2.setFont(QFont("Courier New", 10, QFont.Bold))
        v2.setStyleSheet(f"color: {v2_color};")
        setattr(self, v2_name, v2)

        layout.addWidget(l1, row, 0)
        layout.addWidget(v1, row, 1)
        layout.addWidget(l2, row, 2)
        layout.addWidget(v2, row, 3)

    def _make_progress_bar(self):
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setMaximumHeight(14)
        bar.setTextVisible(False)
        return bar

    def _add_gesture_row(self, layout, row, label_text, bar, count_name, default):
        label = QLabel(label_text)
        label.setFont(QFont("Courier New", 9))
        label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(label, row, 0)

        layout.addWidget(bar, row, 1, 1, 2)

        count = QLabel(default)
        count.setFont(QFont("Courier New", 9, QFont.Bold))
        count.setStyleSheet(f"color: {COLORS['tertiary_neon']};")
        count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        setattr(self, count_name, count)
        layout.addWidget(count, row, 3)

    def update_session_data(self, data: Dict):
        self.session_data.update(data)

    def update_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        if ch == 3:
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        else:
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _update_display(self):
        d = self.session_data

        state = d.get('state', 'idle').upper()
        self.status_value.setText(state)

        time_remaining = d.get('time_remaining', 0)
        self.timer_value.setText(f"{int(time_remaining) // 60:02d}:{int(time_remaining) % 60:02d}")

        self.blink_value.setText(f"{d.get('blinks_per_minute', 0.0):.1f}")
        self.ear_value.setText(f"{d.get('avg_ear', 0.0):.2f}")
        self.gaze_value.setText(f"{d.get('on_screen_percent', 0.0):.0f}%")
        self.block_value.setText(f"{d.get('block_number', 0)}")

        mood = d.get('mood', 'neutral')
        mood_conf = d.get('mood_confidence', 0.0)
        mood_colors = {
            'happy': COLORS['success'], 'sad': '#6666ff', 'angry': COLORS['critical'],
            'surprised': COLORS['secondary_neon'], 'neutral': COLORS['text_primary'],
            'excited': COLORS['accent_neon'], 'disgusted': '#ff6600', 'fearful': COLORS['tertiary_neon'],
        }
        mc = mood_colors.get(mood, COLORS['text_primary'])
        self.mood_label.setText(mood.upper())
        self.mood_label.setStyleSheet(f"color: {mc}; font-family: 'Courier New'; font-size: 13px; font-weight: bold;")
        self.mood_conf_label.setText(f"{mood_conf:.0%}")

        smile_amount = d.get('smile_amount', 0.0)
        self.smile_bar.setValue(int(min(100, smile_amount * 200)))
        self.smile_count_label.setText(f"{d.get('smile_count', 0)}")

        jaw_amount = d.get('jaw_open_amount', 0.0)
        self.mouth_bar.setValue(int(min(100, jaw_amount * 200)))
        self.mouth_count_label.setText(f"{d.get('mouth_open_count', 0)}")

        self.yawn_bar.setValue(100 if d.get('yawning', False) else 0)
        self.yawn_count_label.setText(f"{d.get('yawn_count', 0)}")

        gestures = []
        if d.get('smiling'):
            gestures.append("SMILING")
        if d.get('mouth_open'):
            gestures.append("MOUTH OPEN")
        if d.get('yawning'):
            gestures.append("YAWNING")
        if d.get('brow_raised'):
            gestures.append("BROW UP")
        self.gesture_status.setText(" | ".join(gestures) if gestures else "")

        drowsiness_level = d.get('drowsiness_level', 'normal').lower()
        if drowsiness_level == 'critical':
            self.alert_indicator.setText("!! CRITICAL !!")
            self.alert_indicator.setStyleSheet(f"color: {COLORS['critical']};")
        elif drowsiness_level == 'warning':
            self.alert_indicator.setText("! WARNING !")
            self.alert_indicator.setStyleSheet(f"color: {COLORS['warning']};")
        else:
            self.alert_indicator.setText("NORMAL")
            self.alert_indicator.setStyleSheet(f"color: {COLORS['success']};")

        self.alert_reason.setText(d.get('drowsiness_reason', '')[:60])
        self.statusBar().showMessage(f"Session Active | {state} | Mood: {mood}")

    def _toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        self.sound_btn.setText("SOUND ON" if self.sound_enabled else "SOUND OFF")
        self.sound_toggled.emit(self.sound_enabled)

    def _on_export(self):
        self.export_requested.emit()
        self.statusBar().showMessage("Exporting all analytics...")

    def show_alert(self, title: str, level: str = 'warning'):
        self.session_data['drowsiness_reason'] = title
        self.session_data['drowsiness_level'] = level
        self.statusBar().showMessage(f"!! {title}")

    def closeEvent(self, event):
        self.update_timer.stop()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.showMinimized()
        elif event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
            self.close()
        else:
            super().keyPressEvent(event)
