"""
Eye Tracker Main Application
Complete end-to-end integration of video capture, detection, mood analysis, and GUI.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_qt_plugin_path = os.path.join(_script_dir, 'venv', 'lib', 'python3.14', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
if os.path.isdir(_qt_plugin_path):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = _qt_plugin_path
os.environ.pop('QT_QPA_PLATFORM', None)
os.environ.pop('QT_PLUGIN_PATH', None)

import logging
import time
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eye_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap

import cv2
import numpy as np

from tracker.detector import DetectionPipeline, WebcamCapture
from tracker.blink import BlinkDetector, BlinkStatistics
from tracker.gaze import GazeEstimator, GazeStatistics, OffScreenTracker
from tracker.mood import MoodTracker
from tracker.session import SessionManager, DrowsinessLevel
from database.db import DatabaseHandler
from database.export import CSVExporter
from gui.window import EyeTrackerWindow
from gui.theme import COLORS


DB_LOG_INTERVAL = 5


class TrackingThread(QThread):
    update_gui = pyqtSignal(object)
    alert_triggered = pyqtSignal(str, str)
    status_message = pyqtSignal(str)
    frame_ready = pyqtSignal(object)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.is_running = False
        self.sound_enabled = True
        self.session_id = None
        self.frame_count = 0
        self.start_time = None

        self.detection_pipeline = DetectionPipeline(yolo_model_path="yolov8n.pt")
        self.webcam = WebcamCapture(camera_id=camera_id, width=640, height=480, fps=30)
        self.blink_detector = BlinkDetector(ear_threshold=0.2, blink_frames=3)
        self.gaze_estimator = GazeEstimator()
        self.off_screen_tracker = OffScreenTracker(off_screen_timeout=3.0)
        self.mood_tracker = MoodTracker()
        self.db_handler = DatabaseHandler()
        self.session_manager = SessionManager(work_minutes=25, rest_minutes=5, db_handler=self.db_handler)
        self.csv_exporter = CSVExporter()

        self._prev_blink_count = 0
        self._prev_mood = "neutral"
        self._prev_gestures = {"smiling": False, "mouth_open": False, "yawning": False, "brow_raised": False}

    def run(self):
        logger.info("Tracking thread started")

        self.session_id = self.db_handler.create_session()
        self.session_manager.start_session()
        self.start_time = time.time()
        self.is_running = True

        while self.is_running and self.webcam.is_open:
            try:
                ret, frame = self.webcam.read()
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue

                self.frame_count += 1
                elapsed = time.time() - self.start_time

                detections = self.detection_pipeline.process_frame(frame)
                landmarks = detections.get('landmarks')
                blendshapes = detections.get('blendshapes')
                gray_frame = detections.get('gray_frame')
                face_bbox = detections.get('face_bbox')

                blink_data = self.blink_detector.detect(landmarks, frame_idx=self.frame_count)
                gaze_data = self.gaze_estimator.estimate_gaze_direction(landmarks)
                mood_data = self.mood_tracker.analyze(
                    blendshapes, landmarks=landmarks,
                    frame_gray=gray_frame, face_bbox=face_bbox,
                    frame_bgr=frame, session_id=self.session_id or 0,
                    elapsed=elapsed
                )

                off_screen_result = self.off_screen_tracker.update(
                    not gaze_data.get('is_on_screen', True),
                    elapsed
                )

                gaze_summary = {
                    'on_screen_percent': (100.0 if gaze_data.get('is_on_screen', True) else 0.0),
                    'off_screen_duration': off_screen_result.get('off_screen_duration', 0.0)
                }

                session_update = self.session_manager.update(blink_data, gaze_summary)

                if session_update is None:
                    session_update = {
                        'state_changed': False,
                        'rest_block_due': False,
                        'drowsiness_alert': None,
                        'session_summary': None
                    }

                drowsiness_alert = session_update.get('drowsiness_alert')
                drowsiness_level_str = 'normal'
                if drowsiness_alert:
                    level_obj = drowsiness_alert.get('level')
                    try:
                        drowsiness_level_str = level_obj.value if hasattr(level_obj, 'value') else str(level_obj)
                    except Exception:
                        drowsiness_level_str = str(level_obj)

                    self.alert_triggered.emit(
                        drowsiness_alert.get('reason', '')[:50],
                        drowsiness_level_str
                    )

                if session_update.get('rest_block_due'):
                    self.alert_triggered.emit("REST TIME! Take a 5-minute break", 'warning')
                    logger.info("Rest block due - alert triggered")

                self._log_events(blink_data, mood_data, detections, elapsed)

                if self.frame_count % DB_LOG_INTERVAL == 0:
                    self._log_frame(blink_data, gaze_data, mood_data, detections, elapsed)

                status = self.session_manager.get_session_status()
                display_level = drowsiness_level_str if drowsiness_alert else 'normal'
                display_reason = drowsiness_alert.get('reason', 'Normal') if drowsiness_alert else 'Normal'

                gesture = mood_data.get('gesture', {})

                self.update_gui.emit({
                    'state': status['state'],
                    'time_remaining': status['time_remaining'],
                    'block_number': status['block_number'],
                    'blinks_per_minute': blink_data.get('blinks_per_minute', 0.0),
                    'avg_ear': blink_data.get('avg_ear', 0.0),
                    'on_screen_percent': gaze_summary['on_screen_percent'],
                    'drowsiness_level': display_level,
                    'drowsiness_reason': display_reason,
                    'mood': mood_data.get('mood', 'neutral'),
                    'mood_confidence': mood_data.get('mood_confidence', 0.0),
                    'mouth_open': gesture.get('mouth_open', False),
                    'smiling': gesture.get('smiling', False),
                    'yawning': gesture.get('yawning', False),
                    'brow_raised': gesture.get('brow_raised', False),
                    'jaw_open_amount': gesture.get('jaw_open_amount', 0.0),
                    'smile_amount': gesture.get('smile_amount', 0.0),
                    'smile_count': gesture.get('smile_count', 0),
                    'yawn_count': gesture.get('yawn_count', 0),
                    'mouth_open_count': gesture.get('mouth_open_count', 0),
                })

                frame_display = self._draw_debug_overlay(frame, detections, blink_data, gaze_data, mood_data)
                self.frame_ready.emit(frame_display)

                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in tracking loop: {e}", exc_info=True)
                time.sleep(0.1)

        self._cleanup()

    def _log_events(self, blink_data: dict, mood_data: dict, detections: dict, elapsed: float):
        sid = self.session_id
        if not sid:
            return

        total_blinks = blink_data.get('total_blinks', 0)
        if total_blinks > self._prev_blink_count:
            for _ in range(total_blinks - self._prev_blink_count):
                self.db_handler.log_blink_event(sid, total_blinks, blink_data.get('avg_ear', 0.0))
            self._prev_blink_count = total_blinks

        current_mood = mood_data.get('mood', 'neutral')
        if current_mood != self._prev_mood:
            self.db_handler.log_mood_event(
                sid, current_mood,
                mood_data.get('mood_confidence', 0.0),
                mood_data.get('mood_scores', {})
            )
            self._prev_mood = current_mood

        gesture = mood_data.get('gesture', {})
        for gname in ['smiling', 'mouth_open', 'yawning', 'brow_raised']:
            if gesture.get(gname, False) and not self._prev_gestures.get(gname, False):
                self.db_handler.log_gesture_event(sid, gname, f"{gname} started at {elapsed:.1f}s")
        self._prev_gestures = {
            'smiling': gesture.get('smiling', False),
            'mouth_open': gesture.get('mouth_open', False),
            'yawning': gesture.get('yawning', False),
            'brow_raised': gesture.get('brow_raised', False),
        }

    def _log_frame(self, blink_data, gaze_data, mood_data, detections, elapsed):
        if not self.session_id:
            return
        gesture = mood_data.get('gesture', {})
        frame_data = {
            'frame_number': self.frame_count,
            'ear_left': blink_data.get('left_ear', 0.0),
            'ear_right': blink_data.get('right_ear', 0.0),
            'ear_avg': blink_data.get('avg_ear', 0.0),
            'blink_rate': blink_data.get('blinks_per_minute', 0.0),
            'is_blink': blink_data.get('is_blink', False),
            'gaze_horizontal': gaze_data.get('horizontal_ratio', 0.5),
            'gaze_vertical': gaze_data.get('vertical_ratio', 0.5),
            'is_on_screen': gaze_data.get('is_on_screen', True),
            'mood': mood_data.get('mood', 'neutral'),
            'mood_confidence': mood_data.get('mood_confidence', 0.0),
            'mouth_open_amount': gesture.get('jaw_open_amount', 0.0),
            'smile_amount': gesture.get('smile_amount', 0.0),
            'jaw_open_amount': gesture.get('jaw_open_amount', 0.0),
            'brow_raise_amount': 0.0,
            'is_mouth_open': gesture.get('mouth_open', False),
            'is_smiling': gesture.get('smiling', False),
            'is_yawning': gesture.get('yawning', False),
            'is_brow_raised': gesture.get('brow_raised', False),
            'is_eye_squinting': gesture.get('eye_squinting', False),
            'mar': gesture.get('mar', 0.0),
            'mouth_open_ratio': gesture.get('mouth_open_ratio', 0.0),
            'detection_method': detections.get('detection_method', 'none'),
        }
        self.db_handler.log_frame(self.session_id, frame_data)

    def _draw_debug_overlay(self, frame, detections, blink_data, gaze_data, mood_data):
        frame_copy = frame.copy()
        landmarks = detections.get('landmarks')

        if landmarks is not None and len(landmarks) >= 478:
            self._draw_face_skeleton(frame_copy, landmarks)

        for face in detections.get('faces', []):
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        method = detections.get('detection_method', 'none')
        cv2.putText(frame_copy, f"Detect: {method}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        ear = blink_data.get('avg_ear', 0.0)
        color = (0, 255, 0) if ear > 0.2 else (0, 0, 255)
        cv2.putText(frame_copy, f"EAR: {ear:.2f}", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        bpm = blink_data.get('blinks_per_minute', 0.0)
        cv2.putText(frame_copy, f"BPM: {bpm:.1f}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        on_screen = gaze_data.get('is_on_screen', True)
        gc = (0, 255, 0) if on_screen else (0, 0, 255)
        cv2.putText(frame_copy, "ON SCREEN" if on_screen else "OFF SCREEN", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 2)

        mood = mood_data.get('mood', 'neutral')
        mood_conf = mood_data.get('mood_confidence', 0.0)
        mcm = {
            'happy': (0, 255, 0), 'sad': (255, 100, 100), 'angry': (0, 0, 255),
            'surprised': (0, 255, 255), 'neutral': (200, 200, 200),
            'excited': (0, 200, 255), 'disgusted': (0, 150, 255), 'fearful': (180, 0, 255),
            'drowsy': (100, 100, 255), 'contempt': (180, 180, 0),
            'calibrating': (255, 255, 0),
        }
        mc = mcm.get(mood, (200, 200, 200))
        if mood == 'calibrating':
            pct = mood_data.get('calibration_progress', 0)
            cv2.putText(frame_copy, f"CALIBRATING {pct}%", (10, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc, 2)
        else:
            cv2.putText(frame_copy, f"{mood.upper()} ({mood_conf:.0%})", (10, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mc, 2)

        gesture = mood_data.get('gesture', {})
        mar = gesture.get('mar', 0.0)
        jaw = gesture.get('jaw_open_amount', 0.0)
        smile = gesture.get('smile_amount', 0.0)
        cv2.putText(frame_copy, f"MAR:{mar:.2f} JAW:{jaw:.2f} SMILE:{smile:.2f}", (10, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        tags = []
        if gesture.get('smiling'): tags.append("SMILE")
        if gesture.get('mouth_open'): tags.append("MOUTH")
        if gesture.get('yawning'): tags.append("YAWN")
        if gesture.get('brow_raised'): tags.append("BROW")
        if gesture.get('eye_squinting'): tags.append("SQUINT")
        if gesture.get('tongue_out'): tags.append("TONGUE")
        if tags:
            cv2.putText(frame_copy, " | ".join(tags), (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)

        return frame_copy

    @staticmethod
    def _draw_face_skeleton(frame, lm):
        GREEN = (0, 255, 0)
        CYAN = (255, 255, 0)
        YELLOW = (0, 255, 255)
        MAGENTA = (255, 0, 255)
        ORANGE = (0, 165, 255)
        BLUE = (255, 100, 100)
        WHITE = (255, 255, 255)
        RED = (0, 0, 255)

        def pts(indices):
            return [tuple(lm[i, :2].astype(int)) for i in indices]

        def polyline(indices, color, thickness=1):
            points = pts(indices)
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, thickness)

        def dots(indices, color, radius=2):
            for p in pts(indices):
                cv2.circle(frame, p, radius, color, -1)

        # Left eyebrow (green)
        polyline([70, 63, 105, 66, 107], GREEN, 2)
        dots([70, 63, 105, 66, 107], GREEN, 3)
        # Right eyebrow
        polyline([300, 293, 334, 296, 336], GREEN, 2)
        dots([300, 293, 334, 296, 336], GREEN, 3)

        # Left eye contour (cyan)
        polyline([33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33], CYAN, 1)
        dots([159, 145], CYAN, 3)  # top/bottom
        # Right eye contour
        polyline([263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263], CYAN, 1)
        dots([386, 374], CYAN, 3)

        # Left iris (yellow)
        dots([468, 469, 470, 471, 472], YELLOW, 2)
        # Right iris
        dots([473, 474, 475, 476, 477], YELLOW, 2)

        # Nose bridge + bottom (magenta)
        polyline([168, 6, 197, 195, 5, 4, 1, 19, 94, 2], MAGENTA, 1)
        dots([6, 1, 4, 5, 197, 195], MAGENTA, 2)

        # Outer mouth (orange)
        outer_mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                       409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
        polyline(outer_mouth, ORANGE, 2)
        dots([61, 291], ORANGE, 3)  # corners

        # Inner mouth / lips (blue)
        inner_upper = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
        polyline(inner_upper, BLUE, 1)
        dots([13, 14], RED, 3)  # upper/lower lip center

        # Face oval (white, thin)
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        polyline(face_oval, (80, 80, 80), 1)

        # Forehead line (white)
        polyline([10, 109, 67, 103, 54, 21, 162], (120, 120, 120), 1)
        polyline([10, 338, 297, 332, 284], (120, 120, 120), 1)

        # Chin
        dots([152], WHITE, 3)
        polyline([377, 152, 148], (120, 120, 120), 1)

        # Cheekbones
        dots([116, 117, 118, 345, 346, 347], (180, 130, 50), 2)

        # Tongue
        if len(lm) > 478:
            dots([478, 479, 480, 481, 482, 483, 484, 485, 486, 487], RED, 2)

    def _cleanup(self):
        if self.session_id:
            self.db_handler.end_session(self.session_id)
        self.webcam.release()
        logger.info("Tracking thread cleanup complete")

    def get_export_data(self):
        return self.session_manager.get_session_status()

    def get_csv_exporter(self):
        return self.csv_exporter

    def stop(self):
        self.is_running = False


class EyeTrackerApp:

    def __init__(self, camera_id: int = 0, start_minimized: bool = False):
        logger.info("Initializing Eye Tracker Application...")

        self.camera_id = camera_id
        self.start_minimized = start_minimized
        self.sound_enabled = True
        self.tracking_thread = None

        self.gui = EyeTrackerWindow(start_minimized=start_minimized)
        self.gui.export_requested.connect(self._handle_export)
        self.gui.sound_toggled.connect(self._handle_sound_toggle)

    def start(self):
        logger.info("Starting Eye Tracker...")

        self.tracking_thread = TrackingThread(camera_id=self.camera_id)

        if not self.tracking_thread.webcam.is_open:
            logger.error("Failed to open webcam. Exiting.")
            QMessageBox.critical(self.gui, "Error", "Failed to open webcam.")
            return False

        self.tracking_thread.update_gui.connect(self._on_gui_update)
        self.tracking_thread.alert_triggered.connect(self._on_alert)
        self.tracking_thread.status_message.connect(lambda m: self.gui.statusBar().showMessage(m))
        self.tracking_thread.frame_ready.connect(self.gui.update_frame)

        self.tracking_thread.start()

        return True

    def _on_gui_update(self, data: dict):
        self.gui.update_session_data(data)

    def _on_alert(self, reason: str, level: str):
        self.gui.show_alert(reason, level=level)
        if self.sound_enabled and level == 'critical':
            self._play_alert_sound()

    def _play_alert_sound(self):
        try:
            import sounddevice as sd
            n = np.linspace(0, 0.2, 8820, False)
            sd.play(0.3 * np.sin(2 * np.pi * 440 * n), 44100)
            sd.wait()
        except Exception as e:
            logger.warning(f"Sound error: {e}")

    def _handle_export(self):
        try:
            if self.tracking_thread and self.tracking_thread.isRunning():
                sid = self.tracking_thread.session_id
                exporter = self.tracking_thread.get_csv_exporter()
                paths = exporter.export_full_analytics(sid)
                if paths:
                    msg = f"Exported {len(paths)} files to exports/"
                    self.gui.statusBar().showMessage(msg)
                    logger.info(f"Full analytics export: {paths}")
                else:
                    self.gui.statusBar().showMessage("Export failed - no data")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            self.gui.statusBar().showMessage(f"Export failed: {e}")

    def _handle_sound_toggle(self, enabled: bool):
        self.sound_enabled = enabled
        if self.tracking_thread:
            self.tracking_thread.sound_enabled = enabled
        logger.info(f"Sound {'enabled' if enabled else 'disabled'}")

    def shutdown(self):
        logger.info("Shutting down...")
        if self.tracking_thread:
            self.tracking_thread.stop()
            self.tracking_thread.wait(2000)


def main():
    parser = argparse.ArgumentParser(description="Eye Tracker - Driver Drowsiness Detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--minimized", action="store_true", help="Start with minimized GUI")

    args = parser.parse_args()

    qt_app = QApplication(sys.argv)

    app = EyeTrackerApp(camera_id=args.camera, start_minimized=args.minimized)

    if not app.start():
        sys.exit(1)

    result = qt_app.exec_()

    app.shutdown()
    sys.exit(result)


if __name__ == "__main__":
    main()
