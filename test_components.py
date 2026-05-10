#!/usr/bin/env python3
"""
Quick test script to validate Eye Tracker components.
Runs without needing full YOLO/MediaPipe installation.
"""

import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all core modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from gui.window import EyeTrackerWindow
        logger.info("✓ GUI module imports OK")
    except Exception as e:
        logger.error(f"✗ GUI import failed: {e}")
        return False
    
    try:
        from tracker.blink import BlinkDetector
        logger.info("✓ Blink detector imports OK")
    except Exception as e:
        logger.error(f"✗ Blink detector import failed: {e}")
        return False
    
    try:
        from tracker.gaze import GazeEstimator
        logger.info("✓ Gaze estimator imports OK")
    except Exception as e:
        logger.error(f"✗ Gaze estimator import failed: {e}")
        return False
    
    try:
        from tracker.session import SessionManager
        logger.info("✓ Session manager imports OK")
    except Exception as e:
        logger.error(f"✗ Session manager import failed: {e}")
        return False
    
    try:
        from database.db import DatabaseHandler
        logger.info("✓ Database handler imports OK")
    except Exception as e:
        logger.error(f"✗ Database handler import failed: {e}")
        return False
    
    try:
        from database.export import CSVExporter
        logger.info("✓ CSV exporter imports OK")
    except Exception as e:
        logger.error(f"✗ CSV exporter import failed: {e}")
        return False
    
    try:
        from tracker.detector import DetectionPipeline, WebcamCapture
        logger.info("✓ Detection pipeline imports OK (with fallback support)")
    except Exception as e:
        logger.error(f"✗ Detection pipeline import failed: {e}")
        return False
    
    return True


def test_blink_detector():
    """Test blink detector with fake landmarks."""
    logger.info("\nTesting blink detector...")
    
    from tracker.blink import BlinkDetector
    
    detector = BlinkDetector(ear_threshold=0.2)
    
    # Create fake landmarks
    fake_landmarks = np.random.randn(468, 3) * 100
    
    # Simulate some frames
    for i in range(10):
        result = detector.detect(fake_landmarks, frame_idx=i)
        assert 'avg_ear' in result
        assert 'blinks_per_minute' in result
    
    logger.info(f"✓ Blink detector works (detected {detector.total_blinks} blinks)")
    return True


def test_gaze_estimator():
    """Test gaze estimator with fake landmarks."""
    logger.info("\nTesting gaze estimator...")
    
    from tracker.gaze import GazeEstimator
    
    estimator = GazeEstimator()
    
    # Create fake landmarks
    fake_landmarks = np.random.randn(478, 3) * 100
    
    result = estimator.estimate_gaze_direction(fake_landmarks)
    assert 'direction' in result
    assert 'confidence' in result
    assert 'is_on_screen' in result
    
    logger.info(f"✓ Gaze estimator works (direction: {result['direction']})")
    return True


def test_session_manager():
    """Test session manager."""
    logger.info("\nTesting session manager...")
    
    from tracker.session import SessionManager
    
    manager = SessionManager(work_minutes=1, rest_minutes=1)
    manager.start_session()
    
    # Simulate some data
    for i in range(3):
        blink_data = {
            'total_blinks': i,
            'blinks_per_minute': 15.0,
            'avg_ear': 0.45,
            'perclos': 20.0
        }
        
        gaze_data = {
            'on_screen_percent': 85.0,
            'off_screen_duration': 0.0
        }
        
        result = manager.update(blink_data, gaze_data)
        assert 'state_changed' in result
    
    status = manager.get_session_status()
    logger.info(f"✓ Session manager works (state: {status['state']})")
    return True


def test_database():
    """Test database operations."""
    logger.info("\nTesting database...")
    
    from database.db import DatabaseHandler
    import os
    import tempfile
    
    # Create temp DB
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        handler = DatabaseHandler(db_path)
        
        session_id = handler.create_session()
        assert session_id > 0
        
        success = handler.log_event("test_event", "Test value", session_id)
        assert success
        
        events = handler.get_session_events(session_id)
        assert len(events) > 0
        
        handler.end_session(session_id)
    
    logger.info("✓ Database works")
    return True


def test_csv_export():
    """Test CSV export."""
    logger.info("\nTesting CSV export...")
    
    from database.export import CSVExporter
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = CSVExporter(export_dir=tmpdir)
        
        sample_data = {
            'state': 'working',
            'block_number': 1,
            'blocks_complete': 0,
            'time_elapsed_total': 120.0,
            'total_rest_taken': 0,
            'blink_stats': {'total_blinks': 10, 'blinks_per_minute': 15.0, 'avg_ear': 0.4},
            'gaze_stats': {'on_screen_percent': 90.0, 'off_screen_duration': 0.0},
            'drowsiness_events': []
        }
        
        path = exporter.export_session_summary(sample_data)
        assert os.path.exists(path)
    
    logger.info("✓ CSV export works")
    return True


def test_detection_fallback():
    """Test detection pipeline with fallback."""
    logger.info("\nTesting detection pipeline with cascade fallback...")
    
    from tracker.detector import YOLODetector
    import cv2
    
    detector = YOLODetector(use_openvino=False)
    
    # Create a fake frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    result = detector.detect(frame)
    assert 'faces' in result
    assert 'eyes' in result
    
    logger.info("✓ Detection fallback works")
    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Eye Tracker Component Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Blink Detector", test_blink_detector),
        ("Gaze Estimator", test_gaze_estimator),
        ("Session Manager", test_session_manager),
        ("Database", test_database),
        ("CSV Export", test_csv_export),
        ("Detection Fallback", test_detection_fallback),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}", exc_info=True)
            results.append((name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    sys.exit(0 if passed == total else 1)
