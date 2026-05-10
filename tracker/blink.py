"""
Blink detection using Eye Aspect Ratio (EAR).
Reference: "Real-Time Eye Blink Detection using Facial Landmarks" (Soukupová & Čech, 2016)
"""

import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BlinkDetector:
    """
    Detects blinks using Eye Aspect Ratio (EAR) calculated from facial landmarks.
    Uses 6 landmark points per eye for the EAR formula:
      EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    
    # MediaPipe left eye landmarks (6-point EAR)
    # p1=outer corner, p2=upper-outer, p3=upper-inner,
    # p4=inner corner, p5=lower-inner, p6=lower-outer
    LEFT_EYE_INDICES = [33, 159, 158, 133, 153, 145]
    # MediaPipe right eye landmarks (6-point EAR)
    RIGHT_EYE_INDICES = [263, 386, 385, 362, 380, 374]
    
    def __init__(self, ear_threshold: float = 0.2, blink_frames: int = 3):
        """
        Initialize blink detector.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold for closed eye
            blink_frames: Consecutive frames below threshold to register a blink
        """
        self.ear_threshold = ear_threshold
        self.blink_frames = blink_frames
        
        # State tracking
        self.eye_closed_counter = 0
        self.total_blinks = 0
        self.last_blink_time = None
        
        # For rolling average of blinks/minute
        self.blink_history = deque(maxlen=1800)  # ~30 seconds at 60fps
        self.ear_history = deque(maxlen=30)  # Keep recent EAR values
    
    def eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio (EAR) from eye landmarks.
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Args:
            eye_points: Array of 6 (x, y) points in order:
                        [p1, p2, p3, p4, p5, p6]
                        p1=outer corner, p4=inner corner,
                        p2-3=upper lid, p5-6=lower lid
        
        Returns:
            EAR scalar value (higher = more open, lower = closed)
        """
        if eye_points is None or len(eye_points) < 6:
            return 0.0
        
        try:
            # Euclidean distances
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            
            ear = (A + B) / (2.0 * C) if C > 0 else 0.0
            return float(ear)
        except Exception as e:
            logger.error(f"EAR computation error: {e}")
            return 0.0
    
    def detect(self, landmarks: Optional[np.ndarray], frame_idx: int = 0) -> Dict:
        """
        Detect blinks based on landmarks.
        
        Args:
            landmarks: Full facial landmarks array (468 points, Nx3 with N=468)
            frame_idx: Current frame index (for timestamp tracking)
        
        Returns:
            Dictionary with blink status, EAR values, and stats
        """
        result = {
            'is_blink': False,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'total_blinks': self.total_blinks,
            'is_eye_closed': False,
            'blinks_per_minute': self._compute_blinks_per_minute()
        }
        
        if landmarks is None or len(landmarks) < 468:
            return result
        
        try:
            # Extract eye landmarks
            left_eye = landmarks[self.LEFT_EYE_INDICES, :2]
            right_eye = landmarks[self.RIGHT_EYE_INDICES, :2]
            
            # Compute EAR for both eyes
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            result['left_ear'] = left_ear
            result['right_ear'] = right_ear
            result['avg_ear'] = avg_ear
            
            # Store EAR history
            self.ear_history.append(avg_ear)
            
            # Detect eye closure
            is_closed = avg_ear < self.ear_threshold
            result['is_eye_closed'] = is_closed
            
            if is_closed:
                self.eye_closed_counter += 1
            else:
                # Transition from closed to open = blink detected
                if self.eye_closed_counter >= self.blink_frames:
                    self.total_blinks += 1
                    result['is_blink'] = True
                    self.blink_history.append(frame_idx)
                    logger.debug(f"Blink detected! Total: {self.total_blinks}")
                
                self.eye_closed_counter = 0
            
            result['total_blinks'] = self.total_blinks
            result['blinks_per_minute'] = self._compute_blinks_per_minute()
        
        except Exception as e:
            logger.error(f"Blink detection error: {e}")
        
        return result
    
    def _compute_blinks_per_minute(self) -> float:
        """
        Compute rolling average of blinks per minute (using last 30 seconds of history).
        
        Returns:
            Blinks per minute (float)
        """
        if len(self.blink_history) == 0:
            return 0.0
        
        # Assuming 30 FPS, 1800 frames = 60 seconds
        # Blinks in window / (window_seconds / 60)
        window_size = len(self.blink_history)
        window_seconds = window_size / 30.0  # Assuming 30 FPS
        
        blinks_in_window = len(self.blink_history)
        blinks_per_minute = (blinks_in_window / window_seconds) * 60.0 if window_seconds > 0 else 0.0
        
        return blinks_per_minute
    
    def reset(self):
        """Reset blink counter for new session."""
        self.total_blinks = 0
        self.eye_closed_counter = 0
        self.blink_history.clear()
        self.ear_history.clear()
        logger.info("Blink detector reset.")
    
    def get_eye_closure_duration(self) -> float:
        """
        Estimate continuous eye closure duration (in frames).
        
        Returns:
            Number of consecutive frames with closed eyes
        """
        return float(self.eye_closed_counter)


class BlinkStatistics:
    """
    Aggregate blink statistics over a session/block.
    """
    
    def __init__(self):
        self.blinks = []  # List of blink frame indices
        self.ear_values = []  # History of EAR values
        self.eye_closed_durations = []  # Durations of continuous eye closure
    
    def add_blink(self, frame_idx: int):
        """Record a blink event."""
        self.blinks.append(frame_idx)
    
    def add_ear(self, ear: float):
        """Record an EAR value."""
        self.ear_values.append(ear)
    
    def add_closure_duration(self, duration: float):
        """Record eye closure duration."""
        self.eye_closed_durations.append(duration)
    
    def get_summary(self, duration_seconds: float) -> Dict:
        """
        Get summary statistics.
        
        Args:
            duration_seconds: Total duration of block/session in seconds
        
        Returns:
            Dictionary with summary stats
        """
        blinks_per_minute = (len(self.blinks) / duration_seconds * 60) if duration_seconds > 0 else 0
        avg_ear = np.mean(self.ear_values) if len(self.ear_values) > 0 else 0
        max_closure = max(self.eye_closed_durations) if len(self.eye_closed_durations) > 0 else 0
        
        # PERCLOS: Percentage of eye closure time
        if len(self.eye_closed_durations) > 0:
            total_closure_frames = sum(self.eye_closed_durations)
            perclos = (total_closure_frames / (duration_seconds * 30)) * 100 if duration_seconds > 0 else 0
        else:
            perclos = 0
        
        return {
            'total_blinks': len(self.blinks),
            'blinks_per_minute': blinks_per_minute,
            'average_ear': avg_ear,
            'max_closure_duration_frames': max_closure,
            'perclos': perclos
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    detector = BlinkDetector(ear_threshold=0.2, blink_frames=2)
    
    # Simulate some landmark data
    fake_landmarks = np.random.randn(468, 3)
    
    for i in range(100):
        result = detector.detect(fake_landmarks, frame_idx=i)
        if result['is_blink']:
            print(f"Frame {i}: Blink detected! Total blinks: {result['total_blinks']}")
    
    print(f"Final stats: {detector.get_eye_closure_duration()} frames closed, "
          f"Blinks/min: {result['blinks_per_minute']:.2f}")
