"""
Gaze estimation: determines if user is looking at screen or away.
Uses eye landmarks and simple heuristics, or can integrate with dedicated gaze models.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class GazeDirection(Enum):
    """Enum for gaze direction categories."""
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    AWAY = "away"  # User looking away from screen


class GazeEstimator:
    """
    Estimates gaze direction from eye landmarks.
    Uses iris center and eye bounding box to classify gaze direction.
    """
    
    # MediaPipe eye landmarks
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 155]
    RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 382]
    
    # Iris points (smaller circle inside eye)
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    def __init__(self, horizontal_threshold: float = 0.3, vertical_threshold: float = 0.25):
        """
        Initialize gaze estimator.
        
        Args:
            horizontal_threshold: Threshold for left/right classification
            vertical_threshold: Threshold for up/down classification
        """
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold
        
        # History for smoothing
        self.gaze_history = deque(maxlen=10)
        self.gaze_confidence_history = deque(maxlen=10)
    
    def estimate_gaze_direction(self, landmarks: Optional[np.ndarray]) -> Dict:
        """
        Estimate gaze direction from facial landmarks.
        
        Args:
            landmarks: Full facial landmarks (468 points)
        
        Returns:
            Dictionary with gaze direction, confidence, and normalized position
        """
        result = {
            'direction': GazeDirection.CENTER,
            'horizontal_ratio': 0.5,  # 0 = far left, 1 = far right
            'vertical_ratio': 0.5,    # 0 = up, 1 = down
            'confidence': 0.0,
            'is_on_screen': True,
            'gaze_vector': None
        }
        
        if landmarks is None or len(landmarks) < 478:
            result['confidence'] = 0.0
            return result
        
        try:
            # Extract left eye iris center
            left_iris = landmarks[self.LEFT_IRIS_INDICES, :2]
            left_eye_box = landmarks[self.LEFT_EYE_INDICES, :2]
            
            # Extract right eye iris center
            right_iris = landmarks[self.RIGHT_IRIS_INDICES, :2]
            right_eye_box = landmarks[self.RIGHT_EYE_INDICES, :2]
            
            # Compute normalized positions (0-1 within eye bounding box)
            left_h_ratio = self._compute_horizontal_ratio(left_iris, left_eye_box)
            left_v_ratio = self._compute_vertical_ratio(left_iris, left_eye_box)
            
            right_h_ratio = self._compute_horizontal_ratio(right_iris, right_eye_box)
            right_v_ratio = self._compute_vertical_ratio(right_iris, right_eye_box)
            
            # Average both eyes
            h_ratio = (left_h_ratio + right_h_ratio) / 2.0
            v_ratio = (left_v_ratio + right_v_ratio) / 2.0
            
            result['horizontal_ratio'] = h_ratio
            result['vertical_ratio'] = v_ratio
            
            # Classify direction
            direction, confidence = self._classify_direction(h_ratio, v_ratio)
            result['direction'] = direction
            result['confidence'] = confidence
            
            # Simple heuristic: if looking too far away, mark as "not on screen"
            is_on_screen = self._is_on_screen(h_ratio, v_ratio)
            result['is_on_screen'] = is_on_screen
            
            # Compute gaze vector (simplified)
            gaze_vector = np.array([h_ratio - 0.5, v_ratio - 0.5])
            result['gaze_vector'] = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-6)
            
            # Store in history for smoothing
            self.gaze_history.append(direction)
            self.gaze_confidence_history.append(confidence)
        
        except Exception as e:
            logger.error(f"Gaze estimation error: {e}")
            result['confidence'] = 0.0
        
        return result
    
    def _compute_horizontal_ratio(self, iris: np.ndarray, eye_box: np.ndarray) -> float:
        """
        Compute horizontal ratio (0 = left, 1 = right).
        
        Args:
            iris: Iris center point (x, y)
            eye_box: Eye bounding box points
        
        Returns:
            Normalized ratio 0-1
        """
        if len(iris) < 1 or len(eye_box) < 2:
            return 0.5
        
        iris_center = iris.mean(axis=0)
        eye_left = eye_box[:, 0].min()
        eye_right = eye_box[:, 0].max()
        
        eye_width = eye_right - eye_left
        if eye_width <= 0:
            return 0.5
        
        ratio = (iris_center[0] - eye_left) / eye_width
        return np.clip(ratio, 0.0, 1.0)
    
    def _compute_vertical_ratio(self, iris: np.ndarray, eye_box: np.ndarray) -> float:
        """
        Compute vertical ratio (0 = up, 1 = down).
        
        Args:
            iris: Iris center point (x, y)
            eye_box: Eye bounding box points
        
        Returns:
            Normalized ratio 0-1
        """
        if len(iris) < 1 or len(eye_box) < 2:
            return 0.5
        
        iris_center = iris.mean(axis=0)
        eye_top = eye_box[:, 1].min()
        eye_bottom = eye_box[:, 1].max()
        
        eye_height = eye_bottom - eye_top
        if eye_height <= 0:
            return 0.5
        
        ratio = (iris_center[1] - eye_top) / eye_height
        return np.clip(ratio, 0.0, 1.0)
    
    def _classify_direction(self, h_ratio: float, v_ratio: float) -> Tuple[GazeDirection, float]:
        """
        Classify gaze direction from horizontal and vertical ratios.
        
        Args:
            h_ratio: Horizontal ratio (0-1)
            v_ratio: Vertical ratio (0-1)
        
        Returns:
            Tuple of (GazeDirection, confidence score)
        """
        # Measure distance from center
        h_dist = abs(h_ratio - 0.5)
        v_dist = abs(v_ratio - 0.5)
        
        center_threshold = 0.15
        
        # If both are near center
        if h_dist < center_threshold and v_dist < center_threshold:
            return GazeDirection.CENTER, 0.8
        
        # Primary direction (whichever is further from center)
        if h_dist > v_dist:
            direction = GazeDirection.LEFT if h_ratio < 0.5 else GazeDirection.RIGHT
        else:
            direction = GazeDirection.UP if v_ratio < 0.5 else GazeDirection.DOWN
        
        # Confidence based on deviation
        max_dist = max(h_dist, v_dist)
        confidence = min(1.0, max_dist / 0.4)  # Max confidence at 40% deviation
        
        return direction, confidence
    
    def _is_on_screen(self, h_ratio: float, v_ratio: float) -> bool:
        """
        Heuristic: determine if user is looking at screen.
        Simple version: center ±40% horizontally and ±30% vertically.
        """
        on_h = 0.1 <= h_ratio <= 0.9
        on_v = 0.1 <= v_ratio <= 0.9
        return on_h and on_v
    
    def get_smoothed_direction(self) -> GazeDirection:
        """
        Get smoothed gaze direction from history.
        
        Returns:
            Most common direction in recent history
        """
        if len(self.gaze_history) == 0:
            return GazeDirection.CENTER
        
        # Mode of the direction history
        from collections import Counter
        counts = Counter(self.gaze_history)
        most_common = counts.most_common(1)[0][0]
        return most_common


class GazeStatistics:
    """
    Aggregate gaze statistics over a session/block.
    """
    
    def __init__(self):
        self.direction_counts = {
            'center': 0,
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0,
            'away': 0
        }
        self.on_screen_frames = 0
        self.off_screen_frames = 0
    
    def add_gaze_sample(self, direction: GazeDirection, is_on_screen: bool):
        """Record a gaze sample."""
        self.direction_counts[direction.value] += 1
        if is_on_screen:
            self.on_screen_frames += 1
        else:
            self.off_screen_frames += 1
    
    def get_summary(self, duration_seconds: float) -> Dict:
        """
        Get summary statistics.
        
        Args:
            duration_seconds: Duration of block in seconds
        
        Returns:
            Dictionary with gaze stats
        """
        total_frames = self.on_screen_frames + self.off_screen_frames
        
        on_screen_percent = (self.on_screen_frames / total_frames * 100) if total_frames > 0 else 0
        
        summary = {
            'on_screen_percent': on_screen_percent,
            'off_screen_percent': 100 - on_screen_percent,
            'direction_breakdown': self.direction_counts.copy(),
            'total_samples': total_frames
        }
        
        return summary


class OffScreenTracker:
    """
    Tracks continuous off-screen gaze time.
    Used for drowsiness detection (user staring away).
    """
    
    def __init__(self, off_screen_timeout: float = 3.0):
        """
        Initialize off-screen tracker.
        
        Args:
            off_screen_timeout: Seconds of off-screen gaze to trigger alert
        """
        self.off_screen_timeout = off_screen_timeout
        self.off_screen_start_time = None
        self.off_screen_duration = 0.0
        self.is_off_screen_alert = False
    
    def update(self, is_on_screen: bool, elapsed_seconds: float):
        """
        Update off-screen tracking state.
        
        Args:
            is_on_screen: Whether user is looking at screen
            elapsed_seconds: Time elapsed in current session (for tracking)
        
        Returns:
            Dictionary with alert status
        """
        result = {
            'is_alert': False,
            'off_screen_duration': 0.0
        }
        
        if not is_on_screen:
            if self.off_screen_start_time is None:
                self.off_screen_start_time = elapsed_seconds
            
            self.off_screen_duration = elapsed_seconds - self.off_screen_start_time
            
            if self.off_screen_duration > self.off_screen_timeout:
                self.is_off_screen_alert = True
                result['is_alert'] = True
        else:
            # Reset when back on screen
            self.off_screen_start_time = None
            self.off_screen_duration = 0.0
            self.is_off_screen_alert = False
        
        result['off_screen_duration'] = self.off_screen_duration
        return result
    
    def reset(self):
        """Reset off-screen tracker."""
        self.off_screen_start_time = None
        self.off_screen_duration = 0.0
        self.is_off_screen_alert = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    estimator = GazeEstimator()
    
    # Simulate landmarks
    fake_landmarks = np.random.randn(478, 3)
    
    result = estimator.estimate_gaze_direction(fake_landmarks)
    print(f"Gaze direction: {result['direction']}, Confidence: {result['confidence']:.2f}")
    print(f"On screen: {result['is_on_screen']}")
