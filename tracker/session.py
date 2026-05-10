"""
Pomodoro/Tomato timer and drowsiness detection logic.
Manages work/rest cycles and generates alerts based on behavioral metrics.
"""

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session state enum."""
    IDLE = "idle"
    WORKING = "working"
    BREAK = "break"
    PAUSED = "paused"


class DrowsinessLevel(Enum):
    """Drowsiness alert levels."""
    ALERT = "alert"  # Normal
    WARNING = "warning"  # Minor concerns
    CRITICAL = "critical"  # Immediate action needed


class PomodoroSession:
    """
    Manages a single Pomodoro work/rest cycle session.
    """
    
    def __init__(self, work_minutes: int = 25, rest_minutes: int = 5):
        """
        Initialize Pomodoro session.
        
        Args:
            work_minutes: Work block duration (default 25)
            rest_minutes: Rest block duration (default 5)
        """
        self.work_duration = work_minutes * 60  # Convert to seconds
        self.rest_duration = rest_minutes * 60
        
        # Session state
        self.state = SessionState.IDLE
        self.block_start_time = None
        self.session_start_time = None
        self.elapsed_in_block = 0.0
        self.block_count = 0
        self.total_rest_taken = 0
        
        # Metrics tracking
        self.blink_stats = {
            'total_blinks': 0,
            'blinks_per_minute': 0.0,
            'avg_ear': 0.0
        }
        
        self.gaze_stats = {
            'on_screen_percent': 0.0,
            'off_screen_duration': 0.0
        }
        
        self.drowsiness_events = []  # List of (timestamp, level, reason)
    
    def start_work_block(self):
        """Start a new work block."""
        self.state = SessionState.WORKING
        self.block_start_time = time.time()
        if self.session_start_time is None:
            self.session_start_time = self.block_start_time
        self.block_count += 1
        logger.info(f"Starting work block #{self.block_count}")
    
    def start_rest_block(self):
        """Start a rest block."""
        self.state = SessionState.BREAK
        self.block_start_time = time.time()
        self.total_rest_taken += 1
        logger.info(f"Starting rest block")
    
    def pause_session(self):
        """Pause the current session."""
        self.state = SessionState.PAUSED
        logger.info("Session paused")
    
    def resume_session(self):
        """Resume the paused session."""
        if self.state == SessionState.PAUSED:
            self.block_start_time = time.time()
            self.state = SessionState.WORKING
            logger.info("Session resumed")
    
    def get_time_remaining(self) -> float:
        """
        Get remaining time in current block (seconds).
        
        Returns:
            Seconds remaining
        """
        if self.block_start_time is None:
            return 0.0
        
        elapsed = time.time() - self.block_start_time
        
        if self.state == SessionState.WORKING:
            remaining = max(0, self.work_duration - elapsed)
        elif self.state == SessionState.BREAK:
            remaining = max(0, self.rest_duration - elapsed)
        else:
            remaining = 0.0
        
        return remaining
    
    def is_block_complete(self) -> bool:
        """Check if current block time has expired."""
        remaining = self.get_time_remaining()
        return remaining <= 0
    
    def get_status(self) -> Dict:
        """
        Get current session status.
        
        Returns:
            Dictionary with session state and metrics
        """
        elapsed_total = (time.time() - self.session_start_time) if self.session_start_time else 0
        
        return {
            'state': self.state.value,
            'block_number': self.block_count,
            'blocks_complete': max(0, self.block_count - 1),
            'time_remaining': self.get_time_remaining(),
            'time_elapsed_total': elapsed_total,
            'total_rest_taken': self.total_rest_taken,
            'blink_stats': self.blink_stats.copy(),
            'gaze_stats': self.gaze_stats.copy(),
            'drowsiness_events': self.drowsiness_events.copy()
        }


class DrowsinessDetector:
    """
    Detects drowsiness based on blink rate, eye closure, and gaze patterns.
    Inspired by Tesla's Driver Monitoring System.
    """
    
    def __init__(
        self,
        low_blink_threshold: float = 8.0,  # Blinks per minute
        high_perclos_threshold: float = 80.0,  # Percentage of eye closure
        off_screen_timeout: float = 3.0  # Seconds looking away
    ):
        """
        Initialize drowsiness detector.
        
        Args:
            low_blink_threshold: Blinks/min below which is suspicious
            high_perclos_threshold: PERCLOS % above which is drowsy
            off_screen_timeout: Seconds of off-screen gaze to trigger alert
        """
        self.low_blink_threshold = low_blink_threshold
        self.high_perclos_threshold = high_perclos_threshold
        self.off_screen_timeout = off_screen_timeout
        
        # Tracking
        self.drowsiness_score = 0.0  # 0-100
        self.last_alert_time = None
        self.consecutive_warning_frames = 0
    
    def evaluate(self, blink_stats: Dict, gaze_stats: Dict, ear: float) -> Dict:
        """
        Evaluate drowsiness from multiple metrics.
        
        Args:
            blink_stats: Blink statistics dict (blinks_per_minute, etc.)
            gaze_stats: Gaze statistics dict (on_screen_percent, etc.)
            ear: Current eye aspect ratio (0-1)
        
        Returns:
            Dictionary with drowsiness level, score, and reason
        """
        result = {
            'level': DrowsinessLevel.ALERT,
            'score': 0.0,
            'reason': "",
            'factors': {}
        }
        
        score = 0.0
        reasons = []
        factors = {}
        
        # Factor 1: Low blink rate
        blinks_per_min = blink_stats.get('blinks_per_minute', 0.0)
        if blinks_per_min < self.low_blink_threshold:
            blink_factor = (self.low_blink_threshold - blinks_per_min) / self.low_blink_threshold
            score += 20 * blink_factor
            reasons.append(f"Low blink rate: {blinks_per_min:.1f}/min")
            factors['low_blinks'] = blink_factor
        
        # Factor 2: High PERCLOS (percentage of eye closure)
        perclos = blink_stats.get('perclos', 0.0)
        if perclos > self.high_perclos_threshold:
            perclos_factor = (perclos - self.high_perclos_threshold) / (100 - self.high_perclos_threshold)
            score += 30 * perclos_factor
            reasons.append(f"High eye closure: {perclos:.1f}%")
            factors['high_perclos'] = perclos_factor
        
        # Factor 3: Looking away from screen
        on_screen = gaze_stats.get('on_screen_percent', 100.0)
        if on_screen < 70:  # Less than 70% looking at screen
            look_away_factor = (100 - on_screen) / 30
            score += 15 * look_away_factor
            reasons.append(f"Low screen attention: {on_screen:.1f}%")
            factors['off_screen'] = look_away_factor
        
        # Factor 4: Very low EAR (extreme eye closure)
        if ear < 0.1:
            score += 25
            reasons.append("Extreme eye closure detected")
            factors['extreme_closure'] = 1.0
        
        result['score'] = min(100.0, score)
        result['factors'] = factors
        
        # Determine level
        if score >= 70:
            result['level'] = DrowsinessLevel.CRITICAL
            result['reason'] = "CRITICAL: " + " | ".join(reasons)
        elif score >= 40:
            result['level'] = DrowsinessLevel.WARNING
            result['reason'] = "WARNING: " + " | ".join(reasons)
        else:
            result['level'] = DrowsinessLevel.ALERT
            result['reason'] = "ALERT: " + " | ".join(reasons) if reasons else "Normal"
        
        return result
    
    def should_trigger_alert(self, drowsiness_result: Dict, min_interval_seconds: float = 5.0) -> bool:
        """
        Determine if an alert should be triggered.
        
        Args:
            drowsiness_result: Result dict from evaluate()
            min_interval_seconds: Minimum time between alerts
        
        Returns:
            True if alert should trigger
        """
        if drowsiness_result['level'] == DrowsinessLevel.CRITICAL:
            now = time.time()
            if self.last_alert_time is None or (now - self.last_alert_time) > min_interval_seconds:
                self.last_alert_time = now
                return True
        
        return False


class SessionManager:
    """
    Manages overall session lifecycle: work blocks, rest, drowsiness, DB logging.
    """
    
    def __init__(
        self,
        work_minutes: int = 25,
        rest_minutes: int = 5,
        db_handler=None
    ):
        """
        Initialize session manager.
        
        Args:
            work_minutes: Work block duration
            rest_minutes: Rest block duration
            db_handler: Optional database handler for logging
        """
        self.pomodoro = PomodoroSession(work_minutes, rest_minutes)
        self.drowsiness_detector = DrowsinessDetector()
        self.db_handler = db_handler
        
        # Block statistics accumulation
        self.current_block_blink_stats = {
            'total_blinks': 0,
            'blinks_per_minute': 0.0,
            'avg_ear': 0.0,
            'perclos': 0.0
        }
        
        self.current_block_gaze_stats = {
            'on_screen_percent': 0.0,
            'off_screen_duration': 0.0
        }
    
    def update(self, blink_data: Dict, gaze_data: Dict) -> Dict:
        """
        Update session with new blink and gaze data.
        
        Args:
            blink_data: Blink detection result
            gaze_data: Gaze estimation result
        
        Returns:
            Dictionary with session update (alerts, state changes, etc.)
        """
        update_result = {
            'state_changed': False,
            'rest_block_due': False,
            'drowsiness_alert': None,
            'session_summary': None
        }
        
        # Update blink stats
        self.current_block_blink_stats['total_blinks'] = blink_data.get('total_blinks', 0)
        self.current_block_blink_stats['blinks_per_minute'] = blink_data.get('blinks_per_minute', 0.0)
        self.current_block_blink_stats['avg_ear'] = blink_data.get('avg_ear', 0.0)
        self.current_block_blink_stats['perclos'] = blink_data.get('perclos', 0.0)
        
        # Update gaze stats
        self.current_block_gaze_stats['on_screen_percent'] = gaze_data.get('on_screen_percent', 0.0)
        self.current_block_gaze_stats['off_screen_duration'] = gaze_data.get('off_screen_duration', 0.0)
        
        # Check drowsiness
        drowsiness = self.drowsiness_detector.evaluate(
            self.current_block_blink_stats,
            self.current_block_gaze_stats,
            blink_data.get('avg_ear', 0.0)
        )
        
        # Log drowsiness event if alert
        if self.drowsiness_detector.should_trigger_alert(drowsiness):
            self.pomodoro.drowsiness_events.append({
                'timestamp': datetime.now().isoformat(),
                'level': drowsiness['level'].value,
                'reason': drowsiness['reason']
            })
            update_result['drowsiness_alert'] = drowsiness
            
            # Log to DB if handler available
            if self.db_handler:
                self.db_handler.log_event(
                    event_type='drowsiness_alert',
                    value=drowsiness['reason']
                )
        
        # Check if work block is complete
        if self.pomodoro.is_block_complete():
            if self.pomodoro.state == SessionState.WORKING:
                update_result['rest_block_due'] = True
                update_result['state_changed'] = True
                self._finalize_work_block()
                self.pomodoro.start_rest_block()
            elif self.pomodoro.state == SessionState.BREAK:
                update_result['state_changed'] = True
                self._finalize_rest_block()
                self.pomodoro.start_work_block()
        
        return update_result
    
    def _finalize_work_block(self):
        """Finalize work block and log stats."""
        self.pomodoro.blink_stats = self.current_block_blink_stats.copy()
        self.pomodoro.gaze_stats = self.current_block_gaze_stats.copy()
        
        logger.info(f"Work block #{self.pomodoro.block_count} complete. "
                   f"Blinks: {self.current_block_blink_stats['blinks_per_minute']:.1f}/min")
        
        # Log to DB
        if self.db_handler:
            self.db_handler.log_event(
                event_type='block_complete',
                value=f"Work block, blinks: {self.current_block_blink_stats['blinks_per_minute']:.1f}/min"
            )
    
    def _finalize_rest_block(self):
        """Finalize rest block."""
        logger.info("Rest block complete, ready for next work block")
        
        if self.db_handler:
            self.db_handler.log_event(
                event_type='rest_complete',
                value="Rest period completed"
            )
    
    def get_session_status(self) -> Dict:
        """Get current session status."""
        return self.pomodoro.get_status()
    
    def start_session(self):
        """Start a new session."""
        self.pomodoro.start_work_block()
        if self.db_handler:
            self.db_handler.log_event(
                event_type='session_start',
                value="Eye tracking session started"
            )
        logger.info("Session started")
    
    def end_session(self) -> Dict:
        """
        End session and return final summary.
        
        Returns:
            Summary statistics for the entire session
        """
        status = self.get_session_status()
        
        if self.db_handler:
            self.db_handler.log_event(
                event_type='session_end',
                value=f"Session ended. Blocks: {status['blocks_complete']}, Rest taken: {status['total_rest_taken']}"
            )
        
        logger.info(f"Session ended. Blocks completed: {status['blocks_complete']}, "
                   f"Rest breaks taken: {status['total_rest_taken']}")
        
        return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Quick test
    manager = SessionManager()
    manager.start_session()
    
    # Simulate some data
    for i in range(5):
        blink_data = {
            'total_blinks': i * 2,
            'blinks_per_minute': 15.0,
            'avg_ear': 0.5,
            'perclos': 20.0
        }
        
        gaze_data = {
            'on_screen_percent': 85.0,
            'off_screen_duration': 0.5
        }
        
        result = manager.update(blink_data, gaze_data)
        status = manager.get_session_status()
        print(f"Status: {status['state']}, Time remaining: {status['time_remaining']:.0f}s")
    
    summary = manager.end_session()
    print(f"Final summary: {summary}")
