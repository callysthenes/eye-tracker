"""
SQLite Database access and schema utilities.
Manages sessions, frame-level analytics, mood events, and behavioral data logging.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), 'behavior.sqlite')


def get_connection(db_path=DEFAULT_DB_PATH):
    return sqlite3.connect(db_path)


def init_db(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS event (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            type TEXT,
            value TEXT,
            FOREIGN KEY(session_id) REFERENCES session(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS frame_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            frame_number INTEGER,
            ear_left REAL,
            ear_right REAL,
            ear_avg REAL,
            blink_rate REAL,
            is_blink INTEGER,
            gaze_horizontal REAL,
            gaze_vertical REAL,
            is_on_screen INTEGER,
            mood TEXT,
            mood_confidence REAL,
            mouth_open_amount REAL,
            smile_amount REAL,
            jaw_open_amount REAL,
            brow_raise_amount REAL,
            is_mouth_open INTEGER,
            is_smiling INTEGER,
            is_yawning INTEGER,
            is_brow_raised INTEGER,
            is_eye_squinting INTEGER,
            mar REAL,
            mouth_open_ratio REAL,
            detection_method TEXT,
            FOREIGN KEY(session_id) REFERENCES session(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS mood_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            mood TEXT,
            confidence REAL,
            mood_scores TEXT,
            FOREIGN KEY(session_id) REFERENCES session(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS gesture_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            gesture_type TEXT,
            details TEXT,
            FOREIGN KEY(session_id) REFERENCES session(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS blink_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            blink_number INTEGER,
            ear_at_blink REAL,
            FOREIGN KEY(session_id) REFERENCES session(id)
        )
    ''')

    for idx in [
        'CREATE INDEX IF NOT EXISTS idx_frame_session ON frame_analytics(session_id)',
        'CREATE INDEX IF NOT EXISTS idx_frame_ts ON frame_analytics(timestamp)',
        'CREATE INDEX IF NOT EXISTS idx_mood_session ON mood_events(session_id)',
        'CREATE INDEX IF NOT EXISTS idx_gesture_session ON gesture_events(session_id)',
        'CREATE INDEX IF NOT EXISTS idx_blink_session ON blink_events(session_id)',
        'CREATE INDEX IF NOT EXISTS idx_event_session ON event(session_id)',
    ]:
        c.execute(idx)

    for col, coldef in [('mar', 'REAL'), ('mouth_open_ratio', 'REAL')]:
        try:
            c.execute(f'ALTER TABLE frame_analytics ADD COLUMN {col} {coldef}')
        except Exception:
            pass
    ]:
        c.execute(idx)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")


class DatabaseHandler:

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.current_session_id = None
        init_db(db_path)

    def create_session(self) -> int:
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            start_time = datetime.now().isoformat()
            c.execute('INSERT INTO session (start_time) VALUES (?)', (start_time,))
            conn.commit()
            session_id = c.lastrowid
            conn.close()
            self.current_session_id = session_id
            logger.info(f"Session created with ID: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return -1

    def end_session(self, session_id: Optional[int] = None) -> bool:
        session_id = session_id or self.current_session_id
        if session_id is None:
            return False
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            end_time = datetime.now().isoformat()
            c.execute('UPDATE session SET end_time = ? WHERE id = ?', (end_time, session_id))
            conn.commit()
            conn.close()
            logger.info(f"Session {session_id} ended")
            self.current_session_id = None
            return True
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False

    def log_event(self, event_type: str, value: str, session_id: Optional[int] = None) -> bool:
        session_id = session_id or self.current_session_id
        if session_id is None:
            return False
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            timestamp = datetime.now().isoformat()
            c.execute('INSERT INTO event (session_id, timestamp, type, value) VALUES (?, ?, ?, ?)',
                      (session_id, timestamp, event_type, value))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return False

    def log_frame(self, session_id: int, frame_data: Dict) -> bool:
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO frame_analytics (
                    session_id, timestamp, frame_number,
                    ear_left, ear_right, ear_avg, blink_rate, is_blink,
                    gaze_horizontal, gaze_vertical, is_on_screen,
                    mood, mood_confidence,
                    mouth_open_amount, smile_amount, jaw_open_amount, brow_raise_amount,
                    is_mouth_open, is_smiling, is_yawning, is_brow_raised, is_eye_squinting,
                    mar, mouth_open_ratio, detection_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                datetime.now().isoformat(),
                frame_data.get('frame_number', 0),
                frame_data.get('ear_left', 0.0),
                frame_data.get('ear_right', 0.0),
                frame_data.get('ear_avg', 0.0),
                frame_data.get('blink_rate', 0.0),
                1 if frame_data.get('is_blink', False) else 0,
                frame_data.get('gaze_horizontal', 0.5),
                frame_data.get('gaze_vertical', 0.5),
                1 if frame_data.get('is_on_screen', True) else 0,
                frame_data.get('mood', 'neutral'),
                frame_data.get('mood_confidence', 0.0),
                frame_data.get('mouth_open_amount', 0.0),
                frame_data.get('smile_amount', 0.0),
                frame_data.get('jaw_open_amount', 0.0),
                frame_data.get('brow_raise_amount', 0.0),
                1 if frame_data.get('is_mouth_open', False) else 0,
                1 if frame_data.get('is_smiling', False) else 0,
                1 if frame_data.get('is_yawning', False) else 0,
                1 if frame_data.get('is_brow_raised', False) else 0,
                1 if frame_data.get('is_eye_squinting', False) else 0,
                frame_data.get('mar', 0.0),
                frame_data.get('mouth_open_ratio', 0.0),
                frame_data.get('detection_method', 'none'),
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log frame: {e}")
            return False

    def log_mood_event(self, session_id: int, mood: str, confidence: float, scores: Dict) -> bool:
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO mood_events (session_id, timestamp, mood, confidence, mood_scores)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), mood, confidence, json.dumps(scores)))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log mood event: {e}")
            return False

    def log_gesture_event(self, session_id: int, gesture_type: str, details: str) -> bool:
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO gesture_events (session_id, timestamp, gesture_type, details)
                VALUES (?, ?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), gesture_type, details))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log gesture event: {e}")
            return False

    def log_blink_event(self, session_id: int, blink_number: int, ear: float) -> bool:
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO blink_events (session_id, timestamp, blink_number, ear_at_blink)
                VALUES (?, ?, ?, ?)
            ''', (session_id, datetime.now().isoformat(), blink_number, ear))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log blink event: {e}")
            return False

    def get_session_events(self, session_id: int) -> list:
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            c.execute('SELECT * FROM event WHERE session_id = ? ORDER BY timestamp', (session_id,))
            events = c.fetchall()
            conn.close()
            return events
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            return []

    def get_session_frame_count(self, session_id: int) -> int:
        try:
            conn = get_connection(self.db_path)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM frame_analytics WHERE session_id = ?', (session_id,))
            count = c.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to count frames: {e}")
            return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = DatabaseHandler()
    session_id = handler.create_session()
    handler.log_event("test_event", "This is a test")
    events = handler.get_session_events(session_id)
    print(f"Events: {events}")
    handler.end_session()
