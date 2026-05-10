"""
CSV export functionality for session analytics.
Exports frame-level data, mood timelines, gesture events, and session summaries.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).parent / 'behavior.sqlite')


class CSVExporter:

    def __init__(self, db_path: str = DB_PATH, export_dir: str = "exports"):
        self.db_path = db_path
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)

    def export_session_summary(self, session_data: Dict) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.export_dir / f"session_summary_{timestamp}.csv"
            flattened = self._flatten_session_data(session_data)
            df = pd.DataFrame([flattened])
            df.to_csv(filename, index=False)
            logger.info(f"Session summary exported to {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export session summary: {e}")
            return ""

    def export_frame_data(self, session_id: Optional[int] = None) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            if session_id:
                query = "SELECT * FROM frame_analytics WHERE session_id = ? ORDER BY timestamp"
                df = pd.read_sql_query(query, conn, params=(session_id,))
            else:
                query = "SELECT * FROM frame_analytics ORDER BY timestamp"
                df = pd.read_sql_query(query, conn)
            conn.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_str = f"_session{session_id}" if session_id else "_all"
            filename = self.export_dir / f"frame_data{session_str}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Frame data exported to {filename} ({len(df)} rows)")
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export frame data: {e}")
            return ""

    def export_mood_timeline(self, session_id: Optional[int] = None) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            if session_id:
                query = "SELECT * FROM mood_events WHERE session_id = ? ORDER BY timestamp"
                df = pd.read_sql_query(query, conn, params=(session_id,))
            else:
                query = "SELECT * FROM mood_events ORDER BY timestamp"
                df = pd.read_sql_query(query, conn)
            conn.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_str = f"_session{session_id}" if session_id else "_all"
            filename = self.export_dir / f"mood_timeline{session_str}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Mood timeline exported to {filename} ({len(df)} rows)")
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export mood timeline: {e}")
            return ""

    def export_gesture_events(self, session_id: Optional[int] = None) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            if session_id:
                query = "SELECT * FROM gesture_events WHERE session_id = ? ORDER BY timestamp"
                df = pd.read_sql_query(query, conn, params=(session_id,))
            else:
                query = "SELECT * FROM gesture_events ORDER BY timestamp"
                df = pd.read_sql_query(query, conn)
            conn.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_str = f"_session{session_id}" if session_id else "_all"
            filename = self.export_dir / f"gestures{session_str}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Gesture events exported to {filename} ({len(df)} rows)")
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export gesture events: {e}")
            return ""

    def export_blink_events(self, session_id: Optional[int] = None) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            if session_id:
                query = "SELECT * FROM blink_events WHERE session_id = ? ORDER BY timestamp"
                df = pd.read_sql_query(query, conn, params=(session_id,))
            else:
                query = "SELECT * FROM blink_events ORDER BY timestamp"
                df = pd.read_sql_query(query, conn)
            conn.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_str = f"_session{session_id}" if session_id else "_all"
            filename = self.export_dir / f"blinks{session_str}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Blink events exported to {filename} ({len(df)} rows)")
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export blink events: {e}")
            return ""

    def export_event_log(self, session_id: Optional[int] = None) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            if session_id:
                query = "SELECT * FROM event WHERE session_id = ? ORDER BY timestamp"
                df = pd.read_sql_query(query, conn, params=(session_id,))
            else:
                query = "SELECT * FROM event ORDER BY timestamp"
                df = pd.read_sql_query(query, conn)
            conn.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_str = f"_session{session_id}" if session_id else "_all"
            filename = self.export_dir / f"events{session_str}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export event log: {e}")
            return ""

    def export_all_sessions(self) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT
                s.id as session_id, s.start_time, s.end_time,
                COUNT(DISTINCT fa.id) as total_frames,
                COUNT(DISTINCT be.id) as total_blinks,
                COUNT(DISTINCT ge.id) as total_gestures,
                COUNT(DISTINCT me.id) as total_mood_changes,
                AVG(fa.ear_avg) as avg_ear,
                AVG(fa.blink_rate) as avg_blink_rate,
                AVG(fa.mood_confidence) as avg_mood_confidence,
                SUM(CASE WHEN fa.is_on_screen = 1 THEN 1 ELSE 0 END) as on_screen_frames,
                SUM(CASE WHEN fa.is_smiling = 1 THEN 1 ELSE 0 END) as smiling_frames,
                SUM(CASE WHEN fa.is_mouth_open = 1 THEN 1 ELSE 0 END) as mouth_open_frames,
                SUM(CASE WHEN fa.is_yawning = 1 THEN 1 ELSE 0 END) as yawning_frames,
                COUNT(CASE WHEN e.type = 'drowsiness_alert' THEN 1 END) as drowsiness_alerts,
                COUNT(CASE WHEN e.type = 'block_complete' THEN 1 END) as blocks_completed
            FROM session s
            LEFT JOIN frame_analytics fa ON s.id = fa.session_id
            LEFT JOIN blink_events be ON s.id = be.session_id
            LEFT JOIN gesture_events ge ON s.id = ge.session_id
            LEFT JOIN mood_events me ON s.id = me.session_id
            LEFT JOIN event e ON s.id = e.session_id
            GROUP BY s.id
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.export_dir / f"all_sessions_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"All sessions exported to {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export all sessions: {e}")
            return ""

    def export_full_analytics(self, session_id: Optional[int] = None) -> List[str]:
        exported = []
        exported.append(self.export_frame_data(session_id))
        exported.append(self.export_mood_timeline(session_id))
        exported.append(self.export_gesture_events(session_id))
        exported.append(self.export_blink_events(session_id))
        exported.append(self.export_event_log(session_id))
        if not session_id:
            exported.append(self.export_all_sessions())
        return [p for p in exported if p]

    def _flatten_session_data(self, session_data: Dict) -> Dict:
        flattened = {
            'timestamp': datetime.now().isoformat(),
            'state': session_data.get('state', ''),
            'block_number': session_data.get('block_number', 0),
            'blocks_complete': session_data.get('blocks_complete', 0),
            'time_elapsed_total': session_data.get('time_elapsed_total', 0),
            'total_rest_taken': session_data.get('total_rest_taken', 0),
        }
        blink_stats = session_data.get('blink_stats', {})
        flattened.update({
            'blink_total': blink_stats.get('total_blinks', 0),
            'blink_per_minute': blink_stats.get('blinks_per_minute', 0.0),
            'avg_ear': blink_stats.get('avg_ear', 0.0),
        })
        gaze_stats = session_data.get('gaze_stats', {})
        flattened.update({
            'on_screen_percent': gaze_stats.get('on_screen_percent', 0.0),
            'off_screen_duration': gaze_stats.get('off_screen_duration', 0.0),
        })
        return flattened


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exporter = CSVExporter()
    sample = {
        'state': 'working', 'block_number': 1, 'blocks_complete': 0,
        'time_elapsed_total': 120.0, 'total_rest_taken': 0,
        'blink_stats': {'total_blinks': 15, 'blinks_per_minute': 18.0, 'avg_ear': 0.45},
        'gaze_stats': {'on_screen_percent': 92.0, 'off_screen_duration': 0.5},
    }
    path = exporter.export_session_summary(sample)
    print(f"Exported to: {path}")
