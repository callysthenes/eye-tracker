"""
Facial expression, mood, and gesture analysis.
Uses three complementary signals:
  1. MediaPipe blendshapes (52 face coefficients)
  2. Landmark-based metrics (MAR - mouth aspect ratio, brow height, etc.)
  3. Haar cascade mouth detection (supplementary)
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Mood(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    EXCITED = "excited"
    DROWSY = "drowsy"


BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
]


def _bs(bs_dict: Dict[str, float], name: str) -> float:
    return bs_dict.get(name, 0.0)


# --- Landmark-based metric computation ---

# MediaPipe 478-point mouth landmarks
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
MOUTH_INNER_LOWER = [87, 178, 88, 95, 308, 324, 318, 402, 317, 14]
MOUTH_CORNERS = [61, 291]
LIP_UPPER = [13, 82, 312]
LIP_LOWER = [14, 87, 317]

# Brow landmarks
BROW_LEFT = [70, 63, 105, 66, 107]
BROW_RIGHT = [300, 293, 334, 296, 336]
EYE_TOP_LEFT = [159]
EYE_TOP_RIGHT = [386]


class LandmarkMetrics:
    """
    Computes facial metrics directly from landmark coordinates.
    More robust than blendshapes for mouth open, brow raise, etc.
    """

    @staticmethod
    def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
        if landmarks is None or len(landmarks) < 478:
            return 0.0
        try:
            upper = landmarks[13, :2]
            lower = landmarks[14, :2]
            left = landmarks[61, :2]
            right = landmarks[291, :2]
            vertical = np.linalg.norm(upper - lower)
            horizontal = np.linalg.norm(left - right)
            return float(vertical / horizontal) if horizontal > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def mouth_open_ratio(landmarks: np.ndarray) -> float:
        if landmarks is None or len(landmarks) < 478:
            return 0.0
        try:
            upper_lip = landmarks[[13, 82, 312], 1].mean()
            lower_lip = landmarks[[14, 87, 317], 1].mean()
            left_corner = landmarks[61, 1]
            right_corner = landmarks[291, 1]
            corners_y = (left_corner + right_corner) / 2
            face_height = np.linalg.norm(landmarks[152, :2] - landmarks[10, :2])
            if face_height <= 0:
                return 0.0
            open_dist = lower_lip - upper_lip
            return float(open_dist / face_height)
        except Exception:
            return 0.0

    @staticmethod
    def smile_ratio(landmarks: np.ndarray) -> float:
        if landmarks is None or len(landmarks) < 478:
            return 0.0
        try:
            left_corner = landmarks[61, :2]
            right_corner = landmarks[291, :2]
            upper_lip_center = landmarks[13, :2]
            mouth_width = np.linalg.norm(right_corner - left_corner)
            corner_mid = (left_corner + right_corner) / 2
            corner_to_lip = corner_mid[1] - upper_lip_center[1]
            return float(corner_to_lip / mouth_width) if mouth_width > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def brow_raise_ratio(landmarks: np.ndarray) -> float:
        if landmarks is None or len(landmarks) < 478:
            return 0.0
        try:
            left_brow_y = landmarks[BROW_LEFT, 1].mean()
            right_brow_y = landmarks[BROW_RIGHT, 1].mean()
            left_eye_y = landmarks[EYE_TOP_LEFT, 1].mean()
            right_eye_y = landmarks[EYE_TOP_RIGHT, 1].mean()
            face_height = np.linalg.norm(landmarks[152, :2] - landmarks[10, :2])
            if face_height <= 0:
                return 0.0
            left_dist = left_eye_y - left_brow_y
            right_dist = right_eye_y - right_brow_y
            avg_dist = (left_dist + right_dist) / 2
            return float(avg_dist / face_height)
        except Exception:
            return 0.0

    @staticmethod
    def brow_furrow_ratio(landmarks: np.ndarray) -> float:
        if landmarks is None or len(landmarks) < 478:
            return 0.0
        try:
            inner_brow_left = landmarks[107, :2]
            inner_brow_right = landmarks[336, :2]
            face_width = np.linalg.norm(landmarks[234, :2] - landmarks[454, :2])
            if face_width <= 0:
                return 0.0
            dist = np.linalg.norm(inner_brow_left - inner_brow_right)
            return float(1.0 - dist / face_width)
        except Exception:
            return 0.0

    @staticmethod
    def eye_openness(landmarks: np.ndarray) -> float:
        if landmarks is None or len(landmarks) < 478:
            return 0.5
        try:
            left_upper = landmarks[159, :2]
            left_lower = landmarks[145, :2]
            right_upper = landmarks[386, :2]
            right_lower = landmarks[374, :2]
            face_height = np.linalg.norm(landmarks[152, :2] - landmarks[10, :2])
            if face_height <= 0:
                return 0.5
            left_open = np.linalg.norm(left_upper - left_lower) / face_height
            right_open = np.linalg.norm(right_upper - right_lower) / face_height
            return float((left_open + right_open) / 2)
        except Exception:
            return 0.5

    @staticmethod
    def compute_all(landmarks: np.ndarray) -> Dict:
        return {
            "mar": LandmarkMetrics.mouth_aspect_ratio(landmarks),
            "mouth_open_ratio": LandmarkMetrics.mouth_open_ratio(landmarks),
            "smile_ratio": LandmarkMetrics.smile_ratio(landmarks),
            "brow_raise": LandmarkMetrics.brow_raise_ratio(landmarks),
            "brow_furrow": LandmarkMetrics.brow_furrow_ratio(landmarks),
            "eye_openness": LandmarkMetrics.eye_openness(landmarks),
        }


# --- Haar cascade mouth detector ---

class HaarMouthDetector:
    def __init__(self):
        mouth_path = cv2.data.haarcascades + 'haarcascade_smile.xml' if hasattr(cv2.data, 'haarcascades') else ''
        self.mouth_cascade = None
        if mouth_path:
            try:
                self.mouth_cascade = cv2.CascadeClassifier(mouth_path)
                self.available = not self.mouth_cascade.empty()
            except Exception:
                self.available = False
        else:
            self.available = False

    def detect(self, frame_gray: np.ndarray, face_roi: Tuple[int, int, int, int]) -> Dict:
        result = {"mouth_detected": False, "mouth_bbox": None, "mouth_open": False}
        if not self.available or self.mouth_cascade is None:
            return result
        try:
            fx, fy, fw, fh = face_roi
            lower_face_y = fy + int(fh * 0.55)
            lower_face = frame_gray[lower_face_y:fy + fh, fx:fx + fw]
            if lower_face.size == 0:
                return result
            mouths = self.mouth_cascade.detectMultiScale(
                lower_face, scaleFactor=1.3, minNeighbors=8, minSize=(20, 10)
            )
            if len(mouths) > 0:
                mx, my, mw, mh = max(mouths, key=lambda m: m[2] * m[3])
                result["mouth_detected"] = True
                result["mouth_bbox"] = (fx + mx, lower_face_y + my, fx + mx + mw, lower_face_y + my + mh)
                result["mouth_open"] = mh > mw * 0.45
        except Exception:
            pass
        return result


# --- Mood Classifier ---

class MoodClassifier:

    def classify(self, bs_dict: Dict[str, float], lm: Dict) -> Dict:
        scores = {
            Mood.HAPPY: self._score_happy(bs_dict, lm),
            Mood.SAD: self._score_sad(bs_dict, lm),
            Mood.ANGRY: self._score_angry(bs_dict, lm),
            Mood.SURPRISED: self._score_surprised(bs_dict, lm),
            Mood.DISGUSTED: self._score_disgusted(bs_dict, lm),
            Mood.FEARFUL: self._score_fearful(bs_dict, lm),
            Mood.EXCITED: self._score_excited(bs_dict, lm),
            Mood.DROWSY: self._score_drowsy(bs_dict, lm),
        }
        scores[Mood.NEUTRAL] = max(0.0, 1.0 - max(scores.values()))

        best_mood = max(scores, key=scores.get)
        confidence = scores[best_mood]

        return {
            "mood": best_mood,
            "confidence": confidence,
            "scores": {m.value: round(s, 4) for m, s in scores.items()},
        }

    def _score_happy(self, bs, lm):
        smile_bs = (_bs(bs, "mouthSmileLeft") + _bs(bs, "mouthSmileRight")) / 2
        cheek = (_bs(bs, "cheekSquintLeft") + _bs(bs, "cheekSquintRight")) / 2
        smile_lm = max(0, lm.get("smile_ratio", 0) * 5)
        score = smile_bs * 0.45 + cheek * 0.2 + smile_lm * 0.35
        return min(1.0, score * 2.5)

    def _score_sad(self, bs, lm):
        frown = (_bs(bs, "mouthFrownLeft") + _bs(bs, "mouthFrownRight")) / 2
        brow_inner = _bs(bs, "browInnerUp")
        mouth_down = (_bs(bs, "mouthLowerDownLeft") + _bs(bs, "mouthLowerDownRight")) / 2
        pucker = _bs(bs, "mouthPucker")
        score = frown * 0.35 + brow_inner * 0.25 + mouth_down * 0.2 + pucker * 0.2
        return min(1.0, score * 3.0)

    def _score_angry(self, bs, lm):
        brow_down = (_bs(bs, "browDownLeft") + _bs(bs, "browDownRight")) / 2
        press = (_bs(bs, "mouthPressLeft") + _bs(bs, "mouthPressRight")) / 2
        sneer = (_bs(bs, "noseSneerLeft") + _bs(bs, "noseSneerRight")) / 2
        furrow_lm = max(0, lm.get("brow_furrow", 0) - 0.5) * 4
        score = brow_down * 0.35 + press * 0.2 + sneer * 0.25 + furrow_lm * 0.2
        return min(1.0, score * 3.0)

    def _score_surprised(self, bs, lm):
        brow_inner = _bs(bs, "browInnerUp")
        brow_outer = (_bs(bs, "browOuterUpLeft") + _bs(bs, "browOuterUpRight")) / 2
        eye_wide = (_bs(bs, "eyeWideLeft") + _bs(bs, "eyeWideRight")) / 2
        jaw_open = _bs(bs, "jawOpen")
        mar_lm = lm.get("mar", 0) * 3
        brow_raise_lm = max(0, lm.get("brow_raise", 0) - 0.08) * 10
        score = (brow_inner + brow_outer) / 2 * 0.25 + eye_wide * 0.25 + jaw_open * 0.2 + mar_lm * 0.15 + brow_raise_lm * 0.15
        return min(1.0, score * 2.5)

    def _score_disgusted(self, bs, lm):
        sneer = (_bs(bs, "noseSneerLeft") + _bs(bs, "noseSneerRight")) / 2
        pucker = _bs(bs, "mouthPucker")
        upper = (_bs(bs, "mouthUpperUpLeft") + _bs(bs, "mouthUpperUpRight")) / 2
        nose_wrinkle = _bs(bs, "noseSneerLeft")
        score = sneer * 0.4 + pucker * 0.2 + upper * 0.2 + nose_wrinkle * 0.2
        return min(1.0, score * 3.0)

    def _score_fearful(self, bs, lm):
        brow_inner = _bs(bs, "browInnerUp")
        eye_wide = (_bs(bs, "eyeWideLeft") + _bs(bs, "eyeWideRight")) / 2
        mouth_funnel = _bs(bs, "mouthFunnel")
        jaw_open = _bs(bs, "jawOpen")
        score = brow_inner * 0.3 + eye_wide * 0.3 + mouth_funnel * 0.2 + jaw_open * 0.2
        return min(1.0, score * 3.0)

    def _score_excited(self, bs, lm):
        smile = (_bs(bs, "mouthSmileLeft") + _bs(bs, "mouthSmileRight")) / 2
        jaw_open = _bs(bs, "jawOpen")
        brow_up = (_bs(bs, "browOuterUpLeft") + _bs(bs, "browOuterUpRight")) / 2
        eye_wide = (_bs(bs, "eyeWideLeft") + _bs(bs, "eyeWideRight")) / 2
        mar_lm = lm.get("mar", 0) * 2
        score = smile * 0.3 + jaw_open * 0.2 + brow_up * 0.15 + eye_wide * 0.15 + mar_lm * 0.2
        return min(1.0, score * 2.5)

    def _score_drowsy(self, bs, lm):
        eye_blink = (_bs(bs, "eyeBlinkLeft") + _bs(bs, "eyeBlinkRight")) / 2
        eye_squint = (_bs(bs, "eyeSquintLeft") + _bs(bs, "eyeSquintRight")) / 2
        mouth_open = _bs(bs, "jawOpen")
        eye_open_lm = lm.get("eye_openness", 0.5)
        eye_closed_lm = max(0, 0.04 - eye_open_lm) * 25
        score = eye_blink * 0.3 + eye_squint * 0.25 + mouth_open * 0.1 + eye_closed_lm * 0.35
        return min(1.0, score * 2.5)


class FacialGestureTracker:

    def __init__(self):
        self.mouth_open = False
        self.smiling = False
        self.yawning = False
        self.brow_raised = False
        self.eye_squinting = False

        self.yawn_start_time = None
        self.yawn_count = 0
        self.smile_count = 0
        self.mouth_open_count = 0
        self.brow_raise_count = 0

        self._prev_mouth_open = False
        self._prev_smiling = False
        self._prev_brow_raised = False

        self.mar_history = deque(maxlen=30)
        self.smile_history = deque(maxlen=30)

    def update(self, bs_dict: Dict[str, float], lm: Dict, haar_mouth: Dict,
               elapsed: float = 0.0) -> Dict:
        mar = lm.get("mar", 0.0)
        mouth_open_lm = lm.get("mouth_open_ratio", 0.0)
        smile_lm = lm.get("smile_ratio", 0.0)
        brow_raise_lm = lm.get("brow_raise", 0.0)

        jaw_open_bs = _bs(bs_dict, "jawOpen")
        smile_bs = (_bs(bs_dict, "mouthSmileLeft") + _bs(bs_dict, "mouthSmileRight")) / 2
        squint_bs = (_bs(bs_dict, "eyeSquintLeft") + _bs(bs_dict, "eyeSquintRight")) / 2
        brow_up_bs = (_bs(bs_dict, "browInnerUp") + _bs(bs_dict, "browOuterUpLeft") + _bs(bs_dict, "browOuterUpRight")) / 3

        self.mar_history.append(mar)

        is_mouth_open_bs = jaw_open_bs > 0.2
        is_mouth_open_lm = mar > 0.15 or mouth_open_lm > 0.04
        is_mouth_open_haar = haar_mouth.get("mouth_open", False)
        self.mouth_open = is_mouth_open_bs or is_mouth_open_lm or is_mouth_open_haar

        is_smile_bs = smile_bs > 0.2
        is_smile_lm = smile_lm > 0.03
        self.smiling = is_smile_bs or is_smile_lm

        is_brow_bs = brow_up_bs > 0.25
        is_brow_lm = brow_raise_lm > 0.10
        self.brow_raised = is_brow_bs or is_brow_lm

        self.eye_squinting = squint_bs > 0.3

        self.smile_history.append(self.smiling)

        effective_mar = mar if mar > 0 else jaw_open_bs * 0.5
        if effective_mar > 0.35 or jaw_open_bs > 0.5:
            if not self.yawning:
                self.yawning = True
                self.yawn_start_time = elapsed
            elif self.yawn_start_time and (elapsed - self.yawn_start_time) > 1.2:
                self.yawn_count += 1
                self.yawn_start_time = elapsed + 8
        else:
            self.yawning = False
            self.yawn_start_time = None

        if self.mouth_open and not self._prev_mouth_open:
            self.mouth_open_count += 1
        if self.smiling and not self._prev_smiling:
            self.smile_count += 1
        if self.brow_raised and not self._prev_brow_raised:
            self.brow_raise_count += 1

        self._prev_mouth_open = self.mouth_open
        self._prev_smiling = self.smiling
        self._prev_brow_raised = self.brow_raised

        return {
            "mouth_open": self.mouth_open,
            "mouth_open_count": self.mouth_open_count,
            "smiling": self.smiling,
            "smile_count": self.smile_count,
            "yawning": self.yawning,
            "yawn_count": self.yawn_count,
            "brow_raised": self.brow_raised,
            "brow_raise_count": self.brow_raise_count,
            "eye_squinting": self.eye_squinting,
            "jaw_open_amount": max(jaw_open_bs, mar),
            "smile_amount": max(smile_bs, smile_lm * 5),
            "mar": mar,
            "mouth_open_ratio": mouth_open_lm,
        }


class MoodTracker:

    def __init__(self, smoothing_window: int = 7):
        self.mood_classifier = MoodClassifier()
        self.gesture_tracker = FacialGestureTracker()
        self.landmark_metrics = LandmarkMetrics()
        self.haar_mouth = HaarMouthDetector()
        self.mood_history = deque(maxlen=smoothing_window)
        self._last_mood = Mood.NEUTRAL
        self._last_mood_time = 0.0

    def analyze(self, blendshapes, landmarks: np.ndarray = None,
                frame_gray: np.ndarray = None, face_bbox=None,
                elapsed: float = 0.0) -> Dict:

        lm = LandmarkMetrics.compute_all(landmarks) if landmarks is not None else {}

        haar_result = {"mouth_detected": False, "mouth_bbox": None, "mouth_open": False}
        if frame_gray is not None and face_bbox is not None:
            haar_result = self.haar_mouth.detect(frame_gray, face_bbox)

        bs_dict = {}
        if blendshapes is not None:
            try:
                for i, bs in enumerate(blendshapes):
                    name = BLENDSHAPE_NAMES[i] if i < len(BLENDSHAPE_NAMES) else f"unknown_{i}"
                    bs_dict[name] = bs.score
            except Exception as e:
                logger.error(f"Blendshape parse error: {e}")

        mood_result = self.mood_classifier.classify(bs_dict, lm)
        gesture_result = self.gesture_tracker.update(bs_dict, lm, haar_result, elapsed)

        self.mood_history.append(mood_result["mood"])
        smoothed_mood = self._smoothed_mood()

        return {
            "mood": smoothed_mood.value,
            "mood_confidence": mood_result["confidence"],
            "mood_scores": mood_result["scores"],
            "gesture": gesture_result,
            "landmark_metrics": {k: round(v, 4) for k, v in lm.items()},
            "haar_mouth": haar_result,
            "blendshapes_raw": {k: round(v, 4) for k, v in bs_dict.items()},
        }

    def _smoothed_mood(self) -> Mood:
        if not self.mood_history:
            return Mood.NEUTRAL
        from collections import Counter
        counts = Counter(self.mood_history)
        return counts.most_common(1)[0][0]
