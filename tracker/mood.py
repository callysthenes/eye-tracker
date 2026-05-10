"""
Facial expression and mood analysis using MediaPipe blendshapes.
Classifies emotions (happy, sad, angry, surprised, disgusted, fearful, neutral)
and tracks facial gestures (mouth open, smile, brow raise, etc.) in real time.
"""

import numpy as np
from typing import Dict, Optional, List
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


MOOD_EMOJI = {
    Mood.NEUTRAL: ":|",
    Mood.HAPPY: ":)",
    Mood.SAD: ":(",
    Mood.ANGRY: ">:(",
    Mood.SURPRISED: ":O",
    Mood.DISGUSTED: "XP",
    Mood.FEARFUL: "D:",
    Mood.EXCITED: "xD",
}

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


def _bs(blendshapes: Dict[str, float], name: str) -> float:
    return blendshapes.get(name, 0.0)


class MoodClassifier:
    """
    Classifies mood from MediaPipe blendshapes using rule-based scoring.
    Each emotion gets a score based on weighted blendshape combinations.
    """

    THRESHOLD = 0.15

    def classify(self, blendshapes: Dict[str, float]) -> Dict:
        scores = {
            Mood.HAPPY: self._score_happy(blendshapes),
            Mood.SAD: self._score_sad(blendshapes),
            Mood.ANGRY: self._score_angry(blendshapes),
            Mood.SURPRISED: self._score_surprised(blendshapes),
            Mood.DISGUSTED: self._score_disgusted(blendshapes),
            Mood.FEARFUL: self._score_fearful(blendshapes),
            Mood.EXCITED: self._score_excited(blendshapes),
        }
        scores[Mood.NEUTRAL] = max(0.0, 1.0 - max(scores.values()))

        best_mood = max(scores, key=scores.get)
        confidence = scores[best_mood]

        if confidence < self.THRESHOLD:
            best_mood = Mood.NEUTRAL

        return {
            "mood": best_mood,
            "confidence": confidence,
            "scores": {m.value: s for m, s in scores.items()},
        }

    def _score_happy(self, bs: Dict[str, float]) -> float:
        smile = (_bs(bs, "mouthSmileLeft") + _bs(bs, "mouthSmileRight")) / 2
        cheek = (_bs(bs, "cheekSquintLeft") + _bs(bs, "cheekSquintRight")) / 2
        jaw_open = _bs(bs, "jawOpen")
        return min(1.0, smile * 0.6 + cheek * 0.25 + jaw_open * 0.15)

    def _score_sad(self, bs: Dict[str, float]) -> float:
        frown = (_bs(bs, "mouthFrownLeft") + _bs(bs, "mouthFrownRight")) / 2
        brow_inner = _bs(bs, "browInnerUp")
        mouth_down = (_bs(bs, "mouthLowerDownLeft") + _bs(bs, "mouthLowerDownRight")) / 2
        return min(1.0, frown * 0.45 + brow_inner * 0.3 + mouth_down * 0.25)

    def _score_angry(self, bs: Dict[str, float]) -> float:
        brow_down = (_bs(bs, "browDownLeft") + _bs(bs, "browDownRight")) / 2
        frown = (_bs(bs, "mouthFrownLeft") + _bs(bs, "mouthFrownRight")) / 2
        press = (_bs(bs, "mouthPressLeft") + _bs(bs, "mouthPressRight")) / 2
        sneer = (_bs(bs, "noseSneerLeft") + _bs(bs, "noseSneerRight")) / 2
        return min(1.0, brow_down * 0.4 + frown * 0.2 + press * 0.2 + sneer * 0.2)

    def _score_surprised(self, bs: Dict[str, float]) -> float:
        brow_inner = _bs(bs, "browInnerUp")
        brow_outer = (_bs(bs, "browOuterUpLeft") + _bs(bs, "browOuterUpRight")) / 2
        eye_wide = (_bs(bs, "eyeWideLeft") + _bs(bs, "eyeWideRight")) / 2
        jaw_open = _bs(bs, "jawOpen")
        return min(1.0, (brow_inner + brow_outer) / 2 * 0.35 + eye_wide * 0.3 + jaw_open * 0.35)

    def _score_disgusted(self, bs: Dict[str, float]) -> float:
        sneer = (_bs(bs, "noseSneerLeft") + _bs(bs, "noseSneerRight")) / 2
        pucker = _bs(bs, "mouthPucker")
        upper = (_bs(bs, "mouthUpperUpLeft") + _bs(bs, "mouthUpperUpRight")) / 2
        return min(1.0, sneer * 0.5 + pucker * 0.25 + upper * 0.25)

    def _score_fearful(self, bs: Dict[str, float]) -> float:
        brow_inner = _bs(bs, "browInnerUp")
        eye_wide = (_bs(bs, "eyeWideLeft") + _bs(bs, "eyeWideRight")) / 2
        mouth_open = _bs(bs, "jawOpen") * 0.5 + _bs(bs, "mouthFunnel") * 0.5
        return min(1.0, brow_inner * 0.35 + eye_wide * 0.35 + mouth_open * 0.3)

    def _score_excited(self, bs: Dict[str, float]) -> float:
        smile = (_bs(bs, "mouthSmileLeft") + _bs(bs, "mouthSmileRight")) / 2
        brow_up = (_bs(bs, "browOuterUpLeft") + _bs(bs, "browOuterUpRight")) / 2
        jaw_open = _bs(bs, "jawOpen")
        eye_wide = (_bs(bs, "eyeWideLeft") + _bs(bs, "eyeWideRight")) / 2
        return min(1.0, smile * 0.35 + brow_up * 0.2 + jaw_open * 0.25 + eye_wide * 0.2)


class FacialGestureTracker:
    """
    Tracks discrete facial gestures and events:
      - mouth open/close
      - smile start/end
      - yawn detection
      - brow raise
      - eye squint
    """

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

        self.mouth_open_history = deque(maxlen=30)
        self.smile_history = deque(maxlen=30)

    def update(self, blendshapes: Dict[str, float], elapsed: float = 0.0) -> Dict:
        jaw_open = _bs(blendshapes, "jawOpen")
        smile_l = _bs(blendshapes, "mouthSmileLeft")
        smile_r = _bs(blendshapes, "mouthSmileRight")
        brow_inner = _bs(blendshapes, "browInnerUp")
        brow_outer_l = _bs(blendshapes, "browOuterUpLeft")
        brow_outer_r = _bs(blendshapes, "browOuterUpRight")
        squint_l = _bs(blendshapes, "eyeSquintLeft")
        squint_r = _bs(blendshapes, "eyeSquintRight")

        avg_smile = (smile_l + smile_r) / 2
        avg_brow_up = (brow_inner + brow_outer_l + brow_outer_r) / 3
        avg_squint = (squint_l + squint_r) / 2

        self.mouth_open = jaw_open > 0.3
        self.smiling = avg_smile > 0.3
        self.brow_raised = avg_brow_up > 0.35
        self.eye_squinting = avg_squint > 0.4

        self.mouth_open_history.append(self.mouth_open)
        self.smile_history.append(self.smiling)

        if self.mouth_open and jaw_open > 0.6:
            if not self.yawning:
                self.yawning = True
                self.yawn_start_time = elapsed
            if self.yawn_start_time and (elapsed - self.yawn_start_time) > 1.5:
                self.yawn_count += 1
                self.yawn_start_time = elapsed + 10
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
            "jaw_open_amount": jaw_open,
            "smile_amount": avg_smile,
        }


class MoodTracker:
    """
    Full mood and gesture analysis pipeline.
    Combines MoodClassifier and FacialGestureTracker with smoothing.
    """

    def __init__(self, smoothing_window: int = 10):
        self.mood_classifier = MoodClassifier()
        self.gesture_tracker = FacialGestureTracker()
        self.mood_history = deque(maxlen=smoothing_window)
        self.mood_durations: Dict[str, float] = {}
        self._last_mood = Mood.NEUTRAL
        self._last_mood_time = 0.0

    def analyze(self, blendshapes: Optional[List], elapsed: float = 0.0) -> Dict:
        if blendshapes is None or len(blendshapes) == 0:
            return self._empty_result()

        try:
            bs_dict = {}
            for i, bs in enumerate(blendshapes):
                if i < len(BLENDSHAPE_NAMES):
                    bs_dict[BLENDSHAPE_NAMES[i]] = bs.score
                else:
                    bs_dict[f"unknown_{i}"] = bs.score

            mood_result = self.mood_classifier.classify(bs_dict)
            gesture_result = self.gesture_tracker.update(bs_dict, elapsed)

            self.mood_history.append(mood_result["mood"])
            smoothed_mood = self._smoothed_mood()

            if elapsed > 0:
                if self._last_mood != smoothed_mood:
                    self._last_mood = smoothed_mood
                    self._last_mood_time = elapsed

            return {
                "mood": smoothed_mood.value,
                "mood_confidence": mood_result["confidence"],
                "mood_scores": mood_result["scores"],
                "gesture": gesture_result,
                "blendshapes_raw": {k: round(v, 4) for k, v in bs_dict.items()},
            }

        except Exception as e:
            logger.error(f"Mood analysis error: {e}")
            return self._empty_result()

    def _smoothed_mood(self) -> Mood:
        if not self.mood_history:
            return Mood.NEUTRAL
        from collections import Counter
        counts = Counter(self.mood_history)
        return counts.most_common(1)[0][0]

    def _empty_result(self) -> Dict:
        return {
            "mood": "neutral",
            "mood_confidence": 0.0,
            "mood_scores": {},
            "gesture": {
                "mouth_open": False,
                "mouth_open_count": 0,
                "smiling": False,
                "smile_count": 0,
                "yawning": False,
                "yawn_count": 0,
                "brow_raised": False,
                "brow_raise_count": 0,
                "eye_squinting": False,
                "jaw_open_amount": 0.0,
                "smile_amount": 0.0,
            },
            "blendshapes_raw": {},
        }
