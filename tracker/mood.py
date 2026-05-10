"""
Production-grade facial expression analysis using FACS (Facial Action Coding System).

Core design principles (psychiatry-informed):
  1. CALIBRATION: Learn each user's neutral baseline over first N frames.
     All subsequent readings are DELTAS from baseline. This eliminates the
     false-positive problem where idle blendshape noise triggers emotions.
  2. ACTION UNITS (FACS): Map blendshapes + landmarks to Ekman's Action Units.
     Emotions are detected from AU combinations, not raw blendshape values.
  3. MUSCLE GROUP COVERAGE: Brow (corrugator, frontalis), eyes (orbicularis oculi),
     nose (procerus, nasalis), mouth (orbicularis oris, zygomaticus, depressor),
     jaw (masseter), chin (mentalis) — all tracked via landmarks + blendshapes.
  4. SCREENSHOT: Every mood/expression change is captured to assets/.
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


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
    CONTEMPT = "contempt"


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


# ---------------------------------------------------------------------------
# FACS Action Units — mapped from blendshapes + landmarks
# These correspond to Ekman & Friesen's original AUs
# ---------------------------------------------------------------------------

class ActionUnits:
    """
    Computes FACS Action Units from blendshapes + landmark deltas.
    All values are DELTA from calibrated neutral baseline.
    """

    AU_MAP = {
        "AU1_inner_brow_raiser": ["browInnerUp"],
        "AU2_outer_brow_raiser": ["browOuterUpLeft", "browOuterUpRight"],
        "AU4_brow_lowerer": ["browDownLeft", "browDownRight"],
        "AU5_upper_lid_raiser": ["eyeWideLeft", "eyeWideRight"],
        "AU6_cheek_raiser": ["cheekSquintLeft", "cheekSquintRight"],
        "AU7_lid_tightener": ["eyeSquintLeft", "eyeSquintRight"],
        "AU9_nose_wrinkler": ["noseSneerLeft", "noseSneerRight"],
        "AU12_lip_corner_puller": ["mouthSmileLeft", "mouthSmileRight"],
        "AU14_dimpler": ["mouthDimpleLeft", "mouthDimpleRight"],
        "AU15_lip_corner_depressor": ["mouthFrownLeft", "mouthFrownRight"],
        "AU17_chin_raiser": ["mouthUpperUpLeft", "mouthUpperUpRight"],
        "AU20_lip_stretcher": ["mouthStretchLeft", "mouthStretchRight"],
        "AU23_lip_tightener": ["mouthPressLeft", "mouthPressRight"],
        "AU24_lip_pressor": ["mouthPressLeft", "mouthPressRight"],
        "AU25_lips_part": ["jawOpen"],
        "AU26_jaw_open": ["jawOpen"],
        "AU28_lip_suck": ["mouthRollLower", "mouthRollUpper"],
        "AU43_eyes_closed": ["eyeBlinkLeft", "eyeBlinkRight"],
        "AU45_blink": ["eyeBlinkLeft", "eyeBlinkRight"],
        "AU46_wink": ["eyeBlinkLeft"],
    }

    def compute(self, bs_dict: Dict[str, float], lm_deltas: Dict) -> Dict[str, float]:
        aus = {}
        for au_name, bs_names in self.AU_MAP.items():
            values = [bs_dict.get(n, 0.0) for n in bs_names]
            aus[au_name] = sum(values) / len(values)

        # Override / augment with landmark deltas
        mar_delta = lm_deltas.get("mar_delta", 0.0)
        smile_delta = lm_deltas.get("smile_ratio_delta", 0.0)
        brow_raise_delta = lm_deltas.get("brow_raise_delta", 0.0)
        brow_furrow_delta = lm_deltas.get("brow_furrow_delta", 0.0)
        eye_open_delta = lm_deltas.get("eye_openness_delta", 0.0)

        if mar_delta > 0.02:
            aus["AU25_lips_part"] = max(aus.get("AU25_lips_part", 0), mar_delta * 8)
            aus["AU26_jaw_open"] = max(aus.get("AU26_jaw_open", 0), mar_delta * 6)
        if smile_delta > 0.005:
            aus["AU12_lip_corner_puller"] = max(aus.get("AU12_lip_corner_puller", 0), smile_delta * 30)
        if brow_raise_delta > 0.005:
            aus["AU1_inner_brow_raiser"] = max(aus.get("AU1_inner_brow_raiser", 0), brow_raise_delta * 15)
            aus["AU2_outer_brow_raiser"] = max(aus.get("AU2_outer_brow_raiser", 0), brow_raise_delta * 12)
        if brow_furrow_delta > 0.01:
            aus["AU4_brow_lowerer"] = max(aus.get("AU4_brow_lowerer", 0), brow_furrow_delta * 10)
        if eye_open_delta < -0.005:
            aus["AU43_eyes_closed"] = max(aus.get("AU43_eyes_closed", 0), abs(eye_open_delta) * 30)

        return aus


# ---------------------------------------------------------------------------
# Neutral baseline calibration
# ---------------------------------------------------------------------------

class BaselineCalibrator:
    """
    Collects blendshape + landmark samples during neutral expression
    (first CALIBRATION_FRAMES frames) and computes mean baseline.
    All subsequent analysis uses delta-from-baseline.
    """

    CALIBRATION_FRAMES = 60
    CALIBRATION_INTERVAL = 2  # sample every N frames during calibration

    def __init__(self):
        self.bs_samples: List[Dict[str, float]] = []
        self.lm_samples: List[Dict[str, float]] = []
        self.calibrated = False
        self.bs_baseline: Dict[str, float] = {}
        self.lm_baseline: Dict[str, float] = {}

    def add_sample(self, bs_dict: Dict[str, float], lm: Dict[str, float]):
        if self.calibrated:
            return
        self.bs_samples.append(bs_dict.copy())
        self.lm_samples.append(lm.copy())
        if len(self.bs_samples) >= self.CALIBRATION_FRAMES:
            self._compute_baseline()

    def _compute_baseline(self):
        all_bs_keys = set()
        for s in self.bs_samples:
            all_bs_keys.update(s.keys())
        for key in all_bs_keys:
            vals = [s.get(key, 0.0) for s in self.bs_samples]
            self.bs_baseline[key] = float(np.mean(vals))

        all_lm_keys = set()
        for s in self.lm_samples:
            all_lm_keys.update(s.keys())
        for key in all_lm_keys:
            vals = [s.get(key, 0.0) for s in self.lm_samples]
            self.lm_baseline[key] = float(np.mean(vals))

        self.calibrated = True
        logger.info(f"Baseline calibrated from {len(self.bs_samples)} samples")

    def delta_bs(self, bs_dict: Dict[str, float]) -> Dict[str, float]:
        if not self.calibrated:
            return bs_dict
        return {k: max(0.0, v - self.bs_baseline.get(k, 0.0)) for k, v in bs_dict.items()}

    def delta_lm(self, lm: Dict[str, float]) -> Dict[str, float]:
        if not self.calibrated:
            return {k: 0.0 for k in lm}
        return {k: v - self.lm_baseline.get(k, 0.0) for k, v in lm.items()}


# ---------------------------------------------------------------------------
# Landmark-based metrics (all major muscle groups)
# ---------------------------------------------------------------------------

BROW_LEFT = [70, 63, 105, 66, 107]
BROW_RIGHT = [300, 293, 334, 296, 336]
EYE_TOP_LEFT = [159]
EYE_TOP_RIGHT = [386]
EYE_BOT_LEFT = [145]
EYE_BOT_RIGHT = [374]
FOREHEAD = [10, 109, 338, 67]
CHIN = [152]
NOSE_BRIDGE = [6, 168, 1, 197]
CHEEK_LEFT = [50, 116, 117, 118, 119]
CHEEK_RIGHT = [280, 345, 346, 347, 348]
NASOLABIAL_LEFT = [36, 31, 48, 2]
NASOLABIAL_RIGHT = [266, 261, 278, 272]


class LandmarkMetrics:

    @staticmethod
    def compute_all(landmarks: np.ndarray) -> Dict:
        if landmarks is None or len(landmarks) < 478:
            return {}
        try:
            fh = np.linalg.norm(landmarks[152, :2] - landmarks[10, :2])
            fw = np.linalg.norm(landmarks[454, :2] - landmarks[234, :2])
            if fh <= 0 or fw <= 0:
                return {}

            upper_lip = landmarks[13, :2]
            lower_lip = landmarks[14, :2]
            left_corner = landmarks[61, :2]
            right_corner = landmarks[291, :2]
            mar = np.linalg.norm(upper_lip - lower_lip) / np.linalg.norm(left_corner - right_corner)

            upper_lip_y = landmarks[[13, 82, 312], 1].mean()
            lower_lip_y = landmarks[[14, 87, 317], 1].mean()
            mouth_open_ratio = (lower_lip_y - upper_lip_y) / fh

            corner_mid = (left_corner + right_corner) / 2
            smile_ratio = (corner_mid[1] - upper_lip[1]) / np.linalg.norm(right_corner - left_corner)

            left_brow_y = landmarks[BROW_LEFT, 1].mean()
            right_brow_y = landmarks[BROW_RIGHT, 1].mean()
            left_eye_y = landmarks[EYE_TOP_LEFT, 1].mean()
            right_eye_y = landmarks[EYE_TOP_RIGHT, 1].mean()
            brow_raise = ((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) / 2 / fh

            inner_brow_dist = np.linalg.norm(landmarks[107, :2] - landmarks[336, :2])
            brow_furrow = 1.0 - inner_brow_dist / fw

            left_open = np.linalg.norm(landmarks[159, :2] - landmarks[145, :2]) / fh
            right_open = np.linalg.norm(landmarks[386, :2] - landmarks[374, :2]) / fh
            eye_openness = (left_open + right_open) / 2

            forehead_height = np.linalg.norm(landmarks[10, :2] - landmarks[107, :2]) / fh

            nose_wrinkle = 0.0
            nose_l = landmarks[48, :2]
            nose_r = landmarks[278, :2]
            nose_bridge_mid = landmarks[6, :2]
            expected_width = fw * 0.25
            actual_width = np.linalg.norm(nose_l - nose_r)
            if actual_width < expected_width:
                nose_wrinkle = (expected_width - actual_width) / expected_width

            cheek_puff = 0.0
            left_cheek_x = landmarks[[116, 117], 0].mean()
            right_cheek_x = landmarks[[345, 346], 0].mean()
            face_center_x = (landmarks[234, 0] + landmarks[454, 0]) / 2
            left_dist = abs(left_cheek_x - face_center_x) / fw
            right_dist = abs(right_cheek_x - face_center_x) / fw
            cheek_puff = (left_dist + right_dist) / 2

            jaw_open_lm = np.linalg.norm(landmarks[152, :2] - landmarks[10, :2]) / fh - 0.6
            jaw_forward = (landmarks[152, 0] - landmarks[10, 0]) / fw

            mouth_width = np.linalg.norm(left_corner - right_corner) / fw
            mouth_asymmetry = abs(left_corner[1] - right_corner[1]) / fh

            return {
                "mar": float(mar),
                "mouth_open_ratio": float(mouth_open_ratio),
                "smile_ratio": float(smile_ratio),
                "brow_raise": float(brow_raise),
                "brow_furrow": float(brow_furrow),
                "eye_openness": float(eye_openness),
                "forehead_height": float(forehead_height),
                "nose_wrinkle": float(nose_wrinkle),
                "cheek_puff": float(cheek_puff),
                "jaw_open_lm": float(max(0, jaw_open_lm)),
                "jaw_forward": float(jaw_forward),
                "mouth_width": float(mouth_width),
                "mouth_asymmetry": float(mouth_asymmetry),
            }
        except Exception as e:
            logger.error(f"Landmark metrics error: {e}")
            return {}


# ---------------------------------------------------------------------------
# Haar cascade supplementary detectors
# ---------------------------------------------------------------------------

class HaarDetectors:
    def __init__(self):
        base = cv2.data.haarcascades
        self.smile = cv2.CascadeClassifier(base + 'haarcascade_smile.xml')
        self.eye = cv2.CascadeClassifier(base + 'haarcascade_eye_tree_eyeglasses.xml')
        self.available = not self.smile.empty()

    def detect(self, gray: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        result = {"smile_detected": False, "mouth_open": False}
        if not self.available or face_bbox is None:
            return result
        try:
            fx, fy, fw, fh = face_bbox
            lower = gray[fy + int(fh * 0.55):fy + fh, fx:fx + fw]
            if lower.size == 0:
                return result
            smiles = self.smile.detectMultiScale(lower, 1.3, minNeighbors=12, minSize=(20, 10))
            if len(smiles) > 0:
                result["smile_detected"] = True
                sx, sy, sw, sh = max(smiles, key=lambda m: m[2] * m[3])
                result["mouth_open"] = sh > sw * 0.5
        except Exception:
            pass
        return result


# ---------------------------------------------------------------------------
# FACS-based Mood Classifier (psychiatry-informed)
# Emotion -> AU combinations from Ekman (1978) and updates by Friesen & Hager
# ---------------------------------------------------------------------------

class FACSMoodClassifier:
    """
    Classifies emotions using FACS Action Unit combinations.
    Based on Ekman's universal emotions and their AU prototypes.
    Uses only DELTA values (change from neutral), so idle noise is eliminated.
    """

    ACTIVATION_THRESHOLD = 0.08

    def classify(self, aus: Dict[str, float]) -> Dict:
        scores = {
            Mood.HAPPY: self._happy(aus),
            Mood.SAD: self._sad(aus),
            Mood.ANGRY: self._angry(aus),
            Mood.SURPRISED: self._surprised(aus),
            Mood.DISGUSTED: self._disgusted(aus),
            Mood.FEARFUL: self._fearful(aus),
            Mood.EXCITED: self._excited(aus),
            Mood.DROWSY: self._drowsy(aus),
            Mood.CONTEMPT: self._contempt(aus),
        }

        active_aus = {k: v for k, v in aus.items() if v > self.ACTIVATION_THRESHOLD}
        if not active_aus:
            scores[Mood.NEUTRAL] = 1.0
        else:
            scores[Mood.NEUTRAL] = max(0.0, 1.0 - max(scores.values()) * 1.5)

        best = max(scores, key=scores.get)
        conf = scores[best]

        return {
            "mood": best,
            "confidence": min(1.0, conf),
            "scores": {m.value: round(s, 4) for m, s in scores.items()},
            "active_aus": {k: round(v, 3) for k, v in active_aus.items()},
        }

    def _au(self, aus: Dict, name: str) -> float:
        return aus.get(name, 0.0)

    def _happy(self, aus):
        au12 = self._au(aus, "AU12_lip_corner_puller")
        au6 = self._au(aus, "AU6_cheek_raiser")
        au14 = self._au(aus, "AU14_dimpler")
        au25 = self._au(aus, "AU25_lips_part")
        if au12 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = au12 * 0.5 + au6 * 0.25 + au14 * 0.1 + au25 * 0.15
        return min(1.0, score * 2.0)

    def _sad(self, aus):
        au15 = self._au(aus, "AU15_lip_corner_depressor")
        au1 = self._au(aus, "AU1_inner_brow_raiser")
        au4 = self._au(aus, "AU4_brow_lowerer")
        au17 = self._au(aus, "AU17_chin_raiser")
        if au15 < self.ACTIVATION_THRESHOLD and au1 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = au15 * 0.35 + au1 * 0.25 + au4 * 0.2 + au17 * 0.2
        return min(1.0, score * 2.5)

    def _angry(self, aus):
        au4 = self._au(aus, "AU4_brow_lowerer")
        au23 = self._au(aus, "AU23_lip_tightener")
        au9 = self._au(aus, "AU9_nose_wrinkler")
        au24 = self._au(aus, "AU24_lip_pressor")
        au7 = self._au(aus, "AU7_lid_tightener")
        if au4 < self.ACTIVATION_THRESHOLD and au23 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = au4 * 0.3 + au23 * 0.2 + au9 * 0.15 + au24 * 0.15 + au7 * 0.2
        return min(1.0, score * 2.5)

    def _surprised(self, aus):
        au1 = self._au(aus, "AU1_inner_brow_raiser")
        au2 = self._au(aus, "AU2_outer_brow_raiser")
        au5 = self._au(aus, "AU5_upper_lid_raiser")
        au26 = self._au(aus, "AU26_jaw_open")
        au25 = self._au(aus, "AU25_lips_part")
        brow_up = max(au1, au2)
        if brow_up < self.ACTIVATION_THRESHOLD and au5 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = brow_up * 0.3 + au5 * 0.3 + au26 * 0.2 + au25 * 0.2
        return min(1.0, score * 2.0)

    def _disgusted(self, aus):
        au9 = self._au(aus, "AU9_nose_wrinkler")
        au15 = self._au(aus, "AU15_lip_corner_depressor")
        au25 = self._au(aus, "AU25_lips_part")
        if au9 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = au9 * 0.5 + au15 * 0.2 + au25 * 0.3
        return min(1.0, score * 2.5)

    def _fearful(self, aus):
        au1 = self._au(aus, "AU1_inner_brow_raiser")
        au2 = self._au(aus, "AU2_outer_brow_raiser")
        au5 = self._au(aus, "AU5_upper_lid_raiser")
        au25 = self._au(aus, "AU25_lips_part")
        au20 = self._au(aus, "AU20_lip_stretcher")
        if au1 < self.ACTIVATION_THRESHOLD and au5 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = au1 * 0.25 + au2 * 0.15 + au5 * 0.25 + au25 * 0.15 + au20 * 0.2
        return min(1.0, score * 2.5)

    def _excited(self, aus):
        au12 = self._au(aus, "AU12_lip_corner_puller")
        au5 = self._au(aus, "AU5_upper_lid_raiser")
        au26 = self._au(aus, "AU26_jaw_open")
        au6 = self._au(aus, "AU6_cheek_raiser")
        if au12 < self.ACTIVATION_THRESHOLD and au5 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = au12 * 0.3 + au5 * 0.2 + au26 * 0.25 + au6 * 0.25
        return min(1.0, score * 2.0)

    def _drowsy(self, aus):
        au43 = self._au(aus, "AU43_eyes_closed")
        au7 = self._au(aus, "AU7_lid_tightener")
        au45 = self._au(aus, "AU45_blink")
        if au43 < self.ACTIVATION_THRESHOLD and au45 < 0.3:
            return 0.0
        score = au43 * 0.5 + au7 * 0.2 + au45 * 0.3
        return min(1.0, score * 2.0)

    def _contempt(self, aus):
        au12 = self._au(aus, "AU12_lip_corner_puller")
        au14 = self._au(aus, "AU14_dimpler")
        au15 = self._au(aus, "AU15_lip_corner_depressor")
        au4 = self._au(aus, "AU4_brow_lowerer")
        if au14 < self.ACTIVATION_THRESHOLD and au12 < self.ACTIVATION_THRESHOLD:
            return 0.0
        score = au14 * 0.35 + au12 * 0.25 + au15 * 0.2 + au4 * 0.2
        return min(1.0, score * 2.0)


# ---------------------------------------------------------------------------
# Gesture tracker (mouth, yawn, brow, squint)
# ---------------------------------------------------------------------------

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
        self._prev_mouth = False
        self._prev_smile = False
        self._prev_brow = False

    def update(self, aus: Dict, lm_raw: Dict, lm_deltas: Dict, haar: Dict, elapsed: float, bs_raw: Dict) -> Dict:
        au12 = aus.get("AU12_lip_corner_puller", 0)
        au25 = aus.get("AU25_lips_part", 0)
        au26 = aus.get("AU26_jaw_open", 0)
        au1 = aus.get("AU1_inner_brow_raiser", 0)
        au7 = aus.get("AU7_lid_tightener", 0)
        mar_raw = lm_raw.get("mar", 0.0)
        mar_d = lm_deltas.get("mar_delta", 0.0)
        jaw_open_bs = bs_raw.get("jawOpen", 0.0)
        smile_bs = (bs_raw.get("mouthSmileLeft", 0.0) + bs_raw.get("mouthSmileRight", 0.0)) / 2
        tongue_bs = bs_raw.get("tongueOut", 0.0)

        self.smiling = au12 > 0.06 or smile_bs > 0.2 or haar.get("smile_detected", False)
        self.mouth_open = au25 > 0.06 or au26 > 0.06 or mar_raw > 0.15 or jaw_open_bs > 0.15 or haar.get("mouth_open", False)
        self.brow_raised = au1 > 0.06
        self.eye_squinting = au7 > 0.08

        if self.mouth_open and (mar_raw > 0.3 or jaw_open_bs > 0.4):
            if not self.yawning:
                self.yawning = True
                self.yawn_start_time = elapsed
            elif self.yawn_start_time and (elapsed - self.yawn_start_time) > 1.2:
                self.yawn_count += 1
                self.yawn_start_time = elapsed + 8
        else:
            self.yawning = False
            self.yawn_start_time = None

        if self.mouth_open and not self._prev_mouth:
            self.mouth_open_count += 1
        if self.smiling and not self._prev_smile:
            self.smile_count += 1
        if self.brow_raised and not self._prev_brow:
            self.brow_raise_count += 1
        self._prev_mouth = self.mouth_open
        self._prev_smile = self.smiling
        self._prev_brow = self.brow_raised

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
            "tongue_out": tongue_bs > 0.15,
            "jaw_open_amount": jaw_open_bs,
            "smile_amount": smile_bs,
            "mar": mar_raw,
            "mouth_open_ratio": lm_raw.get("mouth_open_ratio", 0.0),
        }


# ---------------------------------------------------------------------------
# Screenshot capture on mood change
# ---------------------------------------------------------------------------

def capture_mood_screenshot(frame: np.ndarray, mood: str, session_id: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(ASSETS_DIR, f"session{session_id}_{ts}_{mood}.jpg")
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.info(f"Mood screenshot saved: {filename}")
    return filename


# ---------------------------------------------------------------------------
# Main MoodTracker — orchestrates everything
# ---------------------------------------------------------------------------

class MoodTracker:

    CALIBRATION_SAMPLES = 60

    def __init__(self, smoothing_window: int = 7):
        self.calibrator = BaselineCalibrator()
        self.au_engine = ActionUnits()
        self.classifier = FACSMoodClassifier()
        self.gesture_tracker = FacialGestureTracker()
        self.landmark_metrics = LandmarkMetrics()
        self.haar = HaarDetectors()
        self.mood_history = deque(maxlen=smoothing_window)
        self._last_mood = Mood.NEUTRAL
        self._calibration_frame = 0

    @property
    def is_calibrated(self):
        return self.calibrator.calibrated

    @property
    def calibration_progress(self):
        return min(1.0, len(self.calibrator.bs_samples) / self.calibrator.CALIBRATION_FRAMES)

    def analyze(self, blendshapes, landmarks=None, frame_gray=None, face_bbox=None,
                frame_bgr=None, session_id=0, elapsed: float = 0.0) -> Dict:

        lm = self.landmark_metrics.compute_all(landmarks) if landmarks is not None else {}

        haar_result = {"smile_detected": False, "mouth_open": False}
        if frame_gray is not None and face_bbox is not None:
            haar_result = self.haar.detect(frame_gray, face_bbox)

        bs_dict = {}
        if blendshapes is not None:
            try:
                cats = blendshapes.categories if hasattr(blendshapes, 'categories') else blendshapes
                for i, bs in enumerate(cats):
                    name = bs.category_name if hasattr(bs, 'category_name') else BLENDSHAPE_NAMES[i] if i < len(BLENDSHAPE_NAMES) else f"unknown_{i}"
                    bs_dict[name] = bs.score
            except Exception as e:
                logger.error(f"Blendshape parse error: {e}")
            if bs_dict:
                logger.debug(f"Parsed {len(bs_dict)} blendshapes, tongueOut={bs_dict.get('tongueOut', 0):.3f}")

        if not self.calibrator.calibrated:
            self._calibration_frame += 1
            if self._calibration_frame % BaselineCalibrator.CALIBRATION_INTERVAL == 0:
                self.calibrator.add_sample(bs_dict, lm)
            return self._calibrating_result()

        bs_delta = self.calibrator.delta_bs(bs_dict)
        lm_deltas = self.calibrator.delta_lm(lm)
        aus = self.au_engine.compute(bs_delta, lm_deltas)

        mood_result = self.classifier.classify(aus)
        gesture_result = self.gesture_tracker.update(aus, lm, lm_deltas, haar_result, elapsed, bs_dict)

        self.mood_history.append(mood_result["mood"])
        smoothed = self._smoothed_mood()

        screenshot_path = None
        if smoothed != self._last_mood and frame_bgr is not None:
            screenshot_path = capture_mood_screenshot(frame_bgr, smoothed.value, session_id)
            self._last_mood = smoothed

        return {
            "mood": smoothed.value,
            "mood_confidence": mood_result["confidence"],
            "mood_scores": mood_result["scores"],
            "active_aus": mood_result.get("active_aus", {}),
            "gesture": gesture_result,
            "landmark_metrics": {k: round(v, 4) for k, v in lm.items()},
            "screenshot": screenshot_path,
            "calibrated": True,
        }

    def _smoothed_mood(self) -> Mood:
        if not self.mood_history:
            return Mood.NEUTRAL
        from collections import Counter
        counts = Counter(self.mood_history)
        return counts.most_common(1)[0][0]

    def _calibrating_result(self) -> Dict:
        pct = int(self.calibration_progress * 100)
        return {
            "mood": "calibrating",
            "mood_confidence": 0.0,
            "mood_scores": {},
            "active_aus": {},
            "gesture": {
                "mouth_open": False, "mouth_open_count": 0,
                "smiling": False, "smile_count": 0,
                "yawning": False, "yawn_count": 0,
                "brow_raised": False, "brow_raise_count": 0,
                "eye_squinting": False,
                "jaw_open_amount": 0.0, "smile_amount": 0.0,
                "mar": 0.0, "mouth_open_ratio": 0.0,
            },
            "landmark_metrics": {},
            "screenshot": None,
            "calibrated": False,
            "calibration_progress": pct,
        }
