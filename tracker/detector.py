"""
YOLO + OpenVINO face and eye detection pipeline.
Handles webcam capture, face detection, eye ROI extraction, and landmark estimation.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
import os

logger = logging.getLogger(__name__)

# Import OpenVINO (for detection pipeline, but NOT used for YOLO to avoid PyTorch conflicts).
# On CPU-only systems, native YOLO via ultralytics is already optimal.
try:
    from openvino import Core, get_version
    OPENVINO_AVAILABLE = True
except ImportError:
    try:
        from openvino.runtime import Core, get_version
        OPENVINO_AVAILABLE = True
    except ImportError:
        OPENVINO_AVAILABLE = False
        logger.warning("OpenVINO runtime not available.")
    except Exception:
        OPENVINO_AVAILABLE = False
except Exception:
    OPENVINO_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics YOLOv8 not available. Install with: pip install ultralytics")

try:
    import mediapipe as mp
    # Verify the new API is available
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available, landmarks will be unavailable.")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    logger.warning(f"MediaPipe new API not available ({e}), landmarks unavailable.")


class YOLODetector:
    """
    YOLO-based face and eye detector using OpenVINO or native inference.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", use_openvino: bool = False, device: str = "cpu"):
        """
        Initialize YOLO detector.
        Uses native YOLO inference (ultralytics) for best CPU performance.
        
        Args:
            model_path: Path to YOLO model (.pt)
            use_openvino: ignored (OpenVINO disabled to avoid PyTorch conflicts on CPU)
            device: Compute device ('cpu' supported)
        """
        self.model_path = model_path
        self.device = device
        self.use_openvino = False  # Always disabled on CPU-only systems
        self.model = None
        self.use_fallback_cascade = False
        self.face_cascade = None
        self.eye_cascade = None
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model natively (ultralytics)."""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Ultralytics YOLO not available, using fallback OpenCV face detection.")
            self.use_fallback_cascade = True
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade.")
            return
        
        # Load using ultralytics (native)
        logger.info(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path)
    
    def _init_openvino(self, ir_model_path: str):
        """Initialize OpenVINO inference engine."""
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available, skipping initialization.")
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """
        Detect faces and eyes in frame using YOLO or cascade fallback.
        
        Args:
            frame: Input image (BGR, numpy array)
            conf_threshold: Confidence threshold for detections
        
        Returns:
            Dictionary with 'faces' (list of bbox+conf) and 'eyes' (list of bbox+conf)
        """
        h, w = frame.shape[:2]
        detections = {
            'faces': [],
            'eyes': [],
            'frame': frame
        }
        
        try:
            if self.use_fallback_cascade:
                detections = self._detect_cascade(frame, h, w)
            elif self.model:
                detections = self._detect_yolo(frame, h, w, conf_threshold)
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return detections
    
    def _detect_yolo(self, frame: np.ndarray, h: int, w: int, conf_threshold: float) -> Dict:
        """Run inference using ultralytics YOLO."""
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = {
            'faces': [],
            'eyes': [],
            'frame': frame
        }
        
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Class 0 = person/face, we'll treat as face for now
                # In practice, use a face-specific YOLO model or add secondary classifier
                detections['faces'].append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'cls': cls
                })
        
        return detections
    
    def _detect_cascade(self, frame: np.ndarray, h: int, w: int) -> Dict:
        """Run inference using OpenCV cascade classifiers (fallback)."""
        detections = {
            'faces': [],
            'eyes': [],
            'frame': frame
        }
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            if self.face_cascade and not self.face_cascade.empty():
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, width, height) in faces:
                    detections['faces'].append({
                        'bbox': (x, y, x + width, y + height),
                        'conf': 0.8,  # No confidence for cascade, use fixed value
                        'cls': 0
                    })
                    
                    # Detect eyes within face region
                    roi_gray = gray[y:y+height, x:x+width]
                    if self.eye_cascade and not self.eye_cascade.empty():
                        eyes = self.eye_cascade.detectMultiScale(roi_gray)
                        for (ex, ey, ew, eh) in eyes:
                            detections['eyes'].append({
                                'bbox': (x + ex, y + ey, x + ex + ew, y + ey + eh),
                                'conf': 0.7,
                                'cls': 0
                            })
        
        except Exception as e:
            logger.error(f"Cascade detection error: {e}")
        
        return detections


class LandmarkDetector:
    """
    Stage 1: Face detection via MediaPipe FaceLandmarker.
    Returns both face bounding box and dense landmarks (478 pts) in one pass.
    """
    
    FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                         361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                         176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                         162, 21, 54, 103, 67, 109]
    
    def __init__(self, model_path: str = "models/face_landmarker.task", use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.model_path = model_path
        self.face_landmarker = None
        
        if self.use_mediapipe:
            self._init_mediapipe()
    
    def _init_mediapipe(self):
        try:
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import RunningMode
            
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = FaceLandmarker.create_from_options(options)
            logger.info(f"MediaPipe FaceLandmarker initialized (model: {self.model_path})")
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe: {e}")
            self.use_mediapipe = False
    
    def detect(self, frame: np.ndarray) -> Dict:
        result = {'face_bbox': None, 'face_conf': 0.0, 'landmarks': None, 'blendshapes': None}
        
        if not self.use_mediapipe or not self.face_landmarker:
            return result
        
        try:
            import mediapipe as mp
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = self.face_landmarker.detect(mp_image)
            
            if not results.face_landmarks or len(results.face_landmarks) == 0:
                return result
            
            h, w = frame.shape[:2]
            landmarks = results.face_landmarks[0]
            lms = np.array([[lm.x * w, lm.y * h, lm.z] for lm in landmarks])
            
            face_bbox = self._bbox_from_landmarks(lms, w, h)
            
            blendshapes = None
            if results.face_blendshapes and len(results.face_blendshapes) > 0:
                blendshapes = results.face_blendshapes[0]
            
            result['landmarks'] = lms
            result['face_bbox'] = face_bbox
            result['face_conf'] = 1.0
            result['blendshapes'] = blendshapes
        except Exception as e:
            logger.error(f"Landmark detection error: {e}")
        
        return result
    
    def _bbox_from_landmarks(self, landmarks: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        oval = landmarks[self.FACE_OVAL_INDICES, :2]
        x_min = max(0, int(oval[:, 0].min()) - 5)
        x_max = min(w, int(oval[:, 0].max()) + 5)
        y_min = max(0, int(oval[:, 1].min()) - 5)
        y_max = min(h, int(oval[:, 1].max()) + 5)
        return (x_min, y_min, x_max, y_max)


class HaarFaceDetector:
    """
    Stage 1 fallback: Haar cascade face + eye detection.
    Used when MediaPipe is unavailable.
    """
    
    def __init__(self):
        face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        self.available = not self.face_cascade.empty()
        if self.available:
            logger.info("Haar cascade face/eye detector initialized")
    
    def detect(self, frame: np.ndarray) -> Dict:
        result = {'faces': [], 'eyes': []}
        if not self.available:
            return result
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        
        for (x, y, w, h) in faces:
            result['faces'].append({
                'bbox': (x, y, x + w, y + h),
                'conf': 0.8,
                'cls': 0
            })
            
            face_roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
            for (ex, ey, ew, eh) in eyes:
                result['eyes'].append({
                    'bbox': (x + ex, y + ey, x + ex + ew, y + ey + eh),
                    'conf': 0.7,
                    'cls': 0
                })
        
        return result


class DetectionPipeline:
    """
    Two-stage detection pipeline:
      Stage 1 - Face detection: MediaPipe FaceLandmarker (primary) or Haar cascade (fallback)
      Stage 2 - Eye detection: landmarks (478 pts) from MediaPipe, or Haar cascade eyes
    """
    
    def __init__(self, yolo_model_path: str = "yolov8n.pt", use_openvino: bool = True,
                 landmark_model_path: str = "models/face_landmarker.task"):
        self.landmark_detector = LandmarkDetector(model_path=landmark_model_path, use_mediapipe=True)
        self.haar_detector = HaarFaceDetector()
        self._mp_available = self.landmark_detector.use_mediapipe
        logger.info(f"Detection pipeline: {'MediaPipe + Haar fallback' if self._mp_available else 'Haar cascade only'}")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Two-stage processing:
          1. MediaPipe: face detection + dense landmarks (includes eye landmarks)
          2. Fallback Haar cascade if MediaPipe finds no face
        """
        detections = {
            'faces': [],
            'eyes': [],
            'landmarks': None,
            'blendshapes': None,
            'face_rois': [],
            'eye_rois': {'left': None, 'right': None},
            'frame': frame,
            'detection_method': 'none'
        }
        
        # --- Stage 1: MediaPipe (face + landmarks + eyes in one pass) ---
        if self._mp_available:
            mp_result = self.landmark_detector.detect(frame)
            landmarks = mp_result['landmarks']
            
            if landmarks is not None:
                detections['landmarks'] = landmarks
                detections['blendshapes'] = mp_result.get('blendshapes')
                detections['detection_method'] = 'mediapipe'
                
                if mp_result['face_bbox'] is not None:
                    detections['faces'].append({
                        'bbox': mp_result['face_bbox'],
                        'conf': mp_result['face_conf'],
                        'cls': 0
                    })
                
                detections['face_rois'] = self._extract_face_rois(frame, detections['faces'])
                detections['eye_rois'] = self._extract_eye_rois(frame, landmarks)
                
                # Add eye bboxes from landmarks
                eye_boxes = self._eye_boxes_from_landmarks(landmarks, frame.shape[:2])
                if eye_boxes:
                    detections['eyes'] = eye_boxes
                
                return detections
        
        # --- Fallback: Haar cascade face + eye detection ---
        haar_result = self.haar_detector.detect(frame)
        if haar_result['faces']:
            detections['faces'] = haar_result['faces']
            detections['eyes'] = haar_result['eyes']
            detections['face_rois'] = self._extract_face_rois(frame, detections['faces'])
            detections['detection_method'] = 'haar_cascade'
        
        return detections
    
    def _eye_boxes_from_landmarks(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> List[Dict]:
        if landmarks is None or len(landmarks) < 478:
            return []
        
        h, w = frame_shape
        eyes = []
        
        left_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        for label, indices in [('left', left_indices), ('right', right_indices)]:
            pts = landmarks[indices, :2].astype(int)
            x_min = max(0, int(pts[:, 0].min()) - 5)
            x_max = min(w, int(pts[:, 0].max()) + 5)
            y_min = max(0, int(pts[:, 1].min()) - 5)
            y_max = min(h, int(pts[:, 1].max()) + 5)
            eyes.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'conf': 0.9,
                'cls': 0,
                'label': label
            })
        
        return eyes
    
    def _extract_face_rois(self, frame: np.ndarray, face_boxes: List[Dict]) -> List[np.ndarray]:
        """Extract face region-of-interest crops."""
        rois = []
        for face in face_boxes:
            x1, y1, x2, y2 = face['bbox']
            # Add padding
            pad = int(0.1 * (x2 - x1))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            roi = frame[y1:y2, x1:x2]
            rois.append(roi)
        return rois
    
    def _extract_eye_rois(self, frame: np.ndarray, landmarks: Optional[np.ndarray]) -> Dict:
        """
        Extract left and right eye ROIs based on landmarks.
        MediaPipe indices (refine_landmarks=True, 478 pts):
          left eye contour: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
          right eye contour: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
          left iris: [468, 469, 470, 471, 472]
          right iris: [473, 474, 475, 476, 477]
        """
        rois = {'left': None, 'right': None}
        
        if landmarks is None or len(landmarks) < 468:
            return rois
        
        try:
            h, w = frame.shape[:2]
            
            # Left eye landmarks (16 contour points around the eye)
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Extract left eye ROI
            left_pts = landmarks[left_eye_indices, :2].astype(int)
            if len(left_pts) > 0:
                x_min = max(0, int(left_pts[:, 0].min()) - 10)
                x_max = min(w, int(left_pts[:, 0].max()) + 10)
                y_min = max(0, int(left_pts[:, 1].min()) - 10)
                y_max = min(h, int(left_pts[:, 1].max()) + 10)
                rois['left'] = frame[y_min:y_max, x_min:x_max]
            
            # Extract right eye ROI
            right_pts = landmarks[right_eye_indices, :2].astype(int)
            if len(right_pts) > 0:
                x_min = max(0, int(right_pts[:, 0].min()) - 10)
                x_max = min(w, int(right_pts[:, 0].max()) + 10)
                y_min = max(0, int(right_pts[:, 1].min()) - 10)
                y_max = min(h, int(right_pts[:, 1].max()) + 10)
                rois['right'] = frame[y_min:y_max, x_min:x_max]
        
        except Exception as e:
            logger.error(f"Eye ROI extraction error: {e}")
        
        return rois


class WebcamCapture:
    """
    Webcam capture with basic frame preprocessing.
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: Webcam device ID
            width, height: Target frame resolution
            fps: Target FPS
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_open = False
        
        self._init_camera()
    
    def _init_camera(self):
        """Initialize and configure webcam."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if self.cap.isOpened():
                self.is_open = True
                logger.info(f"Webcam initialized (ID: {self.camera_id}, {self.width}x{self.height}, {self.fps} FPS)")
            else:
                logger.error(f"Failed to open webcam {self.camera_id}")
        except Exception as e:
            logger.error(f"Webcam initialization error: {e}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from webcam.
        
        Returns:
            Tuple of (success, frame) where frame is BGR numpy array or None.
        """
        if not self.is_open or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            logger.error(f"Frame read error: {e}")
            return False, None
    
    def release(self):
        """Release webcam resources."""
        if self.cap:
            self.cap.release()
            self.is_open = False
            logger.info("Webcam released.")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    pipeline = DetectionPipeline()
    
    # Initialize webcam
    webcam = WebcamCapture(camera_id=0, width=640, height=480)
    
    print("Detection pipeline initialized. Press 'q' to quit.")
    
    while webcam.is_open:
        ret, frame = webcam.read()
        if not ret or frame is None:
            break
        
        # Process frame
        detections = pipeline.process_frame(frame)
        
        # Draw detections on frame
        for face in detections['faces']:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {face['conf']:.2f}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display
        cv2.imshow("Eye Tracker - Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()
    cv2.destroyAllWindows()
    print("Test complete.")
