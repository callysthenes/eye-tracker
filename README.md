# Eye Tracker - Driver Drowsiness Detection

A real-time eye tracking and drowsiness detection system with a cyberpunk aesthetic GUI. Features behavioral analytics, Pomodoro timer integration, and CSV data export—all optimized for low-power CPUs with OpenVINO support.

## 🎯 Features

- **Real-time Eye Tracking**: Detects eye movements, blinks, and gaze direction
- **Drowsiness Detection**: Monitors blink rate (PERCLOS), eye closure, and off-screen gaze
- **Pomodoro Timer**: Classic 25-minute work / 5-minute rest cycles with audio alerts
- **Behavioral Analytics**: Logs blinks/minute, gaze direction, screen attention
- **CSV Export**: Summary statistics exportable for analysis
- **Cyberpunk UI**: Minimalistic, neon-themed PyQt5 interface
- **Lightweight**: Runs on 7th-gen Intel CPU with integrated graphics
- **OpenVINO Support**: Optional hardware-accelerated inference
- **Fallback Mode**: Works with OpenCV cascade detection if YOLO unavailable

## 📋 Requirements

### Hardware
- **Processor**: Intel i5 7th gen or equivalent (CPU inference)
- **RAM**: 4GB minimum (8GB recommended)
- **Webcam**: 720p or higher
- **OS**: Linux (tested on Ubuntu/Debian)

### Software (Minimum)
```
Python 3.9+
OpenCV
NumPy
Pandas
PyQt5
SQLite3 (built-in)
```

### Optional (For Enhanced Detection)
```
Ultralytics YOLO
MediaPipe
OpenVINO Toolkit
```

## 🚀 Installation

### 1. Clone and Setup Environment

```bash
cd /home/pv/eye_tracker
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - Video capture and image processing
- `numpy`, `pandas` - Data handling
- `PyQt5` - GUI framework
- `openvino` - Optional inference acceleration
- `sounddevice` - Audio alerts
- `sqlalchemy` - Database ORM (optional)

### 3. Optional: Install Enhanced Detection (YOLO + MediaPipe)

For better accuracy with YOLO-based detection:

```bash
pip install ultralytics mediapipe
```

**Note**: This step takes 10-20 minutes due to large package downloads.

### 4. Verify Installation

```bash
python test_components.py
```

Expected output: `Results: 7/7 tests passed`

## 🎮 Usage

### Basic Usage (with Cascade Fallback)

```bash
source venv/bin/activate
python main.py
```

The app will:
1. Open a windowed GUI (cyberpunk dark theme)
2. Initialize webcam
3. Start detecting eye movements and blinks
4. Display real-time stats (blinks/min, gaze %, time remaining)
5. Trigger audio "blick" alert at rest time

### Command-Line Options

```bash
python main.py --help

Options:
  --camera ID       Webcam device ID (default: 0)
  --minimized       Start with minimized GUI
  --no-openvino     Disable OpenVINO (use native inference)
```

### Example: Start Minimized

```bash
python main.py --minimized
```

## 📊 Real-Time Display

The GUI shows:

| Metric | Description |
|--------|-------------|
| **STATUS** | Current state (IDLE / WORKING / BREAK) |
| **TIME LEFT** | Minutes:Seconds remaining in block |
| **BLINKS/MIN** | Blink rate (normal: 12-20/min) |
| **EYE CLOSURE** | Eye Aspect Ratio (0=closed, 1=open) |
| **FOCUS** | % of time looking at screen |
| **BLOCK** | Current Pomodoro block number |
| **ALERT STATUS** | ✓ NORMAL / ⚡ WARNING / ⚠ CRITICAL |

### Drowsiness Levels

- **✓ NORMAL**: All metrics healthy
- **⚡ WARNING**: One or more metrics concerning
- **⚠ CRITICAL**: Immediate intervention needed

## 💾 Data Export

### Export Session Summary

Click **📊 EXPORT CSV** button in GUI to export:

```
timestamp, state, block_number, blink_total, blink_per_minute, avg_ear, on_screen_percent
```

CSV files are saved to `exports/` directory with timestamp.

### Database Structure

Raw data is stored in `database/behavior.sqlite`:

**session** table:
```
id | start_time | end_time
```

**event** table:
```
id | session_id | timestamp | type | value
```

Event types:
- `blink` - Blink detected
- `gaze` - Gaze direction change
- `drowsiness_alert` - Alert triggered
- `block_complete` - Work block finished
- `rest_complete` - Rest period finished
- `session_start` / `session_end`

## 🔧 Configuration

Edit these values in `tracker/session.py`:

```python
# Pomodoro intervals
SessionManager(work_minutes=25, rest_minutes=5)

# Drowsiness thresholds
DrowsinessDetector(
    low_blink_threshold=8.0,         # Blinks/min
    high_perclos_threshold=80.0,     # % eye closure
    off_screen_timeout=3.0           # Seconds away
)

# Blink detection sensitivity
BlinkDetector(
    ear_threshold=0.2,               # Eye Aspect Ratio threshold
    blink_frames=3                   # Consecutive frames
)
```

## 📁 Project Structure

```
eye_tracker/
├── main.py                         # App entry point
├── requirements.txt                # Dependencies
├── test_components.py              # Component tests
│
├── gui/
│   ├── window.py                   # PyQt5 main window
│   └── theme.py                    # Cyberpunk color palette & stylesheet
│
├── tracker/
│   ├── detector.py                 # YOLO + cascade detection
│   ├── blink.py                    # Eye Aspect Ratio (EAR) blink detection
│   ├── gaze.py                     # Gaze direction estimation
│   └── session.py                  # Pomodoro & drowsiness logic
│
├── database/
│   ├── db.py                       # SQLite schema & handler
│   └── export.py                   # CSV export functionality
│
├── assets/
│   └── sound/                      # Audio alert files (future)
│
├── models/                         # YOLO/OpenVINO models (auto-downloaded)
│
└── exports/                        # CSV export directory
```

## 🧪 Testing

Run the full component test suite:

```bash
python test_components.py
```

Tests validate:
- Module imports
- Blink detection logic
- Gaze estimation
- Session management
- Database operations
- CSV export
- Detection fallback (cascade)

## 📝 Logging

Logs are written to:
- Console output (INFO level)
- `eye_tracker.log` file (DEBUG level)

Check logs for issues:

```bash
tail -f eye_tracker.log
```

## 🎨 GUI Controls

| Control | Function |
|---------|----------|
| **🔊 SOUND ON/OFF** | Toggle audio alert |
| **📊 EXPORT CSV** | Export session summary |
| **_** | Minimize window |
| **✕** | Exit application |
| **ESC** | Minimize to tray |
| **Ctrl+Q** | Quit |

## ⚙️ Performance Tips

### For Low-Power Hardware

1. **Reduce Resolution**: Edit `WebcamCapture(width=480, height=360)`
2. **Skip Frames**: Run gaze detection every N frames instead of every frame
3. **Disable Landmarks**: Disable MediaPipe to reduce CPU load
4. **Use Cascade Detection**: Stick with OpenCV fallback (no YOLO)

### For Better Detection

1. Install YOLO: `pip install ultralytics`
2. Convert to OpenVINO: Models auto-convert on first run
3. Use OpenVINO: Hardware-accelerated inference (if supported)

## 🐛 Troubleshooting

### Webcam not opening
- Check if webcam is accessible: `ls /dev/video*`
- Try different camera ID: `python main.py --camera 1`

### Slow performance
- Use cascade fallback (no YOLO)
- Reduce frame resolution
- Disable unnecessary logging

### Module import errors
- Reinstall core deps: `pip install opencv-python numpy PyQt5 pandas`
- Check Python version: `python --version` (need 3.9+)

### Sound not playing
- Check system audio: `pactl info`
- Install PulseAudio: `sudo apt install pulseaudio`

## 📈 Accuracy Notes

The system uses three methods for detection (in order of preference):

1. **YOLOv8n (YOLO)** - Best accuracy, requires ultralytics
2. **MediaPipe Face Mesh** - Fast landmarks, requires mediapipe
3. **OpenCV Cascade Classifiers** - Lightweight fallback (always available)

Drowsiness detection accuracy improves with:
- Better lighting
- Clear face visibility
- Proper distance (30-60cm from webcam)
- High FPS (30+ FPS recommended)

## 📜 License

This project is open-source. Feel free to modify and distribute.

## 🚀 Future Enhancements

- [ ] Real-time eye gaze heatmap overlay
- [ ] Advanced PERCLOS calculation
- [ ] Head pose estimation
- [ ] Multi-face detection
- [ ] Mobile app version
- [ ] Cloud data sync
- [ ] Machine learning-based drowsiness predictor
- [ ] Custom alert sounds

## 💬 Support

For issues or questions:
1. Check `eye_tracker.log` for error messages
2. Run `test_components.py` to validate setup
3. Verify camera with: `python -c "import cv2; print(cv2.__version__)"`

---

**Made with ⚡ for drivers who care about safety.**
