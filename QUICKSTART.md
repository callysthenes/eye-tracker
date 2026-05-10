# Quick Start Guide

Get Eye Tracker running in 5 minutes!

## ⚡ Super Fast Setup (5 mins)

### 1. One-Command Installation

```bash
cd /home/pv/eye_tracker
bash setup.sh
```

That's it! The script handles:
- ✓ Virtual environment creation
- ✓ Dependency installation
- ✓ Component testing
- ✓ Optional YOLO/MediaPipe installation

### 2. Start the Application

```bash
source venv/bin/activate
python main.py
```

## 🎯 Basic Workflow

1. **Start**: Run `python main.py`
2. **See**: Cyberpunk GUI appears with stats
3. **Work**: 25-minute Pomodoro block starts
4. **Rest**: 5-minute break reminder with audio alert
5. **Export**: Click **📊 EXPORT CSV** to save data

## 🔊 Audio Alert (Blick Sound)

- Plays when rest time is due
- Toggle on/off with **🔊 SOUND ON/OFF** button
- Volume controlled by system audio

## 📊 What Gets Tracked

| Data | Logged |
|------|--------|
| Blinks per minute | ✓ CSV |
| Eye closure duration (PERCLOS) | ✓ Database |
| Gaze direction | ✓ CSV summary |
| Time on screen | ✓ CSV |
| Drowsiness events | ✓ Both |
| Pomodoro blocks | ✓ Both |

## 📁 Key Files After Setup

```
eye_tracker/
├── database/behavior.sqlite    # Raw session data
├── exports/                    # CSV summaries
├── eye_tracker.log             # Application logs
└── venv/                       # Python environment
```

## 🧪 Test Installation

Verify everything works:

```bash
source venv/bin/activate
python test_components.py
```

Expected: `Results: 7/7 tests passed`

## 🚀 Pro Tips

### Run in Minimized Mode
```bash
python main.py --minimized
```
GUI starts hidden; press keyboard shortcut to show.

### Use Different Webcam
```bash
python main.py --camera 1
```
(Use `--camera 0`, `1`, `2`, etc.)

### Disable OpenVINO (CPU Only)
```bash
python main.py --no-openvino
```

### View Real-Time Logs
```bash
tail -f eye_tracker.log
```

## 📊 Exporting Data

1. **During Session**: Click **📊 EXPORT CSV**
2. **Auto-Saved**: Also in `database/behavior.sqlite`
3. **Format**: CSV with session summary:
   - Timestamp, state, blink rate, gaze %, etc.

## 🎨 Customize Appearance

Edit `gui/theme.py` to change:
- Colors
- Font sizes
- Border styles
- Alert styling

Example:
```python
COLORS = {
    'primary_neon': '#00f0ff',  # Change this to your color
    ...
}
```

## ⚙️ Adjust Sensitivity

Edit `tracker/session.py`:

```python
# More sensitive to drowsiness
DrowsinessDetector(
    low_blink_threshold=10.0,    # Alert if <10 blinks/min
    high_perclos_threshold=70.0  # Alert if >70% closed
)

# Less sensitive
DrowsinessDetector(
    low_blink_threshold=5.0,
    high_perclos_threshold=90.0
)
```

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| GUI doesn't appear | Check: `xhost +local:` |
| Webcam not found | Try `--camera 1` or `--camera 2` |
| Slow performance | Disable YOLO, use cascade fallback |
| No sound | Check system volume, test: `speaker-test` |
| Import errors | Run `pip install -r requirements.txt` again |

## 📝 Log Output

When you run the app, you'll see:

```
INFO:tracker.detector:Webcam initialized (ID: 0, 640x480, 30 FPS)
INFO:tracker.session:Starting work block #1
INFO:__main__:Main loop started
...
```

This is normal! Check `eye_tracker.log` for detailed info.

## 🔄 Next Steps

1. ✓ Installation complete
2. ⏭ Read full `README.md` for advanced features
3. ⏭ Customize `tracker/session.py` for your needs
4. ⏭ Integrate CSV export with your analytics tools

## 💬 Stuck?

1. Check `eye_tracker.log` for errors
2. Run `python test_components.py`
3. Verify Python 3.9+: `python --version`
4. Verify camera: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

---

**Ready? Let's go!**

```bash
cd /home/pv/eye_tracker
source venv/bin/activate
python main.py
```
