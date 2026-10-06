# Cam Vision

**Cam Vision** is a real-time laptop-webcam computer-vision dashboard built with Python, OpenCV, MediaPipe, and Tkinter. It detects facial mesh landmarks, both hands, and body pose landmarks, then draws them directly over the live camera preview.

> The app uses the computer's built-in/default webcam at camera index `0`. No external camera, depth sensor, GPU, or mobile device is required.

## Features

- Live webcam with a mirror-view toggle; requests 1280×720 and accepts a resolution supported by the camera driver, with 640×480 as a fallback.
- MediaPipe Face Mesh drawn as a clean outline (face edge, eyes/irises, brows, lips). The dense full mesh is off by default; set `face_full_mesh = True` in `app/config.py` to bring it back.
- **Air drawing and gestures** (see below): draw with one finger, show a heart with two hands, clear with a fist.
- Up to two hands, with finger joints and skeleton connections.
- Lightweight body-pose tracking, including upper-body landmarks.
- Independently toggle face, hand, and pose overlays; disabled detectors are skipped for lower CPU use.
- Live FPS, inference latency, camera/model status, face/hand/pose counts, and available hand/pose confidence values.
- Capture the processed preview to `captures/` as a timestamped JPEG.
- Dark, responsive dashboard UI. Camera acquisition and model inference run on a worker thread so the UI remains responsive.

Face Mesh's legacy MediaPipe API does not expose a per-face confidence score. The confidence readout therefore displays the available hand classification and pose-visibility scores, or `--` when those are unavailable.

## Hand gestures and air drawing

| Gesture | What happens |
|---|---|
| **One index finger pointing** (other fingers folded) | Draws a glowing line in the air that follows your fingertip. Write names, letters or any shape. |
| **Lower or fold the finger** | Lifts the pen, so the next move starts a new line. |
| **Two hands making a heart** (index fingertips touching at the top, thumb tips touching at the bottom, wrists apart) | Shows a big animated heart with "LOVE" for about 3 seconds. |
| **Closed fist, held about 0.4 s, facing the camera** | Clears everything drawn (lines and heart). |

The **CLEAR** button in the footer erases the drawing too, and the **Air Draw** switch turns the feature on or off. The sidebar GESTURE row shows what the app sees: `NO HAND` (no hand found), `HAND` (hand found, no gesture), `DRAWING`, `FIST`, `LOVE` or `CLEARED`.

Tips: keep your hand about an arm's length (40 to 70 cm) from the camera with the whole hand inside the picture (a hand held right against the lens is often not detected), use good light, and keep the other three fingers folded when drawing. The slow face and pose models run only on every 4th frame (`face_every`, `pose_every` in `app/config.py`) so hands stay smooth. Avoid strong backlight (a bright window behind you) because washed-out video makes hands hard to detect; for more speed you can also switch off **Face Mesh** and **Pose Tracking**. If the heart or fist is hard to trigger, tune the values in the *Air drawing*, *Closed fist* and *Two-hand heart* sections of `app/config.py`.

## Screenshots

_Add application screenshots here after running the project._

## Requirements

- Windows 10/11 recommended; macOS/Linux may also work if Python Tk support and OpenCV webcam access are available.
- Python 3.11 (64-bit recommended).
- A built-in or default webcam and permission to access it.
- Internet access for dependency installation and MediaPipe's first model initialization (MediaPipe may download its pose model on first launch).
- CPU-only; a GPU is not required.

Tkinter is included with most standard Python installations. On Linux, install the system `python3-tk` package if `tkinter` is missing.

## Installation (Windows PowerShell)

From the project root:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks local activation scripts for this session, use Command Prompt instead:

```bat
py -3.11 -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How to run

From the project root, with the virtual environment activated:

```powershell
python app/main.py
```

The app opens camera index `0` automatically. The first start can take a little longer while MediaPipe initializes and downloads its pose model. The live dashboard remains open and displays an error message if the camera or AI engine cannot start.

## Camera permissions

### Windows 10/11

1. Open **Settings → Privacy & security → Camera** (Windows 10: **Settings → Privacy → Camera**).
2. Turn on **Camera access** and **Let desktop apps access your camera**.
3. Close apps that may already be using the webcam (Teams, Zoom, browser video calls), then relaunch Cam Vision.
4. If a physical privacy shutter or keyboard camera-disable key exists, enable the camera.

## Project structure

```text
cam-vision/
├── README.md
├── requirements.txt
├── captures/                 # Created when the first screenshot is saved
└── app/
    ├── __init__.py
    ├── main.py                # Application entry point
    ├── config.py              # Camera/model/window defaults
    ├── camera.py              # Webcam open, resolution negotiation, frame reads
    ├── worker.py              # Background capture, inference, and metrics
    ├── renderer.py            # Neon landmark overlays and preview label
    ├── gestures.py            # Air drawing, heart and fist gestures
    ├── ui.py                  # Tkinter dashboard and controls
    └── detectors/
        ├── __init__.py
        ├── face.py            # MediaPipe Face Mesh
        ├── hands.py           # MediaPipe Hands
        └── pose.py            # MediaPipe Pose
```

## Troubleshooting

- **Could not open camera 0:** Check Windows camera privacy settings, the physical shutter/function key, and whether another app is using the webcam. This first version uses the system's default built-in camera at index 0.
- **Camera opens but preview is black/empty:** Quit other camera apps, check the shutter, and restart. Try another supported webcam resolution if the camera driver behaves unusually.
- **`No module named tkinter`:** Reinstall Python with **tcl/tk and IDLE** selected in the Windows installer. On Linux, install `python3-tk` through your package manager.
- **MediaPipe or NumPy import errors:** Use 64-bit Python 3.11, activate the project virtual environment, and reinstall the pinned requirements: `pip install --force-reinstall -r requirements.txt`.
- **Low FPS:** Turn off unused detector layers, improve lighting, and close applications that consume CPU. Inference input is automatically reduced to at most 960 px wide; the displayed and captured camera frame retains the camera's negotiated resolution.
- **Permission denied:** Allow desktop application camera access in Windows Privacy & security settings and restart the app.
- **Screenshot not saved:** Ensure the project folder is writable. Captures are saved under `<project-root>/captures/`.

## Future improvements

- Camera selection and an explicit resolution selector.
- Optional detector confidence/model settings and richer performance history.
- More configurable overlay palettes and a recording/export workflow.
- Packaging as a standalone Windows installer.
