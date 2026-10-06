"""Modern lightweight Tkinter UI for the live CAM VISION dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from .config import CONFIG
from .worker import VisionWorker


BG = "#080b14"
PANEL = "#101625"
PANEL_2 = "#141d2e"
BORDER = "#26334a"
TEXT = "#edf3ff"
MUTED = "#8d9bb3"
CYAN = "#50e5ff"
PURPLE = "#a889ff"
GREEN = "#50e3a4"
RED = "#ff6b88"
PINK = "#ff7ac8"


class VisionDashboard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(CONFIG.window_title)
        self.root.configure(bg=BG)
        # Fit the window to the screen (small laptops are often 1366x768).
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        win_w = min(1440, max(CONFIG.window_min_width, screen_w - 80))
        win_h = min(900, max(CONFIG.window_min_height, screen_h - 140))
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2 - 20)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(CONFIG.window_min_width, CONFIG.window_min_height)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.worker = VisionWorker()
        self.worker.start()
        self.project_root = Path(__file__).resolve().parents[1]
        self.captures_dir = self.project_root / "captures"
        self._photo = None
        self._fullscreen = False
        self._toast_after = None
        self._build()
        self.root.after(40, self._refresh)

    def _build(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self.root, bg=BG, padx=26, pady=17)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        brand = tk.Frame(header, bg=BG)
        brand.grid(row=0, column=0, sticky="w")
        tk.Label(brand, text="CAM VISION", bg=BG, fg=TEXT,
                 font=("Segoe UI", 19, "bold")).pack(anchor="w")
        tk.Label(brand, text="AI COMPUTER VISION SYSTEM", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(1, 0))
        tk.Label(header, text="●", bg=BG, fg=GREEN, font=("Segoe UI", 13)).grid(row=0, column=2, padx=(10, 5))
        self.camera_chip = tk.Label(header, text="CAMERA  STARTING", bg=PANEL_2, fg=CYAN,
                                    font=("Segoe UI", 9, "bold"), padx=12, pady=8)
        self.camera_chip.grid(row=0, column=3, padx=(0, 10))
        self.header_fps = tk.Label(header, text="FPS  --", bg=PANEL_2, fg=TEXT,
                                   font=("Consolas", 10, "bold"), padx=13, pady=8)
        self.header_fps.grid(row=0, column=4)
        tk.Frame(self.root, bg=BORDER, height=1).grid(row=0, column=0, sticky="sew")

        body = tk.Frame(self.root, bg=BG, padx=22, pady=18)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self._build_preview(body)
        self._build_sidebar(body)
        self._build_footer()

    def _build_preview(self, parent) -> None:
        outer = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 17))
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        bar = tk.Frame(outer, bg=PANEL, padx=16, pady=12)
        bar.grid(row=0, column=0, sticky="ew")
        tk.Label(bar, text="LIVE CAMERA FEED", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(bar, text="1 FINGER: DRAW   ·   2 HANDS: HEART   ·   FIST: CLEAR", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="right")
        self.preview = tk.Label(outer, text="Initializing camera and AI engine…", bg="#05070c",
                                fg=MUTED, font=("Segoe UI", 13), compound="center")
        self.preview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.error_label = tk.Label(outer, text="", bg="#25131e", fg="#ffb4c4",
                                    font=("Segoe UI", 9), anchor="w", justify="left", padx=12, pady=8,
                                    wraplength=800)
        self.error_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.error_label.grid_remove()
        self.toast = tk.Label(outer, text="", bg="#183b37", fg="#a6ffe4",
                              font=("Segoe UI", 9, "bold"), padx=12, pady=8)
        self.toast.place(relx=0.98, rely=0.97, anchor="se")
        self.toast.place_forget()

    def _build_sidebar(self, parent) -> None:
        side = tk.Frame(parent, bg=BG, width=305)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        self._section_title(side, "SYSTEM STATUS", "01")
        self._status_card = tk.Frame(side, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=9)
        self._status_card.pack(fill="x", pady=(6, 12))
        self.status_values = {}
        for key, label in (("camera", "CAMERA"), ("engine", "AI ENGINE"),
                           ("fps", "FRAME RATE"), ("latency", "LATENCY"),
                           ("confidence", "CONFIDENCE"), ("gesture", "GESTURE")):
            row = tk.Frame(self._status_card, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(side="left")
            value = tk.Label(row, text="--", bg=PANEL, fg=TEXT, font=("Consolas", 9, "bold"))
            value.pack(side="right")
            self.status_values[key] = value

        self._section_title(side, "DETECTION LAYERS", "02")
        control = tk.Frame(side, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=9)
        control.pack(fill="x", pady=(6, 12))
        self.toggles = {}
        for key, label, accent in (("face", "Face Mesh", PURPLE), ("hands", "Hand Tracking", CYAN),
                                   ("pose", "Pose Tracking", GREEN), ("draw", "Air Draw", PINK)):
            row = tk.Frame(control, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=TEXT, font=("Segoe UI", 10)).pack(side="left")
            button = tk.Button(row, text="ON", command=lambda k=key: self._toggle(k),
                               bg="#173a3a", fg=GREEN, activebackground="#20504c", activeforeground=TEXT,
                               relief="flat", bd=0, width=7, font=("Segoe UI", 8, "bold"), cursor="hand2")
            button.pack(side="right")
            self.toggles[key] = {"enabled": True, "button": button, "accent": accent}

        self._section_title(side, "LIVE STATISTICS", "03")
        stats = tk.Frame(side, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=9)
        stats.pack(fill="x", pady=(6, 12))
        self.stat_values = {}
        for key, label in (("faces", "FACES"), ("hands", "HANDS"), ("pose", "POSES")):
            row = tk.Frame(stats, bg=PANEL)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(side="left")
            value = tk.Label(row, text="0", bg=PANEL, fg=self._accent_for(key), font=("Consolas", 14, "bold"))
            value.pack(side="right")
            self.stat_values[key] = value

        mirror = tk.Frame(side, bg=PANEL_2, padx=14, pady=10)
        mirror.pack(fill="x")
        tk.Label(mirror, text="Mirror camera", bg=PANEL_2, fg=TEXT,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.mirror_button = tk.Button(mirror, text="ON", command=self._toggle_mirror,
                                       bg="#173a3a", fg=GREEN, activebackground="#20504c", activeforeground=TEXT,
                                       relief="flat", bd=0, width=7, font=("Segoe UI", 8, "bold"), cursor="hand2")
        self.mirror_button.pack(side="right")

    def _build_footer(self) -> None:
        footer = tk.Frame(self.root, bg="#0b101b", padx=24, pady=12)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.footer_info = tk.Label(footer, text="CAMERA: --   ·   PROCESSING: REAL-TIME   ·   AI: STARTING",
                                    bg="#0b101b", fg=MUTED, font=("Segoe UI", 8, "bold"))
        self.footer_info.grid(row=0, column=0, sticky="w")
        buttons = tk.Frame(footer, bg="#0b101b")
        buttons.grid(row=0, column=1, sticky="e")
        self._action_button(buttons, "CLEAR", self._clear_canvas, "#3d3519", fg="#ffe29a")
        self._action_button(buttons, "CAPTURE", self._capture, CYAN, fg="#06131a")
        self._action_button(buttons, "FULLSCREEN", self._toggle_fullscreen, "#26344c", fg=TEXT)
        self._action_button(buttons, "EXIT", self._on_close, "#3a202c", fg="#ffafc0")

    @staticmethod
    def _section_title(parent, title: str, number: str) -> None:
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x")
        tk.Label(row, text=number, bg=BG, fg=PURPLE, font=("Consolas", 8, "bold")).pack(side="left", padx=(0, 8))
        tk.Label(row, text=title, bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(side="left")

    @staticmethod
    def _accent_for(key: str) -> str:
        return {"faces": PURPLE, "hands": CYAN, "pose": GREEN}.get(key, TEXT)

    @staticmethod
    def _action_button(parent, text: str, command, bg: str, fg: str) -> None:
        tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                  activebackground="#364761", activeforeground=TEXT, relief="flat", bd=0,
                  padx=15, pady=9, font=("Segoe UI", 8, "bold"), cursor="hand2").pack(side="left", padx=(8, 0))

    def _toggle(self, key: str) -> None:
        state = self.toggles[key]
        state["enabled"] = not state["enabled"]
        enabled = state["enabled"]
        button = state["button"]
        button.configure(text="ON" if enabled else "OFF",
                         bg="#173a3a" if enabled else "#302432",
                         fg=GREEN if enabled else "#c58a9a")
        self.worker.set_setting(key, enabled)

    def _toggle_mirror(self) -> None:
        current = self.mirror_button.cget("text") == "ON"
        self.worker.set_setting("mirror", not current)
        self.mirror_button.configure(text="OFF" if current else "ON",
                                     bg="#302432" if current else "#173a3a",
                                     fg="#c58a9a" if current else GREEN)

    def _clear_canvas(self) -> None:
        self.worker.request_clear()
        self._show_toast("Drawing cleared")

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _capture(self) -> None:
        frame, metrics = self.worker.snapshot()
        if frame is None:
            self._show_toast("No live frame to capture yet", error=True)
            return
        try:
            self.captures_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = self.captures_dir / f"capture_{stamp}.jpg"
            # Avoid overwriting if two captures happen in the same second.
            suffix = 1
            while path.exists():
                path = self.captures_dir / f"capture_{stamp}_{suffix}.jpg"
                suffix += 1
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError("OpenCV could not write the image file.")
            self._show_toast(f"Screenshot saved · {path.name}")
        except Exception as exc:
            self._show_toast(f"Could not save screenshot: {exc}", error=True)

    def _show_toast(self, text: str, error: bool = False) -> None:
        self.toast.configure(text=text, bg="#4a1f2d" if error else "#183b37",
                             fg="#ffc0cc" if error else "#a6ffe4")
        self.toast.place(relx=0.98, rely=0.97, anchor="se")
        if self._toast_after:
            self.root.after_cancel(self._toast_after)
        self._toast_after = self.root.after(3000, self.toast.place_forget)

    def _refresh(self) -> None:
        if not self.root.winfo_exists():
            return
        frame, metrics = self.worker.snapshot()
        camera = metrics["camera"]
        color = GREEN if camera == "CONNECTED" else (RED if ("ERROR" in camera or "UNAVAILABLE" in camera) else CYAN)
        self.camera_chip.configure(text=f"CAMERA  {camera}", fg=color)
        fps_text = f"{metrics['fps']:.0f}" if metrics["fps"] else "--"
        self.header_fps.configure(text=f"FPS  {fps_text}")
        self.status_values["camera"].configure(text=camera, fg=color)
        self.status_values["engine"].configure(text=metrics["engine"], fg=GREEN if metrics["engine"] == "ACTIVE" else color)
        self.status_values["fps"].configure(text=f"{metrics['fps']:.1f} FPS" if metrics["fps"] else "--")
        self.status_values["latency"].configure(text=f"{metrics['latency']:.0f} ms" if metrics["latency"] else "--")
        confidence = metrics["confidence"]
        self.status_values["confidence"].configure(text=f"{confidence * 100:.0f}%" if confidence is not None else "--")
        gesture = metrics.get("gesture", "--")
        gesture_color = {"LOVE": PINK, "DRAWING": CYAN, "CLEARED": "#ffd166", "FIST": "#ffd166", "HAND": GREEN, "ERROR": RED, "OFF": MUTED}.get(gesture, TEXT)
        self.status_values["gesture"].configure(text=gesture, fg=gesture_color)
        for key in ("faces", "hands", "pose"):
            self.stat_values[key].configure(text=str(metrics[key]))

        if frame is not None:
            try:
                canvas_w = max(320, self.preview.winfo_width() - 4)
                canvas_h = max(240, self.preview.winfo_height() - 4)
                h, w = frame.shape[:2]
                scale = min(canvas_w / w, canvas_h / h)
                resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.preview.configure(image=self._photo, text="")
            except tk.TclError:
                pass
        elif metrics["error"]:
            self.preview.configure(image="", text="WEBCAM / ENGINE NOT READY\n\nSee the diagnostic below.")

        if metrics["error"]:
            self.error_label.configure(text=metrics["error"])
            self.error_label.grid()
        else:
            self.error_label.grid_remove()
        size = f"{metrics['width']}×{metrics['height']}" if metrics["width"] else "--"
        self.footer_info.configure(text=f"CAMERA: {size}   ·   PROCESSING: REAL-TIME   ·   AI: {metrics['engine']}")
        self.root.after(40, self._refresh)

    def _on_close(self) -> None:
        self.worker.stop()
        # Give the worker a moment to release the camera before the window goes away.
        self.worker.join(timeout=2.0)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
