"""Launch CAM VISION from either ``python -m app.main`` or ``python app/main.py``."""

if __package__ in (None, ""):
    # Allow the simple command documented in README when launched from project root.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    try:
        # Import inside the guarded block so missing packages are reported in a dialog.
        if __package__ in (None, ""):
            from app.ui import VisionDashboard
        else:
            from .ui import VisionDashboard
        VisionDashboard().run()
    except Exception as exc:
        message = (
            f"CAM VISION could not start: {type(exc).__name__}: {exc}\n\n"
            "Activate the project environment and install requirements with:\n"
            "python -m pip install -r requirements.txt\n"
            "If tkinter is missing, install Python with Tcl/Tk support "
            "(Windows) or install python3-tk (Linux)."
        )
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("CAM VISION — Startup Error", message)
            root.destroy()
        except Exception:
            print(message)


if __name__ == "__main__":
    main()
