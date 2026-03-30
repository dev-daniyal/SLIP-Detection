"""
Tkinter UI for the slip detection app.

Provides a file picker, progress bar, and results display.
"""

import os
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from detector import run_detection, save_csv, DetectionResult


class SlipDetectorApp:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Slip Detector — Ultimate Frisbee")
        self.root.geometry("520x340")
        self.root.resizable(False, False)

        self.video_path: str | None = None
        self.result: DetectionResult | None = None
        self.csv_path: str | None = None
        self._cancel = False

        self._build_ui()

        # macOS Tk rendering workaround: force the display pipeline to
        # flush before entering mainloop, otherwise widgets may never
        # paint on macOS Ventura/Sonoma/Sequoia with Homebrew python-tk.
        #
        # update_idletasks() alone only processes geometry/layout idle
        # tasks — it does NOT flush the Quartz draw pipeline.  The
        # withdraw → update_idletasks → deiconify sequence unmaps and
        # remaps the window, forcing the compositor to issue a full
        # repaint of every widget that was already packed.
        if platform.system() == "Darwin":
            self.root.update_idletasks()
            self.root.withdraw()
            self.root.update_idletasks()
            self.root.deiconify()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        # File selection
        file_frame = ttk.LabelFrame(self.root, text="Video File", padding=8)
        file_frame.pack(fill="x", **pad)

        self.file_label = ttk.Label(file_frame, text="No file selected", anchor="w")
        self.file_label.pack(side="left", fill="x", expand=True)

        self.pick_btn = ttk.Button(file_frame, text="Browse…", command=self._pick_file)
        self.pick_btn.pack(side="right")

        # Controls
        ctrl_frame = ttk.Frame(self.root, padding=4)
        ctrl_frame.pack(fill="x", **pad)

        self.run_btn = ttk.Button(ctrl_frame, text="Run Detection", command=self._start,
                                  state="disabled")
        self.run_btn.pack(side="left")

        self.cancel_btn = ttk.Button(ctrl_frame, text="Cancel", command=self._cancel_run,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))

        # Progress
        prog_frame = ttk.LabelFrame(self.root, text="Progress", padding=8)
        prog_frame.pack(fill="x", **pad)

        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill="x")

        self.status_label = ttk.Label(prog_frame, text="Waiting…", anchor="w")
        self.status_label.pack(fill="x", pady=(4, 0))

        # Results
        res_frame = ttk.LabelFrame(self.root, text="Results", padding=8)
        res_frame.pack(fill="x", **pad)

        self.result_label = ttk.Label(res_frame, text="—", anchor="w")
        self.result_label.pack(side="left", fill="x", expand=True)

        self.open_csv_btn = ttk.Button(res_frame, text="Open CSV", command=self._open_csv,
                                       state="disabled")
        self.open_csv_btn.pack(side="right")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select drone footage",
            filetypes=[("Video files", "*.mp4 *.mov *.MP4 *.MOV"), ("All files", "*.*")],
        )
        if path:
            self.video_path = path
            self.file_label.config(text=os.path.basename(path))
            self.run_btn.config(state="normal")
            self.result_label.config(text="—")
            self.open_csv_btn.config(state="disabled")

    def _start(self):
        """Kick off detection in a background thread so the UI stays responsive."""
        self._cancel = False
        self.run_btn.config(state="disabled")
        self.pick_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.open_csv_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_label.config(text="Processing…")
        self.result_label.config(text="—")

        thread = threading.Thread(target=self._run_detection, daemon=True)
        thread.start()

    def _cancel_run(self):
        self._cancel = True
        self.status_label.config(text="Cancelling…")

    def _run_detection(self):
        """Run detection (called in background thread)."""
        def progress_cb(current, total):
            if total > 0:
                pct = (current / total) * 100
                self.root.after(0, self._update_progress, pct, current, total)

        def cancel_cb():
            return self._cancel

        try:
            result = run_detection(
                self.video_path,
                progress_callback=progress_cb,
                cancel_check=cancel_cb,
            )

            self.result = result
            csv_path = save_csv(result)
            self.csv_path = csv_path

            # Update UI on the main thread
            self.root.after(0, self._on_complete)
        except Exception as e:
            self._error_msg = str(e)
            self.root.after(0, self._on_error)

    def _update_progress(self, pct, current, total):
        self.progress["value"] = pct
        self.status_label.config(
            text=f"Frame {current:,} / {total:,}  ({pct:.0f}%)")

    def _on_error(self):
        """Handle an exception from the background detection thread."""
        self.run_btn.config(state="normal")
        self.pick_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_label.config(text="Error during processing.")
        self.result_label.config(text=f"Error: {self._error_msg}")

    def _on_complete(self):
        self.run_btn.config(state="normal")
        self.pick_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.progress["value"] = 100

        r = self.result
        if r is None:
            return

        slip_count = len(r.slips)
        self.status_label.config(text="Done.")
        self.result_label.config(
            text=f"{slip_count} slip(s) detected in {r.processing_time:.1f}s")

        if self.csv_path:
            self.open_csv_btn.config(state="normal")

    def _open_csv(self):
        """Open the CSV file with the system default application."""
        if not self.csv_path or not os.path.exists(self.csv_path):
            return
        abs_path = os.path.abspath(self.csv_path)
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", abs_path])
        elif system == "Windows":
            os.startfile(abs_path)
        else:
            subprocess.Popen(["xdg-open", abs_path])

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()
