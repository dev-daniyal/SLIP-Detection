#!/usr/bin/env python3
"""
Slip Detector — Ultimate Frisbee drone footage analysis.

Launch this script to open the file picker, select a video, and run detection.
"""

from ui import SlipDetectorApp


def main():
    app = SlipDetectorApp()
    app.run()


if __name__ == "__main__":
    main()
