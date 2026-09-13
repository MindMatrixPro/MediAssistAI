"""
MediAssist AI - Streamlit Application Entry Point
"""
import os
import sys
import runpy

# Ensure src directory is in sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Execute main application from src
APP_PATH = os.path.join(SRC_DIR, "main_app.py")
runpy.run_path(APP_PATH, run_name="__main__")
