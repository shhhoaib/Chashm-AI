"""Run Chashm AI with error logging to a file."""
import sys, os, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    from app.main_window import MainWindow
    from app.config import APP_NAME

    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    print("Creating MainWindow...", file=open(log_path, "w"))
    window = MainWindow()
    print("MainWindow created successfully", file=open(log_path, "a"))
    window.show()
    print("Window shown", file=open(log_path, "a"))
    sys.exit(app.exec())
except Exception:
    with open(log_path, "a") as f:
        f.write("=== CRASH ===\n")
        traceback.print_exc(file=f)
        f.write("\n")
    print(f"CRASH! Check {log_path} for details", flush=True)
    raise
