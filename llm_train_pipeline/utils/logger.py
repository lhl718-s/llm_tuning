import sys
import time
import os

class TimestampedLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.terminal.write(f"{message}")
        self.logfile.write(f"{message}")

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

def setup_logger(log_dir, script_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{script_name}.log")
    sys.stdout = TimestampedLogger(log_file)
    return sys.stdout