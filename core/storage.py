import os
from datetime import datetime

BASE = "data"

DIRS = {
    "recordings": f"{BASE}/recordings",
    "denoised": f"{BASE}/denoised",
    "tts": f"{BASE}/tts",
    "logs": f"{BASE}/logs",
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
