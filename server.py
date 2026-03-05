from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uuid, os, asyncio, json, queue, threading
from pipeline import AssistantPipeline

app = FastAPI()
assistant = AssistantPipeline(debug=False)

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def home():
    return FileResponse("index.html")

from core.storage import DIRS, timestamp

@app.post("/listen")
async def listen(audio: UploadFile):
    fname = f"{timestamp()}_{uuid.uuid4().hex}.webm"
    temp_path = os.path.join(DIRS["recordings"], fname)

    # 🔥 1. Save uploaded WebM first
    with open(temp_path, "wb") as f:
        f.write(await audio.read())

    # 🔥 2. Convert to 16kHz mono WAV for Whisper
    import subprocess
    wav_path = temp_path.replace(".webm", ".wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_path,
        "-ar", "16000",
        "-ac", "1",
        wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Use converted WAV for pipeline
    temp_path = wav_path

    status_queue = queue.Queue()

    def run_pipeline():
        try:
            result = assistant.run(
                temp_path,
                status_callback=lambda msg: status_queue.put({
                    "type": "status",
                    "text": msg
                })
            )
            status_queue.put({"type": "result", "data": result})
        except Exception as e:
            status_queue.put({"type": "error", "text": str(e)})
        finally:
            status_queue.put(None)

    async def status_generator():
        thread = threading.Thread(target=run_pipeline)
        thread.start()

        while True:
            try:
                msg = await asyncio.to_thread(status_queue.get, timeout=0.1)
                if msg is None:
                    break
                yield json.dumps(msg) + "\n"
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

    return StreamingResponse(
        status_generator(),
        media_type="application/x-ndjson"
    )

