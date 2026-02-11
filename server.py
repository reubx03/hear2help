from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uuid, os
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
    fname = f"{timestamp()}_{uuid.uuid4().hex}.wav"
    temp_path = os.path.join(DIRS["recordings"], fname)

    with open(temp_path, "wb") as f:
        f.write(await audio.read())

    result = assistant.run(temp_path)
    return result

