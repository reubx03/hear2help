import json
from datetime import datetime
from core.storage import DIRS

LOG_FILE = f"{DIRS['logs']}/queries.log"

def log_query(raw_text, lang, intent, entities, response):
    entry = {
        "time": datetime.now().isoformat(),
        "language": lang,
        "query": raw_text,
        "intent": intent,
        "entities": entities,
        "response": response,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
