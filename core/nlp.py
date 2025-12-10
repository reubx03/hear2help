from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
import re
import dateparser

class NLPDecisionUnit:
    def __init__(self):
        print("[INIT] NLP Engine Loaded")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.INTENTS = {
            "train_timing": [
                "when is the train",
                "train timing",
                "what time does the train leave",
                "schedule of the next train",
                "when is next train",
            ],
            "train_between": [
                "trains from place to place",
                "available trains between two stations",
                "next train between stations",
                "show trains between",
            ],
            "train_status": [
                "train status",
                "is the train delayed",
                "where is the train",
            ],
            "pnr_status": [
                "check pnr",
                "pnr status",
                "ticket confirmation",
            ],
            "fare_query": [
                "ticket price",
                "train fare",
                "how much is the ticket",
            ],
            "route": [
                "show train route",
                "route of the train",
                "what stations does the train stop at",
            ],
            "general": [
                "general question",
                "help",
                "i need help"
            ]
        }

        self.STATIONS = [
            "Kozhikode", "Kannur", "Delhi", "Mumbai",
            "Chennai", "Bengaluru", "Hyderabad"
        ]

    # ------------------------------------------------------
    #                    INTENT DETECTION
    # ------------------------------------------------------
    def extract_intent(self, text: str) -> str:
        text_emb = self.model.encode(text, normalize_embeddings=True)
        best_intent, best_score = None, -1

        for intent, examples in self.INTENTS.items():
            example_emb = self.model.encode(examples, normalize_embeddings=True)
            score = max(util.cos_sim(text_emb, example_emb)[0])
            if score > best_score:
                best_intent, best_score = intent, score

        print(f"[INTENT] {best_intent} (confidence={float(best_score):.2f})")
        return best_intent

    # ------------------------------------------------------
    #                    ENTITY EXTRACTION
    # ------------------------------------------------------
    def extract_entities(self, text: str) -> dict:
        entities = {}

        # ---- Train No. ----
        train_no = re.search(r"\b\d{4,6}\b", text)
        if train_no:
            entities["train_no"] = train_no.group()

        # ---- Candidate station tokens ----
        words = re.findall(r"[A-Za-z]+", text)
        candidate_chunks = []

        for i in range(len(words)):
            w = words[i]
            if w[0].isupper():
                candidate_chunks.append(w)
            if i < len(words) - 1:
                candidate_chunks.append(w + " " + words[i + 1])

        print(f"[DEBUG] Candidate station tokens: {candidate_chunks}")

        # ---- Fuzzy match stations ----
        stations_found = set()
        for chunk in candidate_chunks:
            match = process.extractOne(chunk, self.STATIONS, scorer=fuzz.token_set_ratio)
            if match and match[1] > 70:
                stations_found.add(match[0])

        stations_found = list(stations_found)
        if stations_found:
            entities["stations"] = stations_found

        # ------------------------------------------------------
        #         FIXED ORIGIN–DESTINATION LOGIC
        # ------------------------------------------------------
        text_lower = text.lower()

        def fuzzy_find_station(subtext):
            """Return best fuzzy match inside a phrase."""
            match = process.extractOne(subtext, self.STATIONS, scorer=fuzz.token_set_ratio)
            return match[0] if match else None

        # Case 1: BOTH "from" AND "to" present → most reliable
        if "from" in text_lower and "to" in text_lower:
            try:
                after_from = text_lower.split("from", 1)[1]
                origin_text, dest_text = after_from.split("to", 1)

                origin = fuzzy_find_station(origin_text)
                destination = fuzzy_find_station(dest_text)

                if origin:
                    entities["origin"] = origin
                if destination:
                    entities["destination"] = destination
            except:
                pass

        # Case 2: Only "from"
        elif "from" in text_lower:
            part = text_lower.split("from", 1)[1]
            origin = fuzzy_find_station(part)
            if origin:
                entities["origin"] = origin

            # destination is remaining station
            if "stations" in entities:
                others = [s for s in entities["stations"] if s != origin]
                if others:
                    entities["destination"] = others[0]

        # Case 3: Only "to"
        elif "to" in text_lower:
            part = text_lower.split("to", 1)[1]
            destination = fuzzy_find_station(part)
            if destination:
                entities["destination"] = destination

            # origin is remaining station
            if "stations" in entities:
                others = [s for s in entities["stations"] if s != destination]
                if others:
                    entities["origin"] = others[0]

        # Case 4: No keywords → fallback to text order
        else:
            if len(stations_found) >= 2:
                # sort based on position in text
                positions = {
                    s: text_lower.find(s.lower()) if text_lower.find(s.lower()) != -1 else 9999
                    for s in stations_found
                }
                ordered = sorted(stations_found, key=lambda s: positions[s])

                entities["origin"], entities["destination"] = ordered[:2]

        # ---- Date detection ----
        parsed_date = dateparser.parse(text)
        if parsed_date:
            entities["date"] = parsed_date.strftime("%Y-%m-%d")

        print(f"[ENTITIES] {entities}")
        return entities

    # ------------------------------------------------------
    #              INTENT → SERVICE ACTION ROUTER
    # ------------------------------------------------------
    def route_intent(self, intent, entities):
        if intent == "train_timing":
            return {
                "action": "get_next_train_time",
                "origin": entities.get("origin"),
                "destination": entities.get("destination"),
                "date": entities.get("date"),
            }

        elif intent == "train_between":
            return {
                "action": "get_trains_between",
                "origin": entities.get("origin"),
                "destination": entities.get("destination"),
                "date": entities.get("date"),
            }

        elif intent == "train_status":
            return {
                "action": "get_status",
                "train_no": entities.get("train_no"),
            }

        elif intent == "pnr_status":
            return {
                "action": "check_pnr",
                "pnr": entities.get("train_no"),
            }

        elif intent == "route":
            return {
                "action": "get_route",
                "train_no": entities.get("train_no")
            }

        elif intent == "fare_query":
            return {
                "action": "get_fare",
                "origin": entities.get("origin"),
                "destination": entities.get("destination"),
            }

        return {"action": "unknown", "raw": {"intent": intent, **entities}}

    def format_response(self, result):
        return f"Response: {result}"
