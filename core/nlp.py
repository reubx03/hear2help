from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
import re
import dateparser
from core.train_routes import TRAIN_ROUTES
from core.static_timetable import STATION_TIMETABLE
from core.train_service import STATION_ALIASES


def _extract_all_stations():
    """Extract all unique stations from routes, timetable, and aliases."""
    stations = set()
    
    # Extract from train routes
    for route in TRAIN_ROUTES.values():
        stations.update(route)
    
    # Extract from station timetable (station names as keys)
    stations.update(STATION_TIMETABLE.keys())
    
    # Extract from timetable entries (origin and destination)
    for train_list in STATION_TIMETABLE.values():
        for train in train_list:
            if "origin" in train:
                stations.add(train["origin"])
            if "destination" in train:
                stations.add(train["destination"])
    
    # Add normalized station names from aliases (the canonical names)
    stations.update(STATION_ALIASES.values())
    
    # Also add alias keys that might be useful for matching
    stations.update(STATION_ALIASES.keys())
    
    return sorted(list(stations))


def _normalize_station(name):
    """Normalize station name using aliases."""
    return STATION_ALIASES.get(name, name)


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

        # Dynamically extract all stations from routes and timetable
        self.STATIONS = _extract_all_stations()
        print(f"[INIT] Loaded {len(self.STATIONS)} stations for recognition")

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
        text_lower = text.lower()

        # ---- Train No. ----
        train_no = re.search(r"\b\d{4,6}\b", text)
        if train_no:
            entities["train_no"] = train_no.group()

        # ---- Station Detection Strategy ----
        # 1. Broad fuzzy search for all possible stations in text
        # 2. Heuristic-based origin/destination assignment
        
        # Extract candidate chunks (up to 3 words)
        words = re.findall(r"[A-Za-z]+", text)
        candidate_chunks = []
        for i in range(len(words)):
            candidate_chunks.append(words[i])
            if i < len(words) - 1:
                candidate_chunks.append(words[i] + " " + words[i+1])
            if i < len(words) - 2:
                candidate_chunks.append(words[i] + " " + words[i+1] + " " + words[i+2])

        found_stations = []
        for chunk in candidate_chunks:
            # We use a higher threshold and token_sort_ratio for better precision
            match = process.extractOne(chunk, self.STATIONS, scorer=fuzz.token_sort_ratio)
            if match and match[1] > 80:
                normalized = _normalize_station(match[0])
                # Store with its original position in the text for ordering
                pos = text_lower.find(chunk.lower())
                if pos != -1:
                    found_stations.append({
                        "name": normalized,
                        "pos": pos,
                        "chunk": chunk.lower()
                    })

        # Remove duplicates (choose best match if chunks overlap)
        found_stations.sort(key=lambda x: len(x["chunk"]), reverse=True)
        unique_stations = []
        taken_ranges = []
        for s in found_stations:
            start = s["pos"]
            end = start + len(s["chunk"])
            if not any(max(start, ts) < min(end, te) for ts, te in taken_ranges):
                unique_stations.append(s)
                taken_ranges.append((start, end))

        unique_stations.sort(key=lambda x: x["pos"])
        station_names = [s["name"] for s in unique_stations]
        entities["stations"] = list(dict.fromkeys(station_names)) # Maintain order, remove dupes

        # ---- Origin/Destination Logic (ROBUST) ----
        # Keywords for destination
        to_keywords = ["to", "towards", "for", "going to", "reaching", "at"]
        # Keywords for origin
        from_keywords = ["from", "starting at", "starting from", "leaving"]

        # Default: Origin is ALWAYS Thrippunithura
        entities["origin"] = "Thrippunithura"
        
        if not unique_stations:
            pass
        elif len(unique_stations) == 1:
            # If only one station mentioned, check if it's preceded by "from"
            match_name = unique_stations[0]["name"]
            prefix = text_lower[:unique_stations[0]["pos"]].strip()
            
            is_from = any(prefix.endswith(k) for k in from_keywords)
            # If specifically mentions "from [Station]", and it's NOT Thrippunithura,
            # we might want to respect it, but the USER SAID origin is always Thrippunithura.
            # However, if it's "to [Station]", then it's certainly a destination.
            is_to = any(prefix.endswith(k) for k in to_keywords)
            
            if is_to:
                entities["destination"] = match_name
            elif is_from:
                # User specifically said 'from X', but rule says origin is Thrippunithura.
                # We'll treat Thrippunithura as origin and X as some other context (maybe destination fallback?)
                # For now, let's treat any single non-Thrippunithura station as destination if it fits
                if match_name != "Thrippunithura":
                    entities["destination"] = match_name
            else:
                # Ambiguous single station -> assume it's the destination
                if match_name != "Thrippunithura":
                    entities["destination"] = match_name

        else:
            # Multiple stations. Check relationship.
            # Look for keywords near each station
            for i, s in enumerate(unique_stations):
                prefix = text_lower[:s["pos"]].strip()
                if any(prefix.endswith(k) for k in to_keywords):
                    entities["destination"] = s["name"]
                    break # Found a clear destination
            
            # If no destination found via keywords, use the last station if there are 2+
            if "destination" not in entities:
                # Filter out Thrippunithura from possible destinations
                others = [s["name"] for s in unique_stations if s["name"] != "Thrippunithura"]
                if others:
                    entities["destination"] = others[0]

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
