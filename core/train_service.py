import random
from datetime import datetime
from core.static_timetable import STATION_TIMETABLE
from core.train_routes import TRAIN_ROUTES


STATION_ALIASES = {
    "Kozhikode": "Kozhikkode",
    "Calicut": "Kozhikkode",
    "കോഴിക്കോട്": "Kozhikkode",
    "ERS": "Ernakulam Jn",
    "TVC": "Thiruvananthapuram Central",
    "Kollam": "Kollam Jn",
    "Chennai": "Chennai Central"
}

def normalize_station(name):
    return STATION_ALIASES.get(name, name)

# --------------------------------------------------
# Time helpers (UNCHANGED)
# --------------------------------------------------

def _time_to_minutes(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m

def _now_minutes():
    now = datetime.now()
    return now.hour * 60 + now.minute

def _today_idx():
    return datetime.now().weekday()

def _tomorrow_idx():
    return (datetime.now().weekday() + 1) % 7


# --------------------------------------------------
# Train Service
# --------------------------------------------------

class TrainService:
    def __init__(self):
        print("[TrainService] Ready (STATIC timetable + ROUTE mode).")
        self.current_station = "Thrippunithura"


    # ---------------------------------------
    # Main command router (UNCHANGED)
    # ---------------------------------------
    def execute(self, request: dict):
        action = request.get("action")

        if action == "get_next_train_time":
            return self.get_next_train_time(
                request.get("origin"),
                request.get("destination"),
                request.get("date")
            )

        elif action == "get_trains_between":
            return self.get_trains_between(
                request.get("origin"),
                request.get("destination"),
                request.get("date")
            )

        elif action == "get_status":
            return self.get_status(request.get("train_no"))

        elif action == "check_pnr":
            return self.check_pnr(request.get("pnr"))

        elif action == "get_route":
            return self.get_route(request.get("train_no"))

        elif action == "get_fare":
            return self.get_fare(
                request.get("origin"),
                request.get("destination")
            )

        return {"error": f"Unknown action: {action}"}


    # ---------------------------------------
    # ROUTE-AWARE STATIC LOGIC
    # ---------------------------------------

    def _valid_direction(self, train_no, origin, destination):
        route = TRAIN_ROUTES.get(train_no)
        if not route:
            return False

        if origin not in route or destination not in route:
            return False

        return route.index(origin) < route.index(destination)


    def get_next_train_time(self, origin, destination, date):
        # Always use Thrippunithura as origin
        origin = "Thrippunithura"
        
        if destination:
            destination = normalize_station(destination)
        now_min = _now_minutes()
        today = _today_idx()
        tomorrow = _tomorrow_idx()

        upcoming_today = []
        upcoming_tomorrow = []
        upcoming_today_any = []  # All trains if destination doesn't match
        upcoming_tomorrow_any = []

        # Check today's trains
        for t in STATION_TIMETABLE.get(origin, []):
            if today not in t["running_days"]:
                continue

            dep_min = _time_to_minutes(t["departure_time"])
            if dep_min <= now_min:
                continue

            # Always add to "any" list for fallback
            upcoming_today_any.append((dep_min, t, "today"))

            if destination:
                if self._valid_direction(
                    t["train_no"], origin, destination
                ):
                    upcoming_today.append((dep_min, t, "today"))
            else:
                # No destination specified, show all trains
                upcoming_today.append((dep_min, t, "today"))

        # Check tomorrow's trains
        for t in STATION_TIMETABLE.get(origin, []):
            if tomorrow not in t["running_days"]:
                continue

            dep_min = _time_to_minutes(t["departure_time"])

            # Always add to "any" list for fallback
            upcoming_tomorrow_any.append((dep_min, t, "tomorrow"))

            if destination:
                if self._valid_direction(
                    t["train_no"], origin, destination
                ):
                    upcoming_tomorrow.append((dep_min, t, "tomorrow"))
            else:
                # No destination specified, show all trains
                upcoming_tomorrow.append((dep_min, t, "tomorrow"))

        # If destination specified but no matches, fall back to showing any available train
        destination_matched = True
        if destination and not upcoming_today and not upcoming_tomorrow:
            # Check if destination is even in our database at all
            known_stations = set()
            for route in TRAIN_ROUTES.values():
                known_stations.update(route)
            
            if destination not in known_stations:
                return {
                    "type": "train_timing",
                    "origin": origin,
                    "destination": destination,
                    "message": f"I'm sorry, I don't have information for trains reaching {destination}. However, here are the next trains from {origin}."
                }

            if upcoming_today_any or upcoming_tomorrow_any:
                upcoming_today = upcoming_today_any
                upcoming_tomorrow = upcoming_tomorrow_any
                destination_matched = False  # Mark that destination didn't match

        # Get the next train today
        next_train_today = None
        if upcoming_today:
            _, next_train_today, _ = min(upcoming_today, key=lambda x: x[0])

        # Get the next train tomorrow
        next_train_tomorrow = None
        if upcoming_tomorrow:
            _, next_train_tomorrow, _ = min(upcoming_tomorrow, key=lambda x: x[0])

        # If no trains at all
        if not next_train_today and not next_train_tomorrow:
            return {
                "type": "train_timing",
                "origin": origin,
                "destination": destination,
                "message": "No trains available today or tomorrow"
            }

        # Return both today and tomorrow's next train
        result = {
            "type": "train_timing",
            "origin": origin,
            "destination": destination,
            "destination_matched": destination_matched if destination else True
        }

        if next_train_today:
            result.update({
                "train_no": next_train_today["train_no"],
                "train_name": next_train_today["train_name"],
                "departure_time": next_train_today["departure_time"],
                "date": date or "today",
                "is_tomorrow": False
            })
            if next_train_tomorrow:
                result.update({
                    "next_train_tomorrow": {
                        "train_no": next_train_tomorrow["train_no"],
                        "train_name": next_train_tomorrow["train_name"],
                        "departure_time": next_train_tomorrow["departure_time"]
                    }
                })
        else:
            # Only tomorrow's train available
            result.update({
                "train_no": next_train_tomorrow["train_no"],
                "train_name": next_train_tomorrow["train_name"],
                "departure_time": next_train_tomorrow["departure_time"],
                "date": date or "tomorrow",
                "is_tomorrow": True
            })

        return result


    def get_trains_between(self, origin, destination, date):
        # Always use Thrippunithura as origin
        origin = "Thrippunithura"

        if destination:
            destination = normalize_station(destination)
        today = _today_idx()
        tomorrow = _tomorrow_idx()
        now_min = _now_minutes()
        trains_today = []
        trains_tomorrow = []

        # Get today's trains (only future ones)
        for t in STATION_TIMETABLE.get(origin, []):
            if today not in t["running_days"]:
                continue

            dep_min = _time_to_minutes(t["departure_time"])
            if dep_min <= now_min:
                continue

            if destination:
                if not self._valid_direction(
                    t["train_no"], origin, destination
                ):
                    continue

            trains_today.append({
                "train_no": t["train_no"],
                "train_name": t["train_name"],
                "departure_time": t["departure_time"],
                "arrival_time": t["arrival_time"],
                "date": "today"
            })

        # Get tomorrow's trains
        for t in STATION_TIMETABLE.get(origin, []):
            if tomorrow not in t["running_days"]:
                continue

            if destination:
                if not self._valid_direction(
                    t["train_no"], origin, destination
                ):
                    continue

            trains_tomorrow.append({
                "train_no": t["train_no"],
                "train_name": t["train_name"],
                "departure_time": t["departure_time"],
                "arrival_time": t["arrival_time"],
                "date": "tomorrow"
            })

        # Combine: today's trains first, then tomorrow's
        all_trains = trains_today + trains_tomorrow

        if not all_trains and destination:
            # Check if destination is known
            known_stations = set()
            for route in TRAIN_ROUTES.values():
                known_stations.update(route)
            if destination not in known_stations:
                return {
                    "type": "train_between",
                    "origin": origin,
                    "destination": destination,
                    "error": f"I don't have any route information for {destination} in my database."
                }

        return {
            "type": "train_between",
            "origin": origin,
            "destination": destination,
            "date": date or "today and tomorrow",
            "trains": all_trains,
            "has_tomorrow": len(trains_tomorrow) > 0
        }


    # ---------------------------------------
    # AUXILIARY (now correct)
    # ---------------------------------------

    def get_route(self, train_no):
        return {
            "type": "route",
            "train_no": train_no,
            "stops": TRAIN_ROUTES.get(train_no, [])
        }

    def get_status(self, train_no):
        if not train_no or train_no not in TRAIN_ROUTES:
            return {
                "type": "status",
                "train_no": train_no,
                "status": "Unknown",
                "error": "Train number not found in our database."
            }
        return {
            "type": "status",
            "train_no": train_no,
            "status": "Running as per schedule"
        }

    def check_pnr(self, pnr):
        return {
            "type": "pnr",
            "pnr": pnr,
            "status": "Confirmed"
        }

    def get_fare(self, origin, destination):
        # Always use Thrippunithura as origin
        return {
            "type": "fare",
            "origin": "Thrippunithura",
            "destination": destination,
            "fare": "₹350"
        }
