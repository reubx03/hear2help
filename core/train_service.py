import random
import datetime

class TrainService:
    def __init__(self):
        print("[TrainService] Ready (mock mode).")

        # persistent variable to store user's last known boarding station
        self.current_station = "Kozhikode"  # default fallback until NLP sets it

        # mock dataset
        self.mock_trains = [
            {"no": "12218", "name": "Sampark Kranti Express"},
            {"no": "16307", "name": "Kannur Intercity Express"},
            {"no": "16605", "name": "Maveli Express"},
            {"no": "12602", "name": "Chennai Mail"},
            {"no": "22638", "name": "West Coast Express"}
        ]


    # ---------------------------------------
    # Main command router
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
    # MOCK RESPONSE FUNCTIONS
    # ---------------------------------------

    def get_next_train_time(self, origin, destination, date):
        # apply global station override if origin missing
        if not origin:
            origin = self.current_station

        # update known user origin
        self.current_station = origin

        train = random.choice(self.mock_trains)
        mock_time = f"{random.randint(5, 23)}:{random.choice(['00', '15', '30', '45'])}"

        return {
            "type": "train_timing",
            "train_no": train["no"],
            "train_name": train["name"],
            "origin": origin,
            "destination": destination,
            "departure_time": mock_time,
            "date": date or "today"
        }


    def get_trains_between(self, origin, destination, date):
        if not origin:
            origin = self.current_station

        self.current_station = origin

        return {
            "type": "train_between",
            "origin": origin,
            "destination": destination,
            "date": date or "today",
            "trains": self.mock_trains[: random.randint(1, 4)]
        }


    def get_status(self, train_no):
        status = random.choice(["On Time", "Delayed 25 min", "Expected Soon", "Departed"])
        return {
            "type": "status",
            "train_no": train_no,
            "status": status
        }


    def check_pnr(self, pnr):
        return {
            "type": "pnr",
            "pnr": pnr,
            "status": random.choice(["Confirmed", "RAC", "Waiting List 12"])
        }


    def get_route(self, train_no):
        return {
            "type": "route",
            "train_no": train_no,
            "stops": [
                "Kozhikode", "Kannur", "Mangalore", "Udupi", "Goa", "Mumbai"
            ]
        }


    def get_fare(self, origin, destination):
        fare = random.randint(120, 1800)
        return {
            "type": "fare",
            "origin": origin or self.current_station,
            "destination": destination,
            "fare": f"₹{fare}"
        }
