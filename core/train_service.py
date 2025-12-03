class TrainService:
    def __init__(self):
        print("[TrainService] Ready (mock).")

    def get_route(self, train_no):
        return {
            "train_no": train_no,
            "route": ["Station A", "Station B", "Station C"],
            "total_stops": 3
        }
