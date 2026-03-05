from datetime import datetime


# --------------------------------------------------
# Spoken English time helpers
# --------------------------------------------------

EN_NUM = {
    0: "zero", 1: "one", 2: "two", 3: "three",
    4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty",
    30: "thirty", 40: "forty", 50: "fifty"
}

def _say_number(n: int) -> str:
    if n < 20:
        return EN_NUM[n]
    return EN_NUM[n // 10 * 10] + " " + EN_NUM[n % 10]


def time_to_spoken(time_24: str) -> str:
    """
    23:25 → eleven twenty five PM
    09:05 → nine five AM
    """
    try:
        h, m = map(int, time_24.split(":"))

        period = "AM"
        if h >= 12:
            period = "PM"

        if h == 0:
            h = 12
        elif h > 12:
            h -= 12

        return f"{_say_number(h)} {_say_number(m)} {period}"

    except Exception:
        return time_24


# --------------------------------------------------
# Response Formatter
# --------------------------------------------------

class ResponseFormatter:
    def format(self, response: dict) -> str:
        if not response or "type" not in response:
            return "Sorry, I could not find the information you requested."

        rtype = response["type"]

        # ----------------------------
        # NEXT TRAIN
        # ----------------------------
        if rtype == "train_timing":
            if "message" in response:
                return response["message"]

            train_name = response.get("train_name", "the train").title()
            origin = response.get("origin", "your station")
            destination = response.get("destination", "your destination")
            date = response.get("date", "today")
            is_tomorrow = response.get("is_tomorrow", False)
            next_tomorrow = response.get("next_train_tomorrow")
            destination_matched = response.get("destination_matched", True)

            dep_time = time_to_spoken(
                response.get("departure_time", "")
            )

            # Note: If destination didn't match, we're showing any available train

            if is_tomorrow:
                # Only tomorrow's train available
                if destination_matched:
                    return (
                        f"There are no more trains today. "
                        f"The next available train from {origin} to {destination} "
                        f"is the {train_name}, departing tomorrow at {dep_time}."
                    )
                else:
                    return (
                        f"There are no more trains today to {destination}. "
                        f"The next available train from {origin} "
                        f"is the {train_name}, departing tomorrow at {dep_time}."
                    )
            else:
                # Today's train available
                if destination_matched:
                    lines = [
                        f"The next train from {origin} to {destination} "
                        f"is the {train_name}, departing today at {dep_time}."
                    ]
                else:
                    lines = [
                        f"No direct trains to {destination} available. "
                        f"The next train from {origin} "
                        f"is the {train_name}, departing today at {dep_time}."
                    ]
                
                # Also mention tomorrow's next train if available
                if next_tomorrow:
                    tomorrow_name = next_tomorrow["train_name"].title()
                    tomorrow_time = time_to_spoken(next_tomorrow["departure_time"])
                    lines.append(
                        f"The next train tomorrow is the {tomorrow_name} at {tomorrow_time}."
                    )
                
                return " ".join(lines)

        # ----------------------------
        # TRAINS BETWEEN
        # ----------------------------
        if rtype == "train_between":
            trains = response.get("trains", [])
            if not trains:
                return "There are no trains available for this route today or tomorrow."

            origin = response.get("origin", "")
            destination = response.get("destination", "")
            has_tomorrow = response.get("has_tomorrow", False)

            # Separate today and tomorrow trains
            trains_today = [t for t in trains if t.get("date") == "today"]
            trains_tomorrow = [t for t in trains if t.get("date") == "tomorrow"]

            if "error" in response:
                return response["error"]

            lines = []

            if trains_today:
                lines.append(f"I found {len(trains_today)} train{'s' if len(trains_today) > 1 else ''} today from {origin} to {destination}:")
                for t in trains_today[:3]:  # limit to 3 today
                    name = t["train_name"].title()
                    dep = time_to_spoken(t.get("departure_time", ""))
                    lines.append(f"{name} at {dep}")
            
            if trains_tomorrow:
                if trains_today:
                    lines.append(f"And {len(trains_tomorrow)} train{'s' if len(trains_tomorrow) > 1 else ''} tomorrow:")
                else:
                    lines.append(f"I found {len(trains_tomorrow)} train{'s' if len(trains_tomorrow) > 1 else ''} tomorrow from {origin} to {destination}:")
                for t in trains_tomorrow[:2]:  # limit to 2 tomorrow
                    name = t["train_name"].title()
                    dep = time_to_spoken(t.get("departure_time", ""))
                    lines.append(f"{name} at {dep}")

            return ". ".join(lines) + "."

        # ----------------------------
        # ROUTE
        # ----------------------------
        if rtype == "route":
            stops = response.get("stops", [])
            if not stops:
                return "Route information is not available for this train."

            return (
                "This train passes through the following stations: "
                + ", ".join(stops)
                + "."
            )

        # ----------------------------
        # STATUS
        # ----------------------------
        if rtype == "status":
            if "error" in response:
                return response["error"]
            return f"Train {response.get('train_no')} is running as per schedule."

        # ----------------------------
        # PNR
        # ----------------------------
        if rtype == "pnr":
            return f"Your ticket status is {response.get('status', 'unknown')}."

        # ----------------------------
        # FARE
        # ----------------------------
        if rtype == "fare":
            return (
                f"The fare from {response.get('origin')} to "
                f"{response.get('destination')} is approximately "
                f"{response.get('fare')}."
            )

        return "I have the information, but I cannot phrase it properly yet."
