class NLPDecisionUnit:
    def __init__(self):
        print("[INIT] NLP placeholder...")
        self.intents = {
            "route": "Get train route",
            "search": "Find train between stations"
        }

    def extract_intent(self, text):
        print("[INTENT] Placeholder match...")
        return "route" if "route" in text.lower() else "search"

    def extract_entities(self, text):
        print("[NER] Placeholder entity extraction...")
        # Dummy train number recognition
        train_no = "".join([c for c in text if c.isdigit()])
        return {"train_no": train_no or "00000"}

    def route_intent(self, intent, entities):
        if intent == "route":
            return {"route": "Sample Station A → B → C", "train_no": entities["train_no"]}
        return {"error": "Intent not implemented"}

    def format_response(self, result):
        return f"Response: {result}"
