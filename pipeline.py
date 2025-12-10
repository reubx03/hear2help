from core.speech import SpeechUnit
from core.nlp import NLPDecisionUnit
from core.train_service import TrainService
from core.output import OutputUnit

class AssistantPipeline:
    def __init__(self, debug: bool = True):
        self.debug = debug

        self.speech = SpeechUnit()
        self.nlp = NLPDecisionUnit()
        self.train_api = TrainService()
        self.output = OutputUnit()

        print("\n=== Railway Assistant Ready (Placeholder Mode) ===")

    def _log(self, label, val):
        if self.debug:
            print(f"[DEBUG] {label}: {val}")

    def run(self, audio_path):
        print("\n--- Pipeline Triggered ---")

        lang = self.speech.detect_language(audio_path)
        text = self.speech.speech_to_text(audio_path, lang)
        english_text = self.speech.translate_to_english(text)
        
        intent = self.nlp.extract_intent(english_text)
        entities = self.nlp.extract_entities(english_text)

        # Step 1: NLP decides what the user wants
        request = self.nlp.route_intent(intent, entities)
        
        # Step 2: Train service processes the request
        service_result = self.train_api.execute(request)
        
        # Step 3: Format final text response
        response = self.nlp.format_response(service_result)
        
        final_text = self.output.translate_back(response, lang)

        # Generate TTS and play it
        audio_path = self.output.speak(final_text, lang)
        
        return {
            "text": final_text,
            "audio": audio_path
        }
