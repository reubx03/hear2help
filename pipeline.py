from core.speech import SpeechUnit
from core.nlp import NLPDecisionUnit
from core.train_service import TrainService
from core.output import OutputUnit
from core.response_formatter import ResponseFormatter
from core.logger import log_query


class AssistantPipeline:
    def __init__(self, debug: bool = True):
        self.debug = debug

        print("\n[BOOT] Initializing Railway Assistant...")

        self.speech = SpeechUnit()
        self.nlp = NLPDecisionUnit()
        self.train_api = TrainService()
        self.formatter = ResponseFormatter()
        self.output = OutputUnit()

        # 🔥 FORCE EVERYTHING TO LOAD AT STARTUP
        self._warmup()

        print("\n=== Railway Assistant READY (HOT) ===")

    def _warmup(self):
        """Load all models + warm GPU to avoid first-query latency"""
        try:
            print("[WARMUP] Loading ASR models...")
            self.speech._load_whisper()
            self.speech._load_indic()

            print("[WARMUP] Warming NLP embedding model...")
            _ = self.nlp.model.encode("warmup")

            print("[WARMUP] Warming TTS model...")
            _ = self.output.speak("System ready.", "en")

            print("[WARMUP] All models loaded successfully")

        except Exception as e:
            print(f"[WARMUP WARNING] {e}")

    def _log(self, label, val):
        if self.debug:
            print(f"[DEBUG] {label}: {val}")

    def run(self, audio_path, status_callback=None):
        def notify(msg):
            if status_callback:
                status_callback(msg)
            print(f"--- {msg} ---")

        notify("Cleaning background noise")
        audio_path = self.speech.preprocess_audio(audio_path)

        # Step 0: Speech to Text and Language Detection
        notify("Detecting language")
        lang = self.speech.detect_language(audio_path)
        
        notify(f"Transcribing {lang} speech")
        text = self.speech.speech_to_text(audio_path, lang)
        
        notify("Translating to English")
        english_text = self.speech.translate_to_english(text)

        notify("Extracting intent and entities")
        intent = self.nlp.extract_intent(english_text)
        entities = self.nlp.extract_entities(english_text)

        # Step 1: NLP decides what the user wants
        notify("Routing request")
        request = self.nlp.route_intent(intent, entities)

        # Step 2: Train service processes the request
        notify("Looking for train details")
        service_result = self.train_api.execute(request)

        # Step 3: Format final text response
        notify("Formatting response")
        response = self.formatter.format(service_result)

        notify(f"Translating back to {lang}")
        final_text = self.output.translate_back(response, lang)

        # Logging
        log_query(
            raw_text=text,
            lang=lang,
            intent=intent,
            entities=entities,
            response=final_text
        )

        # Generate TTS (audio file path returned)
        notify("Synthesizing audio response")
        audio_path = self.output.speak(final_text, lang)

        return {
            "text": final_text,
            "audio": audio_path
        }
