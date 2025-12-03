class OutputUnit:
    def __init__(self):
        print("[INIT] OutputUnit loaded")

    def translate_back(self, text, lang):
        print("[TRANSLATE BACK] Placeholder...")
        return text

    def speak(self, text, lang):
        print(f"[TTS] Speaking ({lang}):", text)
        return "audio_output_placeholder.wav"
