# core/speech.py

import logging
import os
import torch
import librosa
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, AutoModel
from deep_translator import GoogleTranslator

# -----------------------------
# Singleton Model Holders
# -----------------------------
_WHISPER_MODEL = None
_INDIC_MODEL = None


class SpeechUnit:
    def __init__(self, checkpoint_path=None):
        # checkpoint_path no longer used, kept only to avoid breaking pipeline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[INIT] SpeechUnit ready - models will load on first use.")

    # -----------------------------
    # MODEL LOADERS (Lazy Singleton)
    # -----------------------------
    def _load_whisper(self):
        global _WHISPER_MODEL
        if _WHISPER_MODEL is None:
            print("[LOAD] Whisper-tiny (Language Detection Only)...")
            _WHISPER_MODEL = whisper.load_model("tiny")
        return _WHISPER_MODEL

    def _load_indic(self):
        """Lazily load IndicConformer 600M multilingual ASR model."""
        global _INDIC_MODEL
        if _INDIC_MODEL is None:
            print("[LOAD] IndicConformer-600M Multilingual (AI4Bharat)...")
            _INDIC_MODEL = AutoModel.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                trust_remote_code=True
            ).to(self.device)
        return _INDIC_MODEL

    # -----------------------------
    # RUNTIME FUNCTIONS
    # -----------------------------
    def detect_language(self, audio_path):
        model = self._load_whisper()

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)

        print(f"[LANG DETECTED] {lang}")
        return lang

    def speech_to_text(self, audio_path, lang):
        """
        Runs ASR using IndicConformer-600M model.
        """
        try:
            print(f"[ASR] Running IndicConformer (CTC mode) for {lang}…")

            model = self._load_indic()

            # load & resample audio to 16kHz mono
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)

            wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Perform CTC decoding
            result = model(wav_tensor, lang, "ctc")

            # Normalize output
            if isinstance(result, list):
                text = result[0]
            else:
                text = str(result)

            text = text.strip()

            print(f"[TEXT] → {text}")
            return text

        except Exception as e:
            logging.error(f"[ERROR] ASR Failed: {e}")
            return "[ERROR] Speech failed."

    

    def translate_to_english(self, text, lang=None):
        try:
            translated = GoogleTranslator(
                source=lang or "auto",
                target="en"
            ).translate(text)

            print(f"[TRANSLATED] {translated}")
            return translated

        except Exception as e:
            print(f"[TRANSLATION ERROR] {e}")
            return text  # Fallback: return original text if translation fails

