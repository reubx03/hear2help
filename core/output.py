from deep_translator import GoogleTranslator
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import torch
import uuid
import os
import soundfile as sf
from playsound import playsound


class OutputUnit:
    def __init__(self, output_dir="data/tts"):
        print("[INIT] OutputUnit loaded (Indic Parler-TTS + GoogleTranslate)")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Select CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Using device: {self.device}")

        # Load Indic Parler-TTS
        print("[TTS] Loading Indic Parler-TTS model...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts"
        ).to(self.device)

        # Tokenizers needed for prompts + descriptions
        self.prompt_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        self.desc_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)

        # Language → recommended speaker mapping (official list)
        self.speaker_map = {
            "en": "Jenny",
            "english": "Jenny",

            "hi": "Aditi",
            "hindi": "Aditi",

            "ml": "Anu",
            "malayalam": "Anu",
        }

        # Generic caption: natural, clear voice
        self.default_caption = (
            "A natural, expressive voice with clear audio, moderate speed, and close-up microphone quality."
        )


    # ----------------------------------------------------------------------
    # TRANSLATE BACK TO USER LANGUAGE (keep same signature)
    # ----------------------------------------------------------------------
    def translate_back(self, text, lang):
        if not text:
            return text

        if lang is None:
            return text

        lang = lang.lower()

        lang_map = {
            "english": "en",
            "en": "en",
            "hindi": "hi",
            "hi": "hi",
            "malayalam": "ml",
            "ml": "ml",
        }

        target = lang_map.get(lang, "en")

        if target == "en":
            return text

        try:
            # Import station names to preserve them during translation
            from core.train_routes import TRAIN_ROUTES
            from core.train_service import STATION_ALIASES
            
            # Extract all station names (longest first to avoid partial matches)
            all_stations = set()
            for route in TRAIN_ROUTES.values():
                all_stations.update(route)
            all_stations.update(STATION_ALIASES.keys())
            all_stations.update(STATION_ALIASES.values())
            
            # Replace station names with placeholders before translation
            # Sort by length descending to match longer names first
            station_placeholders = {}
            placeholder_text = text
            sorted_stations = sorted(all_stations, key=lambda x: (-len(x), x))
            
            for i, station in enumerate(sorted_stations):
                if station in placeholder_text:
                    placeholder = f"__STN{i}__"
                    station_placeholders[placeholder] = station
                    placeholder_text = placeholder_text.replace(station, placeholder)
            
            # Translate the text with placeholders
            translated = GoogleTranslator(source="en", target=target).translate(placeholder_text)
            
            # Replace placeholders back with original station names
            for placeholder, station in station_placeholders.items():
                translated = translated.replace(placeholder, station)
            
            print(f"[TRANSLATED BACK] {translated}")
            return translated
        except Exception as e:
            print(f"[TRANSLATION ERROR] {e}")
            return text


    # ----------------------------------------------------------------------
    # TTS: Speak the text using Indic Parler-TTS
    # ----------------------------------------------------------------------
    def speak(self, text, lang):
        if not text:
            return None

        lang = lang.lower()
        speaker = self.speaker_map.get(lang, "Thoma")  # fallback English
        caption = f"{speaker}'s voice speaking with natural clarity, moderate speed, and expressive tone."

        file_path = os.path.join(self.output_dir, f"{uuid.uuid4().hex}.wav")

        print(f"[TTS] Synthesizing speech for language '{lang}', speaker '{speaker}' → {file_path}")

        try:
            # Encode description + prompt
            desc_inputs = self.desc_tokenizer(caption, return_tensors="pt").to(self.device)
            prompt_inputs = self.prompt_tokenizer(text, return_tensors="pt").to(self.device)

            # Generate audio
            generated = self.model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )

            audio = generated.cpu().numpy().squeeze()

            # Save WAV
            sf.write(file_path, audio, self.model.config.sampling_rate)

            # Play out loud
            print("[AUDIO] Playing output...")
            playsound(file_path)

            return file_path

        except Exception as e:
            print(f"[TTS ERROR] {e}")
            return None
