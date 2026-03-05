import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Singleton
_TRANSLATORS = {}

LANG_MAP = {
    "ml": "malayalam",
    "hi": "hindi",
    "en": "english"
}

MODEL_MAP = {
    ("ml", "en"): "ai4bharat/indictrans2-ml-en",
    ("hi", "en"): "ai4bharat/indictrans2-hi-en",
    ("en", "ml"): "ai4bharat/indictrans2-en-ml",
    ("en", "hi"): "ai4bharat/indictrans2-en-hi",
}


def get_translator(src, tgt):
    key = (src, tgt)
    if key not in MODEL_MAP:
        return None

    if key not in _TRANSLATORS:
        model_name = MODEL_MAP[key]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        _TRANSLATORS[key] = (tokenizer, model, device)

    return _TRANSLATORS[key]


def translate(text, src, tgt):
    bundle = get_translator(src, tgt)
    if bundle is None:
        return text  # fallback

    tokenizer, model, device = bundle

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=256)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
