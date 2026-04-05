import os
from dotenv import load_dotenv

load_dotenv()


def _bool(val, default=False):
    if val is None:
        return default
    return str(val).strip().lower() in ('true', '1', 'yes')


class Config:
    def __init__(self):
        self.whisper_model = os.getenv('WHISPER_MODEL', 'small')
        self.save_audio = _bool(os.getenv('SAVE_AUDIO'), default=False)
        self.save_transcript_txt = _bool(os.getenv('SAVE_TRANSCRIPT_TXT'), default=False)
        self.save_text_in_json = _bool(os.getenv('SAVE_TEXT_IN_JSON'), default=True)
        self.classifier = os.getenv('CLASSIFIER', 'ollama')  # 'ollama' or 'regex'
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'gemma3:1b')
        self.chunk_interval = int(os.getenv('CHUNK_INTERVAL', '30'))   # seconds per processing chunk
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '3'))      # seconds of overlap between chunks

    def apply_overrides(self, model=None, save_audio=None, save_transcript_txt=None, save_text_in_json=None):
        if model:
            self.whisper_model = model
        if save_audio is not None:
            self.save_audio = save_audio
        if save_transcript_txt is not None:
            self.save_transcript_txt = save_transcript_txt
        if save_text_in_json is not None:
            self.save_text_in_json = save_text_in_json


config = Config()
