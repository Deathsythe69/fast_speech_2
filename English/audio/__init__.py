try:
    from . import audio_processing, stft, tools
except (ImportError, ValueError):
    from audio import audio_processing, stft, tools
