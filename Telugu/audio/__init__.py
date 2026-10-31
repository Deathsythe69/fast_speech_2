try:
    from . import tools
    from . import stft
    from . import audio_processing
except (ImportError, ValueError):
    import audio.tools as tools
    import audio.stft as stft
    import audio.audio_processing as audio_processing
