# Model Licenses Declaration

All models used in this submission are commercially friendly (permissive open-source).

## Models

| Model | Parameters | License | Source |
|-------|-----------|---------|--------|
| faster-whisper (whisper small) | 244M | MIT | https://github.com/SYSTRAN/faster-whisper |
| Whisper (small) base weights | 244M | MIT | https://github.com/openai/whisper |
| Oriserve/Whisper-Hindi2Hinglish-Swift | 72.6M | Apache-2.0 | https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Swift |
| Silero VAD | ~1M | MIT | https://github.com/snakers4/silero-vad |

## Runtime Dependencies

| Library | License | Notes |
|---------|---------|-------|
| faster-whisper (CTranslate2) | MIT | CPU inference engine for fast path |
| transformers | Apache-2.0 | HuggingFace transformers for specialist model |
| torch | BSD-3 | PyTorch runtime |
| numpy | BSD-3 | Audio processing |
| soundfile | BSD-3 | Audio I/O |
| silero-vad | MIT | Voice activity detection |

All models load locally from HuggingFace cache during warmup. No network calls
during the scored run (verified by offline_guard).
