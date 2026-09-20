"""BS 2.0 additional analyses beyond Big Five, all from components already used by the pipeline:

emotions_text  - 7 emotions from the transcript (EmoRoBERTa classification head, English translation)
emotions_voice - arousal / dominance / valence of the voice (audeering wav2vec2 regression head)
face_expr      - facial expressions per frame (ViT trained on FER) and head-motion / face-visibility proxies
speech_stats   - speaking rate, pauses, filler words, sentence length and vocabulary from Whisper timestamps
"""
