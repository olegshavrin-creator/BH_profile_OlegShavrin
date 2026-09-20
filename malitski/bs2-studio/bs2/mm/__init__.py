"""Own Big Five model following the MM-PSYCHE recipe (LEYA-HSE, IEEE Access 2026), personality task only.

faces.py       - 30 uniform frames, MediaPipe face detection, crops (port of MM-PSYCHE video_preprocessor)
extractors.py  - frozen encoders: CLIP ViT-B/32 (face), CLAP (audio), EmoRoBERTa (transcript, behavior text)
data.py        - FIV2 index from the MM-PSYCHE csv files, audio extraction, per-modality feature cache
model.py       - MCDM fusion model (MultiModalFusionModel_v1 port) with the personality head only
train.py       - training / evaluation
"""
