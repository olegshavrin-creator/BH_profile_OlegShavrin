"""Own Big Five model following the MM-PSYCHE recipe (LEYA-HSE, IEEE Access 2026), personality task only.

faces.py       - 30 uniform frames, MediaPipe face detection, crops (port of MM-PSYCHE video_preprocessor)
extractors.py  - frozen encoders: CLIP ViT-B/32 (face), CLAP (audio), EmoRoBERTa (transcript, behavior text)
model.py       - MCDM fusion model (MultiModalFusionModel_v1 port) with the personality head only

Feature extraction and training on FIV2 live outside the package, in bs3-studio/training/.
"""
