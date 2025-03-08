# SpeechT5 Fine-Tuning for Dutch TTS

## Project Overview

This project fine-tunes the SpeechT5 text-to-speech (TTS) model for Dutch using the Facebook VoxPopuli dataset. The process includes dataset preparation, text normalization, speaker embedding extraction, model training, and inference to generate high-quality Dutch speech from text.

## Dataset

- **Source:** Facebook VoxPopuli (`nl` subset for Dutch language)
- **Preprocessing Steps:**
  - Resampling audio to 16kHz.
  - Normalizing text by replacing unsupported characters.
  - Filtering speakers with 100-400 examples to ensure a balanced dataset.
  - Generating speaker embeddings using `speechbrain/spkrec-xvect-voxceleb`.

## Model and Preprocessing

- **Model Used:** `microsoft/speecht5_tts`
- **Preprocessing Steps:**
  - Tokenization using `SpeechT5Processor`.
  - Generating log-mel spectrograms as training labels.
  - Creating speaker embeddings with SpeechBrain.
  - Filtering out samples exceeding 200 tokens.

## Training Setup

- **Framework:** Hugging Face `transformers`
- **Training Configuration:**
  - **Learning rate:** `1e-5`
  - **Batch size:** `2` (with gradient accumulation)
  - **Warmup steps:** `500`
  - **Total steps:** `2000`
  - **Mixed precision training:** `fp16=True`
  - **Model checkpointing and evaluation:** every `1000` steps
  - **Logging:** TensorBoard
- **Trainer Used:** `Seq2SeqTrainer`

## Inference

- The fine-tuned model generates Dutch speech from text.
- The `microsoft/speecht5_hifigan` vocoder converts generated mel spectrograms into audio.
- The final speech output can be played back using `IPython.display.Audio`.

## Fine-Tuned Model

The fine-tuned model is available on the Hugging Face Hub:  
🔗 [`AhmedNabil1/speecht5_finetuned_voxpopuli_dutch`](https://huggingface.co/AhmedNabil1/speecht5_finetuned_voxpopuli_dutch)

---

## 🚀 Try the model and contribute!

