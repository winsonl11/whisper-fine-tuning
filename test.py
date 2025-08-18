# test.py, file for testing WER of models
import os
import librosa
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer


# paths & config

MODEL_NAME = "./whisper_base_lora_finetuned3"  # path to model for test
TEST_METADATA = "processed/test/metadata.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load model + processor

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.forced_decoder_ids = None
model.generation_config.language = "en" # make sure it transcribes in english


# load dataset

dataset = load_dataset("csv", data_files={"test": TEST_METADATA})["test"]


# evaluate WER

all_predictions = []
all_references = []

for sample in dataset:
    audio_path = sample["path"]
    reference_text = sample["text"]

    # load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(DEVICE)

    # generate predictions
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features, 
        )

    # decode predictions
    predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    all_predictions.append(predicted_text.lower().strip())
    all_references.append(reference_text.lower().strip())

# compute WER
print(all_predictions,all_references)
error_rate = wer(all_references, all_predictions)
print(f"Post Test WER: {error_rate:.4f}")
