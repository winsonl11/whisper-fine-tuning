import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import re

def segment_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        meta_file.write("path,text\n")  # CSV header

        for txt_file in Path(input_dir).rglob("*.txt"):
            wav_path = txt_file.with_suffix(".wav")

            audio, sr = librosa.load(wav_path, sr=None, mono=True)

            with open(txt_file, "r") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split("\t")

                    start_sec, end_sec, text = parts
                    start_sec = float(start_sec)
                    end_sec = float(end_sec)

                    # convert timestamps
                    start_sample = int(start_sec * sr)
                    end_sample = int(end_sec * sr)

                    segment = audio[start_sample:end_sample]

                    # trim leading and trailing silence
                    segment, _ = librosa.effects.trim(segment)

                    # filter out clips shorter than 500 ms
                    min_length_samples = int(0.5 * sr)
                    if len(segment) < min_length_samples:
                        continue

                    # resample to 16 kHz
                    target_sr = 16000
                    if sr != target_sr:
                        segment = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr

                    # normalize segment to -20 dB
                    rms = np.sqrt(np.mean(segment**2))
                    if rms > 0:
                        target_rms = 10**(-20 / 20)  # Convert dB to linear RMS
                        gain = target_rms / rms
                        segment = segment * gain

                    # save segment as WAV
                    folder_name = txt_file.parent.name
                    seg_filename = f"{folder_name}_{i}.wav"
                    seg_path = os.path.join(output_dir, seg_filename)

                    sf.write(seg_path, segment, sr)
                    text = normalize_text(text)
                    # save transcript mapping
                    meta_file.write(f"{seg_path},{text}\n")

    print(f"Segmented: {input_dir}")

def normalize_text(text: str) -> str:
    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s']", "", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # strip leading/trailing spaces
    text = text.strip()

    return text

segment_audio("data/test", "processed/test")
