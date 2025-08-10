import os
from pydub import AudioSegment
from pathlib import Path

#segments audio files based on timestamps

def segment_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        meta_file.write("path,text\n")  # CSV header

        for txt_file in Path(input_dir).rglob("*.txt"):
            wav_path = txt_file.with_suffix(".wav")
            
            audio = AudioSegment.from_wav(wav_path)
            with open(txt_file, "r") as f:

                for i, line in enumerate(f):

                    parts = line.strip().split("\t")
                    start_sec, end_sec, text = parts
                    start_ms = float(start_sec) * 1000
                    end_ms = float(end_sec) * 1000
                    
                    segment = audio[start_ms:end_ms]

                    segment = segment.set_frame_rate(16000).set_channels(1) #standardize sample rate and audio channel
                    # save segment
                    folder_name = txt_file.parent.name
                    seg_filename = f"{folder_name}_{i}.wav"
                    seg_path = os.path.join(output_dir, seg_filename)
                    segment.export(seg_path, format="wav")

                    # save transcript mapping
                    meta_file.write(f"{seg_path},{text}\n")

    print(f"Segmented: {input_dir}")

segment_audio("data/train", "processed/train")
