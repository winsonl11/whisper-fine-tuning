from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import soundfile as sf  

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

dataset = load_dataset(
    "csv",
    data_files={"train": "processed/train/metadata.csv", "test": "processed/test/metadata.csv"}
)

def prepare_dataset(batch):

    audio_array, sampling_rate = sf.read(batch["path"])

    batch["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=sampling_rate
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

tokenized_dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names
)

tokenized_dataset.save_to_disk("whisper_dataset")
