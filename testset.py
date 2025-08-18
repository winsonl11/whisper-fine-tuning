# file used for quick checks of some things
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import soundfile as sf  

MODEL_NAME = "./whisper_base_lora_finetuned2"  
TEST_METADATA = "processed/test/metadata.csv"

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.task = "transcribe"
model.config.language = "en"
print(model)