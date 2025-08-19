# file used for quick checks of some things
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import soundfile as sf  
import json
import pandas as pd
import matplotlib.pyplot as plt
"""
MODEL_NAME = "./whisper_base_lora_finetuned2"  
TEST_METADATA = "processed/test/metadata.csv"

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME
)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.task = "transcribe"
model.config.language = "en"
print(model)
"""
with open("whisper_base_lora_finetuned3/checkpoint-400/trainer_state.json") as f:
    trainer_state = json.load(f)

df = pd.DataFrame(trainer_state['log_history'])

df['learning_rate'] = df['learning_rate'].interpolate()
plt.plot(df['epoch'], df['learning_rate'], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.show()