# whisper_lora_train.py, file to actually fine tune model
import os
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, DatasetDict
from jiwer import wer

# config 

PROCESSED_DATA_DIR = "whisper_dataset"    
OUTPUT_DIR = "whisper_base_lora_finetuned"
MODEL_NAME = "openai/whisper-base"
LANGUAGE = "en"
TASK = "transcribe"   

# training hyperparams
PER_DEVICE_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LEARNING_RATE =  5.582949617313887e-06
NUM_EPOCHS = 2
FP16 = True
SAVE_STRATEGY = "steps"
EVAL_STRATEGY = "steps"

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj",
                  "fc1", "fc2"]


# load processor + dataset

print("Loading processor and dataset...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

dataset = DatasetDict({
    "train": load_from_disk(os.path.join(PROCESSED_DATA_DIR, "train")),
    "test": load_from_disk(os.path.join(PROCESSED_DATA_DIR, "test"))
})

print("train size:", len(dataset["train"]), "test size:", len(dataset["test"]))

# -----------------------
# Model + LoRA
# -----------------------
print("Loading base model and applying LoRA...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# ensure padding token id is set for generation and label padding mapping
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = "<|pad|>"

model.config.pad_token_id = processor.tokenizer.pad_token_id

# set task and language (may be deprecated?)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
model.config.forced_decoder_ids = forced_decoder_ids
model.config.suppress_tokens = []

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Data collator

def data_collator(features):

    input_features = [f["input_features"] for f in features]
    # create batched input_features
    input_batch = processor.feature_extractor.pad(
        {"input_features": input_features},
        return_tensors="pt"
    )
    # prepare labels
    labels = [f["labels"] for f in features]
    labels_batch = processor.tokenizer.pad(
        {"input_ids": labels}, 
        padding=True,             # pad to the longest in batch
        return_tensors="pt"
    )

    labels_ids = labels_batch["input_ids"].clone()
    # replace pad token id by -100 so it's ignored by loss
    labels_ids[labels_ids == processor.tokenizer.pad_token_id] = -100

    # final batch
    batch = {
        "input_features": input_batch["input_features"], 
        "labels": labels_ids
    }


    return batch


# Seq2Seq TrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    eval_steps=100,
    save_steps=100,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=FP16,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    predict_with_generate=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",  
    greater_is_better=False,      # lower WER is better
    generation_max_length=225, 
)
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=1,  # stop if no improvement after x evals
    early_stopping_threshold=0.0  # minimum WER improvement to reset patience
)
def compute_metrics(pred): # eval for early stopping

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # decode predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, language=LANGUAGE)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # ignore padding
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True, language=LANGUAGE)

    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}

# Trainer

print("Creating trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,   # needed for generation/prediction callbacks
    compute_metrics=compute_metrics,
    #callbacks=[early_stopping],
)
print(TARGET_MODULES)

# train

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Training complete. Model saved to", OUTPUT_DIR)


# evaluation: compute WER on test set

print("Running evaluation...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

preds = []
refs = []

from torch.utils.data import DataLoader

eval_dataloader = DataLoader(dataset["test"], batch_size=PER_DEVICE_BATCH_SIZE, collate_fn=data_collator)

for batch in eval_dataloader:
    # move to device
    input_features = batch["input_features"].to(device)
    # generate
    generated_ids = model.generate(input_features=input_features, max_length=225)
    # decode
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True, language="en")
    # get references
    labels = batch["labels"]
    # replace -100 with pad_token_id for decode
    labels[labels == -100] = processor.tokenizer.pad_token_id
    references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds.extend([t.lower().strip() for t in transcriptions])
    refs.extend([r.lower().strip() for r in references])

# compute WER
score = wer(refs, preds)
print(f"Test WER: {score:.4f}")
