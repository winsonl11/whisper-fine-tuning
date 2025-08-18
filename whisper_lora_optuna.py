# whisper_lora_optuna.py, used for finding good hyperparameters
import os
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, DatasetDict
from jiwer import wer
import optuna
from functools import partial

# config 

PROCESSED_DATA_DIR = "whisper_dataset"
OUTPUT_DIR = "whisper_base_lora_optuna2"
MODEL_NAME = "openai/whisper-base"
LANGUAGE = "en"
TASK = "transcribe"
FP16 = True
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
GRAD_ACCUM_STEPS = 2
PER_DEVICE_BATCH_SIZE = 8
NUM_EPOCHS = 4
SAVE_TOTAL_LIMIT = 3
GENERATION_MAX_LENGTH = 225

# load processor + dataset

print("Loading processor and dataset...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

dataset = DatasetDict({
    "train": load_from_disk(os.path.join(PROCESSED_DATA_DIR, "train")),
    "test": load_from_disk(os.path.join(PROCESSED_DATA_DIR, "test"))
})

print("train size:", len(dataset["train"]), "test size:", len(dataset["test"]))


# data collator

def data_collator(features):
 
    input_features = [f["input_features"] for f in features]
    input_batch = processor.feature_extractor.pad(
        {"input_features": input_features},
        return_tensors="pt"
    )

    labels = [f["labels"] for f in features]
    labels_batch = processor.tokenizer.pad(
        {"input_ids": labels}, 
        padding=True,             # pad to the longest in batch
        return_tensors="pt"
    )
    labels_ids = labels_batch["input_ids"].clone()
    labels_ids[labels_ids == processor.tokenizer.pad_token_id] = -100

    batch = {
        "input_features": input_batch["input_features"],
        "labels": labels_ids
    }
    return batch


# metrics function for trainer

def compute_metrics(pred):

    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    references = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = wer(references, predictions)
    return {"wer": wer_score}


# model initialization function for optuna

def model_init(trial):
    """
    Initializes a new model for each trial with suggested hyperparameters.
    """
    if trial is None:
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        # set padding and decoder tokens for the base model
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = "<|pad|>"
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.language = LANGUAGE
        model.config.task = TASK
        forced_decoder_ids = None
        model.config.forced_decoder_ids = forced_decoder_ids
        model.config.suppress_tokens = []
        return model
        
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # set padding and decoder tokens
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = "<|pad|>"
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    # suggest hyperparameters for LoRA
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2, step=0.05)
    
    # define target modules for LoRA
    # using a common starting point
    target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    
    # suggest a learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    
    # update training arguments
    training_args.learning_rate = learning_rate
    
    return model


# define training arguments for the search

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    num_train_epochs=NUM_EPOCHS,
    fp16=FP16,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    predict_with_generate=True,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    generation_max_length=GENERATION_MAX_LENGTH,
    metric_for_best_model="wer", # minimize WER
    greater_is_better=False,
)

# trainer Setup and hyperparameter search

print("Creating trainer and starting hyperparameter search with Optuna...")

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
)

# start the hyperparameter search

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=None,
    n_trials=10, 
)

print("\n\n-----------------")
print("Best trial found:")
print(best_trial)
print("-----------------")
