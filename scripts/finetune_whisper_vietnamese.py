#!/usr/bin/env python
# coding: utf-8

import os
import gc
import psutil
import pandas as pd
from datasets import load_dataset, Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def print_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% ({memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f} GB)")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB allocated")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def load_csv_to_dataset(csv_path):
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("path", Audio(sampling_rate=16000))
    return ds

train_dataset = load_csv_to_dataset("fpt_train.csv")
val_dataset = load_csv_to_dataset("fpt_val.csv")
test_dataset = load_csv_to_dataset("fpt_test.csv")

model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

def prepare_dataset(batch):
    audio = batch["path"]
    if isinstance(audio, list):
        input_features = [processor.feature_extractor(a["array"], sampling_rate=a["sampling_rate"]).input_features[0] for a in audio]
    else:
        input_features = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    if isinstance(batch["transcription"], list):
        labels = [processor.tokenizer(text).input_ids for text in batch["transcription"]]
    else:
        labels = processor.tokenizer(batch["transcription"]).input_ids
    return {"input_features": input_features, "labels": labels}

train_dataset = train_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=4,
    num_proc=1,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=True,
    keep_in_memory=False,
    desc="Processing training data"
)
gc.collect()
val_dataset = val_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=4,
    num_proc=1,
    remove_columns=val_dataset.column_names,
    load_from_cache_file=True,
    keep_in_memory=False,
    desc="Processing validation data"
)
gc.collect()
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = TrainingArguments(
    output_dir="./whisper-vi-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    num_train_epochs=3,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    push_to_hub=False,
    remove_unused_columns=True,
    prediction_loss_only=True,
)

def compute_metrics(pred):
    try:
        from jiwer import wer
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer_score = wer(label_str, pred_str)
        del pred_str, label_str
        gc.collect()
        return {"wer": wer_score}
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"wer": 1.0}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

test_dataset = test_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=4,
    num_proc=1,
    remove_columns=test_dataset.column_names,
    load_from_cache_file=True,
    keep_in_memory=False,
    desc="Processing test data"
)
gc.collect()
test_results = trainer.evaluate(test_dataset)
print(test_results)

model.save_pretrained("./whisper-vi-finetuned")
processor.save_pretrained("./whisper-vi-finetuned")
