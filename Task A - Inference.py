#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,TrainingArguments,Trainer
import huggingface_hub as hf_hub
from pathlib import Path
import datasets as ds
import evaluate
import re
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import random
import time
import GPUtil
import wandb
import os
from tqdm import tqdm
import config as code_config
import json
from pprint import pprint
import copy
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    TaskType,
    PeftType,
    PeftConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from fire import Fire
import shutil

# %%
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_hub.login(code_config.HF_API, add_to_git_credential=True)
WANDB_PROJECT = code_config.MULTI_CLASS_WANDB_PROJECT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
train_path = Path.cwd().joinpath("2023_ImageCLEFmed_Mediqa","dataset","TaskB","TaskB-TrainingSet.csv")
validation_path = Path.cwd().joinpath("2023_ImageCLEFmed_Mediqa","dataset","TaskB","TaskB-ValidationSet.csv")
augmented_path = Path.cwd().joinpath("TaskA-augmented_data.csv")

train_df = pd.read_csv(train_path,index_col="ID")
valid_df = pd.read_csv(validation_path,index_col="ID")
valid_index = {idx:idx+train_df.shape[0] for idx in valid_df.index}
valid_df.rename(mapper=valid_index,inplace=True)
augmented_data = pd.read_csv(augmented_path,index_col="ID")
augmented_sections = augmented_data["section_header"].unique().tolist()
merge_df = pd.concat([train_df,valid_df,augmented_data],axis=0,ignore_index=False)
merge_df["dialogue_wo_whitespaces"] = merge_df["dialogue"].apply(lambda x: re.sub(r'[\r\n\s]+',' ',x))
merge_df.reset_index(inplace=True)
merge_df.rename(mapper={'index':'ID'},axis=1,inplace=True)

with open("TaskA_and_B-idx2label.json","r") as f:
    idx2label = json.load(f)

with open("TaskA_and_B-label2idx.json","r") as f:
    label2idx = json.load(f)

merge_df["label"] = merge_df["section_header"].apply(lambda x: label2idx[x])

if any([True if k in code_config.MULTI_CLASS_MODEL_CHECKPOINT else False for k in ["gpt", "opt", "bloom"]]):
    padding_side = "left"
else:
    padding_side = "right"

# %%
config = AutoConfig.from_pretrained(code_config.MULTI_CLASS_MODEL_CHECKPOINT)
config.balanced_loss = code_config.MUTLI_CLASS_BALANCE_LOSS
config.num_labels = merge_df["label"].nunique()
tokenizer = AutoTokenizer.from_pretrained(
    code_config.MULTI_CLASS_MODEL_CHECKPOINT, do_lower_case=True, force_download=True, padding_side=padding_side
)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%
def my_tokenizer(data, labels, max_length):
    complete_input_ids = []
    input_ids = []
    attention_mask = []

    for sentence in data:
        non_truncated_sentence = tokenizer.encode(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            verbose=False,
            max_length=3000,
        )
        complete_input_ids.append(non_truncated_sentence)

        tokenized_sentence = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=code_config.MULTI_CLASS_MAX_LENGTH,
            verbose=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids.append(tokenized_sentence["input_ids"])
        attention_mask.append(tokenized_sentence["attention_mask"])

    non_truncated_sentence_tensors = torch.cat(complete_input_ids, dim=0)
    input_ids_tensor = torch.cat(input_ids, dim=0)
    attention_mask_tensor = torch.cat(attention_mask, dim=0)
    labels_tensor = torch.tensor(labels.tolist())

    return (
        input_ids_tensor,
        attention_mask_tensor,
        labels_tensor,
        non_truncated_sentence_tensors,
    )


def preprocess_function(examples):
    inputs = examples["dialogue_wo_whitespaces"]
    targets = examples["label"]

    model_inputs = \
    tokenizer(inputs, \
              padding="max_length", \
              truncation=True, \
              max_length=code_config.MULTI_CLASS_MAX_LENGTH, \
              return_tensors="pt")

    model_inputs["labels"] = targets
    return model_inputs

def data_creation(df,train_idx,valid_idx,test_idx):
    train_df = df.loc[df["ID"].isin(train_idx),:]
    _ = train_df.pop("ID")
    valid_df = df.loc[df["ID"].isin(valid_idx),:]
    _ = valid_df.pop("ID")
    test_df = df.loc[df["ID"].isin(test_idx),:]
    _ = test_df.pop("ID")

    train_ds = ds.Dataset.from_pandas(train_df)
    valid_ds = ds.Dataset.from_pandas(valid_df)
    test_ds = ds.Dataset.from_pandas(test_df)

    raw_dataset = ds.DatasetDict()
    raw_dataset["train"] = train_ds
    raw_dataset["valid"] = valid_ds
    raw_dataset["test"] = test_ds

    columns = raw_dataset["train"].column_names
    processed_datasets = \
    raw_dataset.map(function=preprocess_function, \
                    batched=True, \
                    remove_columns=columns, \
                    load_from_cache_file=False, \
                    desc="Running tokenizer on dataset")

    train_dataset = processed_datasets["train"]
    valid_dataset = processed_datasets["valid"]
    test_dataset = processed_datasets["test"]

    return train_dataset,valid_dataset,test_dataset

# %%
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    return (pred_flat == labels_flat).sum() / len(labels_flat)


# %%
def log_validation_predictions(full_input_ids, input_ids, labels, logits):
    if len(full_input_ids) != len(input_ids):
        raise Exception(
            "Length of full_input_ids must be equal to length of truncated_input_ids"
        )

    if len(input_ids) != len(labels):
        raise Exception(
            "Length of truncated_input_ids must be equal to length of labels"
        )

    if len(labels) != len(logits):
        raise Exception("Length of labels must be equal to length of logits")

    columns = ["id", "full_sentence", "truncated_sentence", "label", "prediction"]
    for section in label2idx.keys():
        columns.append(f"Score_{section}")
    valid_table = wandb.Table(columns=columns)

    full_input_ids = torch.cat(full_input_ids, dim=0)
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.cat(labels, dim=0).float()
    logits = torch.cat(logits, dim=0).float()

    scores = F.softmax(logits, dim=-1)
    predictions = torch.argmax(scores, dim=-1)
    log_full_input_ids = full_input_ids
    log_truncated_input_ids = input_ids
    log_scores = scores.detach().cpu()
    log_labels = [idx2label[l.item()] for l in labels]
    log_preds = [idx2label[p.item()] for p in predictions]

    for idx, (lfs, lts, ll, lp, ls) in enumerate(
        zip(
            log_full_input_ids,
            log_truncated_input_ids,
            log_labels,
            log_preds,
            log_scores,
        )
    ):
        log_full_sentences = tokenizer.decode(lfs, skip_special_tokens=True)
        log_truncated_sentences = tokenizer.decode(lts, skip_special_tokens=True)

        sentence_id = str(idx)
        valid_table.add_data(
            sentence_id, log_full_sentences, log_truncated_sentences, ll, lp, *ls
        )
    wandb.log({"validation_table": valid_table})


# %%
def main():
    
    with open("taskA_and_B_train_valid_test_split.json","r") as f:
        split_data = json.load(f)

    for split, split_w_indices in split_data.items():

        train_idx = split_w_indices["train"]
        valid_idx = split_w_indices["valid"]
        test_idx = split_w_indices["test"]

        train_ds,valid_ds,test_ds = \
        data_creation(merge_df,train_idx,valid_idx,test_idx)

        peft_model_id = f"suryakiran786/LoRA-3-stratified-cv-Bio_ClinicalBERT-lora-{int(split)}"
        metric = evaluate.load("accuracy")

        config = PeftConfig.from_pretrained(peft_model_id)
        inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,num_labels=20)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        inference_model = PeftModel.from_pretrained(inference_model, peft_model_id)
        inference_model.to(device)
        inference_model.eval()

        test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=code_config.MULTI_CLASS_MICRO_BATCH_SIZE)

        for step, batch in enumerate(tqdm(test_dataloader)):

            input_ids = [b[None,:] for b in batch["input_ids"]]
            input_ids = torch.cat(input_ids,dim=0).to(device)
            batch["input_ids"] = input_ids.T

            token_type_ids = [b[None,:] for b in batch["token_type_ids"]]
            token_type_ids = torch.cat(token_type_ids,dim=0).to(device)
            batch["token_type_ids"] = token_type_ids.T

            attention_mask = [b[None,:] for b in batch["attention_mask"]]
            attention_mask = torch.cat(attention_mask,dim=0).to(device)
            batch["attention_mask"] = attention_mask.T

            batch["labels"] = batch["labels"][None,:].to(device)

            with torch.no_grad():
                outputs = inference_model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            predictions = predictions.detach().cpu().numpy()
            references = references.detach().cpu().numpy().squeeze(0)
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(peft_model_id)
        print(eval_metric)

#        wandb.finish()

if __name__ == "__main__":
    main()
