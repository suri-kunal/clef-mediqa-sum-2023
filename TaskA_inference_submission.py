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
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preprocess_function(examples,tokenizer):
    inputs = examples["dialogue_wo_whitespaces"]

    model_inputs = \
    tokenizer(inputs, \
              padding="max_length", \
              truncation=True, \
              max_length=code_config.MULTI_CLASS_MAX_LENGTH, \
              return_tensors="pt")

    return model_inputs

def data_creation(df,test_idx,tokenizer):
    test_df = df.loc[df["ID"].isin(test_idx),:]
    _ = test_df.pop("ID")

    test_ds = ds.Dataset.from_pandas(test_df)

    raw_dataset = ds.DatasetDict()
    raw_dataset["test"] = test_ds

    columns = raw_dataset["test"].column_names
    processed_datasets = \
    raw_dataset.map(function=preprocess_function, \
                    fn_kwargs={"tokenizer":tokenizer}, \
                    batched=True, \
                    remove_columns=columns, \
                    load_from_cache_file=False, \
                    desc="Running tokenizer on dataset")

    test_dataset = processed_datasets["test"]

    return test_dataset

# %%
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    return (pred_flat == labels_flat).sum() / len(labels_flat)


# %%
def main(filename):

    if not filename.endswith("csv"):
        raise Exception(f"{filename} must be a csv file")

    test_path = Path(filename)

    test_df = pd.read_csv(test_path,index_col="ID")
    test_df["dialogue_wo_whitespaces"] = test_df["dialogue"].apply(lambda x: re.sub(r'[\r\n\s]+',' ',x))
    test_df.reset_index(inplace=True)
    test_df.rename(mapper={'index':'ID'},axis=1,inplace=True)

    with open("TaskA_and_B-idx2label.json","r") as f:
        idx2label = json.load(f)

    with open("TaskA_and_B-label2idx.json","r") as f:
        label2idx = json.load(f)
    
    for idx in tqdm(test_df.index,desc="Multi Class"):
        sentence = test_df.loc[idx,"dialogue_wo_whitespaces"]
        preds_list = []
        for split in [0,1,2]:
            peft_model_id = f"suryakiran786/LoRA-3-stratified-cv-Bio_ClinicalBERT-lora-{split}"
            config = PeftConfig.from_pretrained(peft_model_id)
            inference_model = \
            AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,num_labels=20)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            inference_model = PeftModel.from_pretrained(inference_model, peft_model_id)
            inference_model.to(device)
            inference_model.eval()

            batch = \
            tokenizer(sentence, \
                      padding="max_length", \
                      truncation=True, \
                      max_length=code_config.MULTI_CLASS_MAX_LENGTH, \
                      return_tensors="pt")

            with torch.no_grad():
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = inference_model(**batch)
            preds_list.append(outputs.logits)
        # Ensembling BERT Models
        preds_tensor = torch.cat(preds_list,dim=0)
        preds_tensor = preds_tensor.mean(dim=0).squeeze(0)
        preds_tensor = preds_tensor.detach().cpu().numpy()
        best_idx = np.argmax(preds_tensor).item()
        section_header = idx2label[str(best_idx)]
        test_df.loc[idx,"SystemOutput"] = section_header.upper()

    test_df.rename(mapper={"ID":"TestID"},axis=1,inplace=True)
    test_df[["TestID","SystemOutput"]].to_csv("taskA_SuryaKiran_run1_mediqaSum.csv",index=False)

if __name__ == "__main__":
    Fire(main)
