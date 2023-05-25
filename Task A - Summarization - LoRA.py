#!/usr/bin/env python
# coding: utf-8
# %%
# Imports

from pprint import pprint
from pathlib import Path
import shutil

import datasets as ds
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from tqdm.auto import tqdm

import transformers
from filelock import FileLock
import huggingface_hub as hf_hub

from transformers import AutoConfig, \
                         AutoModelForSeq2SeqLM, \
                         AutoTokenizer, \
                         BartTokenizer, \
                         DataCollatorForSeq2Seq, \
                         SchedulerType, \
                         get_scheduler, \
                         set_seed, \
                         get_linear_schedule_with_warmup, \
                         SchedulerType, \
                         EncoderDecoderModel, \
                         Seq2SeqTrainer, \
                         Seq2SeqTrainingArguments

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    TaskType,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    prepare_model_for_int8_training
)

from bert_score import score
import evaluate
import wandb
import pandas as pd
import random

import re
import os
import config as code_config

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import math
import json
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from fire import Fire

# %%
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_hub.login(code_config.HF_API, add_to_git_credential=True)
WANDB_PROJECT = code_config.MULTI_CLASS_WANDB_PROJECT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%
config = AutoConfig.from_pretrained(code_config.TASKA_SUMMARY_MODEL_CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(code_config.TASKA_SUMMARY_MODEL_CHECKPOINT, \
                                          do_lower_case=True, \
                                          force_download=True)
label_pad_token_id = \
-100 if code_config.TASKA_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS else tokenizer.pad_token_id


# %%
def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    
    model_inputs = \
    tokenizer(inputs, \
              padding=code_config.TASKA_SUMMARY_PADDING, \
              truncation=True, \
              max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH)
    
    labels = tokenizer(text_target=targets, \
                        padding=code_config.TASKA_SUMMARY_PADDING, \
                        truncation=True, \
                        max_length=code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH)
    
    if code_config.TASKA_SUMMARY_PADDING == "max_length" and code_config.TASKA_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS:
        labels["input_ids"] = \
        [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# %%
def postprocess_text(preds,labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels


# %%
section_header_mapping = \
{"fam/sochx": ["FAMILY HISTORY","SOCIAL HISTORY"], \
"genhx": ["HISTORY of PRESENT ILLNESS"], \
"pastmedicalhx": ["PAST MEDICAL HISTORY"], \
"cc": ["CHIEF COMPLAINT"], \
"pastsurgical": ["PAST SURGICAL HISTORY"], \
"allergy": ["allergy"], \
"ros": ["REVIEW OF SYSTEMS"], \
"medications": ["medications"], \
"assessment": ["assessment"], \
"exam": ["exam"], \
"diagnosis": ["diagnosis"], \
"disposition": ["disposition"], \
"plan": ["plan"], \
"edcourse": ["EMERGENCY DEPARTMENT COURSE"], \
"immunizations": ["immunizations"], \
"imaging": ["imaging"], \
"gynhx": ["GYNECOLOGIC HISTORY"], \
"procedures": ["procedures"], \
"other_history": ["other_history"], \
"labs": ["labs"]}


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

merge_df["section_header_desription"] = \
merge_df["section_header"].apply(lambda x: " and ".join(section_header_mapping[x.lower()]))
merge_df["section_header_desription"] = merge_df["section_header_desription"].str.lower()

summary_column = "section_text"

if tokenizer.sep_token is None:
    tokenizer.sep_token = tokenizer.eos_token

if (code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE is True) and \
    (code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC is True):
    raise Exception("SUMMARY_DIALOGUE_W_SECTION_CODE and SUMMARY_DIALOGUE_W_SECTION_CODE_DESC cannot true together")
elif code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE is True:
    text_column = "dialogue_w_section_header"
    merge_df[text_column] = \
    merge_df["section_header"] + f" {str(tokenizer.sep_token)} " + merge_df["dialogue_wo_whitespaces"]    
elif code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC is True:
    text_column = "dialogue_w_section_header_desc"
    merge_df[text_column] = \
    merge_df["section_header_desription"] + f" {str(tokenizer.sep_token)} " + merge_df["dialogue_wo_whitespaces"]    
else:
    text_column = "dialogue_wo_whitespaces"
    
merge_df[text_column] = code_config.TASKA_SUMMARY_PROMPT + "" + merge_df[text_column]

with open("TaskA_and_B-label2idx.json","r") as f:
    label2idx = json.load(f)

merge_df["label"] = merge_df["section_header"].apply(lambda x: label2idx[x])

# %%
# ######## Load Metrics from HuggingFace ########
# print('Loading ROUGE, BERTScore, BLEURT from HuggingFace')
# scorers = {
#     'rouge': (
#         evaluate.load('rouge'),
#         {'use_aggregator': False},
#         ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
#         ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
#     ),
#     'bert_scorer': (
#         evaluate.load('bertscore'),
#         {'model_type': 'microsoft/deberta-xlarge-mnli'},
#         ['precision', 'recall', 'f1'],
#         ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
#     ),
#     'bluert': (
#         evaluate.load('bleurt', config_name='BLEURT-20'),
#         {},
#         ['scores'],
#         ['bleurt']
#     ),
# }


# %%
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
def generate_summarization(model,valid_dl):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    generated_data_list, reference_list = [], []
    
    if model is None:
        raise Exception("Model cannot be None")
    
    model = model.to(device)
    model.eval()
    if model.training is True:
        raise Exception("Model should not be trainable")
    
    gen_kwargs = {
        "max_length": code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH, \
        "min_length": code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH, \
        "num_beams": code_config.TASKA_SUMMARY_NUM_BEAMS
    }
    
    for valid_step, valid_batch in enumerate(valid_dl):
        input_ids = valid_batch["input_ids"].to(device)
        attention_mask = valid_batch["attention_mask"].to(device)
        labels = valid_batch["labels"].to(device)
        decoder_input_ids = valid_batch["decoder_input_ids"].to(device)
        
        generated_tokens = \
        model.generate(inputs=input_ids, \
                       attention_mask=attention_mask, \
                       **gen_kwargs)
        
        if isinstance(generated_tokens,tuple):
            generated_tokens = generated_tokens[0]
            
        generated_tokens_decoded = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
        labels_w_padding_tokens = \
        [[l.item() if l != -100 else tokenizer.pad_token_id for l in label]for label in labels.cpu()]
        labels_decoded = \
        tokenizer.batch_decode(labels_w_padding_tokens,skip_special_tokens=True)
        
        generated_tokens_decoded,labels_decoded = \
        postprocess_text(generated_tokens_decoded,labels_decoded)
        
        generated_data_list.extend(generated_tokens_decoded)
        reference_list.extend(labels_decoded)
        
    return generated_data_list, reference_list


# %%
def log_validation_data(generated_tokens_list,reference_list,score_dict,split):
    
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    table_data_dict = {f"Split_{split}_ID":np.arange(len(generated_tokens_list)), \
                      "Reference Sentence":reference_list, \
                     "Generated Sentence":generated_tokens_list}
    
    
    for k, v in score_dict.items():
        table_data_dict[k] = v
        if isinstance(v,list) or isinstance(v,np.ndarray):
            wandb.config.update({f"Final {k}":np.mean(v)})
            wandb.log({f"Final Metric/{k}":np.mean(v)})
        else:
            wandb.config.update({f"Final {k}":v})
            wandb.log({f"Final Metric/{k}":v})
    
    table_data_df = pd.DataFrame.from_dict(table_data_dict)    
    valid_table = wandb.Table(data=table_data_df)
    wandb.log({"Validation Table":valid_table})
    
    for k in score_dict:        
        wandb.log({f"Final Metric/{k}-Distribution": \
                   wandb.plot.histogram(table=valid_table,value=k,title=f"Distribution-{k}")})


# %%
def get_data_artifacts(df,train_idx,valid_idx,test_idx):
    
    train_dataset,valid_dataset,test_dataset = \
    data_creation(merge_df,train_idx,valid_idx,test_idx)
    
    return train_dataset,valid_dataset,test_dataset

def create_model():
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, \
                             inference_mode=False, \
                             r=8, \
                             lora_alpha=32, \
                             lora_dropout=0.1)

    model = \
    AutoModelForSeq2SeqLM.from_pretrained(code_config.TASKA_SUMMARY_MODEL_CHECKPOINT, \
                                              device_map="auto")

    model = get_peft_model(model,peft_config)

    return model




# %%
def main():
    
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    with open("taskA_and_B_train_valid_test_split.json","r") as f:
        split_data = json.load(f)

    for split, split_w_indices in split_data.items():
        if int(split) != 2:
            continue
        model_name = code_config.TASKA_SUMMARY_SINGLE_MODEL_NAME
        model_name = f"lora-{model_name}-{split}"
        output_dir = model_name

        run = wandb.init(project=code_config.TASKA_SUMMARY_WANDB_PROJECT,name=model_name)
        run_id = wandb.run.id

        train_idx = split_w_indices["train"]
        valid_idx = split_w_indices["valid"]
        test_idx = split_w_indices["test"]
    
        train_ds,valid_ds,test_ds = \
        get_data_artifacts(merge_df,train_idx,valid_idx,test_idx)
    

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, \
                                               label_pad_token_id=label_pad_token_id, \
                                               pad_to_multiple_of=8)
    
        num_of_batches = len(train_ds) / code_config.TASKA_SUMMARY_TRAIN_MICRO_BATCH_SIZE_PER_GPU
        num_of_batches = int(np.ceil(num_of_batches))
        total_steps = \
        code_config.TASKA_SUMMARY_EPOCHS * num_of_batches / code_config.TASKA_SUMMARY_GRADIENT_ACCUMULATION_STEPS
        total_steps = np.ceil(total_steps)
        total_steps = int(total_steps)
        num_warmup_steps = code_config.TASKA_SUMMARY_NUM_WARMUP_STEPS * total_steps
        num_warmup_steps = np.ceil(num_warmup_steps)
        num_warmup_steps = int(num_warmup_steps)
    
        # Define training args
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=code_config.TASKA_SUMMARY_LEARNING_RATE, # higher learning rate
            weight_decay=code_config.TASKA_SUMMARY_OPTIMIZER_WEIGHT_DECAY,
            adam_epsilon=code_config.TASKA_SUMMARY_OPTIMIZER_EPS,
            per_device_train_batch_size=code_config.TASKA_SUMMARY_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
            per_device_eval_batch_size=2*code_config.TASKA_SUMMARY_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
            gradient_accumulation_steps=code_config.TASKA_SUMMARY_GRADIENT_ACCUMULATION_STEPS,
            max_steps=total_steps,
            warmup_steps=num_warmup_steps,
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            seed=code_config.TASKA_SUMMARY_SEED
        )

        # Create Trainer instance
        trainer = Seq2SeqTrainer(
            model_init=create_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
        )
        trainer.train()
        trainer.model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)
        wandb.finish()

# %%
if __name__ == "__main__":
    main()
