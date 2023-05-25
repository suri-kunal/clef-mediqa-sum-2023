#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# Imports
from pathlib import Path

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
                         AutoModelForSequenceClassification, \
                         GenerationConfig

from peft import PeftModel, PeftConfig
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
import gc
from tqdm import tqdm
import warnings
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
warnings.filterwarnings("ignore")
tqdm.pandas()


# %%


os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "online"
hf_hub.login(code_config.HF_API)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%


try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


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

with open("TaskA_and_B-label2idx.json","r") as f:
    label2idx = json.load(f)

merge_df["label"] = merge_df["section_header"].apply(lambda x: label2idx[x])

if any([True if k in code_config.MULTI_CLASS_MODEL_CHECKPOINT else False for k in ["gpt", "opt", "bloom"]]):
    padding_side = "left"
else:
    padding_side = "right"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocess_text(preds,labels):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels

scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
        'bert_scorer': (
            evaluate.load('bertscore'),
            {'model_type': 'microsoft/deberta-xlarge-mnli',"batch_size":4},
            ['precision', 'recall', 'f1'],
            ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        ),
#         'bluert': (
#             evaluate.load('bleurt', config_name='BLEURT-20'),
#             {},
#             ['scores'],
#             ['bleurt']
#         ),
    }

# %%
def calculate_metrics(references,predictions,scorer,key,save_key,**kwargs):
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        if isinstance(scores[key],list):
            if len(scores[key]) > 1:
                raise Exception("scores[key] have more than one elements")
            return scores[key][0]
        return scores[key]


# %%
def filter_and_aggregate(obj, indices):
    agg_obj = {}
    for k, v in obj.items():
        agg_obj[k] = float(np.mean([v[i] for i in indices]))
    return agg_obj

# %%

TASKA_SUMMARY_MODEL_LIST = \
[
    "suryakiran786/lora-3-fold-stratified-cv-flan-t5-large-lora",
    "suryakiran786/lora-3-fold-stratified-cv-biobart-v2-large-peft-int8"
]

wandb_kwargs = {"project": "CLEF-TaskB-metric-optimization-hpo-rouge-bertscore","group":"beam-search"}
wandbc = WeightsAndBiasesCallback(metric_name="rouge_bertscore_bleurt_score", \
                                  wandb_kwargs=wandb_kwargs, \
                                  as_multirun=True)

def metric_calculation(summary_model, \
                       df, split):
    
    @wandbc.track_in_wandb()
    def objective(trial):
        
        early_stopping = trial.suggest_categorical("early_stopping",[True])
        num_beams = trial.suggest_int("num_beams",5,10)
        no_repeat_ngram_size = trial.suggest_int("no_repeat_ngram_size",5,15)
        length_penalty = trial.suggest_float("length_penalty",-2,2,step=0.1)
        min_length = trial.suggest_categorical("min_length",[code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH])
        max_length = trial.suggest_categorical("max_length",[code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH])
        
        generate_kwargs = {
            "early_stopping": early_stopping, \
            "min_length": min_length, \
            "max_length": max_length, \
            "num_beams": num_beams, \
            "length_penalty": length_penalty, \
            "no_repeat_ngram_size": no_repeat_ngram_size
        }
    
        TASKA_SUMMARY_CHECKPOINT = summary_model

        config = PeftConfig.from_pretrained(TASKA_SUMMARY_CHECKPOINT)
        taska_summary_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, \
                                                                do_lower_case=True, \
                                                                force_download=False)
        taska_summary_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, \
                                                                    config=config, \
                                                                    force_download=False)
        taska_summary_model = PeftModel.from_pretrained(taska_summary_model, TASKA_SUMMARY_CHECKPOINT)

        taska_summary_model = taska_summary_model.to(device)
        taska_summary_model.eval()

        test_df = df
        if taska_summary_tokenizer.sep_token is None:
            taska_summary_tokenizer.sep_token = taska_summary_tokenizer.eos_token

        text_column = "dialogue_w_section_header_desc"
        test_df[text_column] = \
        test_df["section_header_desription"] + f" {str(taska_summary_tokenizer.sep_token)} " + test_df["dialogue_wo_whitespaces"]

        test_df[text_column] = code_config.TASKA_SUMMARY_PROMPT + "" + test_df[text_column]
        test_df["predicted_section_text_postprocessed"] = None
        test_df["reference_section_text_postprocessed"] = None

        summary_column = "section_text"
        text_column = "dialogue_w_section_header_desc"

        for idx in tqdm(test_df.index):
            sentence = test_df.loc[idx,text_column]
            summary = test_df.loc[idx,summary_column]        

            model_inputs = \
            taska_summary_tokenizer(sentence, \
                                    padding=code_config.TASKA_SUMMARY_PADDING, \
                                    truncation=True, \
                                    max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH, \
                                    return_tensors="pt")

            labels = \
            taska_summary_tokenizer(text_target=summary, \
                                    padding=code_config.TASKA_SUMMARY_PADDING, \
                                    truncation=True, \
                                    max_length=code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH, \
                                    return_tensors="pt")

            model_inputs["labels"] = labels["input_ids"]

            with torch.no_grad():

                input_ids = model_inputs["input_ids"].to(device)
                attention_mask = model_inputs["attention_mask"].to(device)
                labels = model_inputs["labels"].to(device)

                generated_tokens = \
                taska_summary_model.generate(inputs=input_ids, \
                                             attention_mask=attention_mask, \
                                             **generate_kwargs)

                if isinstance(generated_tokens,tuple):
                    generated_tokens = generated_tokens[0]

                generated_tokens_decoded = \
                taska_summary_tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
                labels_w_padding_tokens = \
                [[l.item() if l != -100 else taska_summary_tokenizer.pad_token_id for l in label] \
                 for label in labels.cpu()]
                labels_decoded = \
                taska_summary_tokenizer.batch_decode(labels_w_padding_tokens,skip_special_tokens=True)

                generated_tokens_decoded,labels_decoded = \
                postprocess_text(generated_tokens_decoded,labels_decoded)

                test_df.loc[idx,"predicted_section_text_postprocessed"] = \
                generated_tokens_decoded[0]

                test_df.loc[idx,"reference_section_text_postprocessed"] = \
                labels_decoded[0]

                
        references = test_df['reference_section_text_postprocessed'].tolist()
        predictions = test_df['predicted_section_text_postprocessed'].tolist()
        test_df['dataset'] = 0
        num_test = len(test_df)
        
        all_scores = {}
        for name, (scorer, kwargs, keys, save_keys) in scorers.items():
            print(name)
            scores = scorer.compute(references=references, predictions=predictions, **kwargs)
            for score_key, save_key in zip(keys, save_keys):
                all_scores[save_key] = scores[score_key]
                
        cohorts = [
        ('all', list(range(num_test))),
        ]
        
        outputs = {k: filter_and_aggregate(all_scores, idxs) for (k, idxs) in cohorts}
        
        rouge1 = outputs["all"]["rouge1"]
        rouge2 = outputs["all"]["rouge2"]
        rougeL = outputs["all"]["rougeL"]
        rougeLsum = outputs["all"]["rougeLsum"]
        bertscore_f1 = outputs["all"]["bertscore_f1"]
#         bleurt = outputs["all"]["bleurt"]
        
        return rouge1,rouge2,bertscore_f1
    
    return objective


# %%
with open("taskA_and_B_train_valid_test_split.json","r") as f:
    split_data = json.load(f)

for split, split_w_indices in split_data.items():

    train_idx = split_w_indices["train"]
    valid_idx = split_w_indices["valid"]
    test_idx = split_w_indices["test"]

    for summary_model in TASKA_SUMMARY_MODEL_LIST:
        split = int(split)
        model_name = f"{summary_model}-{split}"
        test_df = merge_df.loc[merge_df["ID"].isin(test_idx),:]
        _ = test_df.pop("ID")
        objective_fn = \
        metric_calculation(model_name, \
                           test_df, \
                           split)

        study_name = model_name.split("/")[-1]
        study = optuna.create_study(study_name=study_name, \
                                    directions=["maximize","maximize","maximize"])
<<<<<<< HEAD
        study.optimize(objective_fn, n_trials=20, callbacks=[wandbc])
=======
        study.optimize(objective_fn, n_trials=40, callbacks=[wandbc])
>>>>>>> 50ea67408e31739b074ace404ecf968fba24e63b
