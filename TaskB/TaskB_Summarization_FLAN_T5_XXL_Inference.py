#!/usr/bin/env python
# coding: utf-8
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
from sentence_transformers import SentenceTransformer, util
from optuna.integration.wandb import WeightsAndBiasesCallback
from fire import Fire
warnings.filterwarnings("ignore")
tqdm.pandas()


# %%
sentence_model = SentenceTransformer('all-mpnet-base-v2')

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
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# %%
# early_stopping = trial.suggest_categorical("early_stopping",[True])
# num_beams = trial.suggest_int("num_beams",5,10)
# no_repeat_ngram_size = trial.suggest_int("no_repeat_ngram_size",5,15)
# length_penalty = trial.suggest_float("length_penalty",-2,2,step=0.1)
# min_length = trial.suggest_categorical("min_length",[code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH])
# max_length = trial.suggest_categorical("max_length",[code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH])

# generate_kwargs = {
#     "early_stopping": early_stopping, \
#     "min_length": min_length, \
#     "max_length": max_length, \
#     "num_beams": num_beams, \
#     "length_penalty": length_penalty, \
#     "no_repeat_ngram_size": no_repeat_ngram_size
# }

# %%
TASKA_SUMMARY_MODEL_LIST = \
[
    "lora-3-fold-stratified-cv-flan-t5-large-lora",
]

def summary_generation(summary_model, \
                       section_description, \
                       dialogue):
        
        model_name = summary_model.split("/")[-1]
        oof_scores = pd.read_csv("TaskB_OOF_Scores.csv")
        
        early_stopping = oof_scores.loc[oof_scores["Tags"] == model_name,"early_stopping"].item()
        min_length = oof_scores.loc[oof_scores["Tags"] == model_name,"min_length"].item()
        max_length = oof_scores.loc[oof_scores["Tags"] == model_name,"max_length"].item()
        num_beams = oof_scores.loc[oof_scores["Tags"] == model_name,"num_beams"].item()
        length_penalty = oof_scores.loc[oof_scores["Tags"] == model_name,"length_penalty"].item()
        no_repeat_ngram_size = oof_scores.loc[oof_scores["Tags"] == model_name,"no_repeat_ngram_size"].item()
        
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
        
        if taska_summary_tokenizer.sep_token is None:
            taska_summary_tokenizer.sep_token = taska_summary_tokenizer.eos_token
        
        sentence_to_be_summarized = \
        section_description + \
        f" {str(taska_summary_tokenizer.sep_token)} " + \
        dialogue

        model_inputs = \
        taska_summary_tokenizer(sentence_to_be_summarized, \
                                padding=code_config.TASKA_SUMMARY_PADDING, \
                                truncation=True, \
                                max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH, \
                                return_tensors="pt")

        with torch.no_grad():

            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)

            generated_tokens = \
            taska_summary_model.generate(inputs=input_ids, \
                                         attention_mask=attention_mask, \
                                         **generate_kwargs)

        if isinstance(generated_tokens,tuple):
            generated_tokens = generated_tokens[0]

        generated_tokens_decoded = \
        taska_summary_tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
        
        return generated_tokens_decoded[0]


# %%
def calculating_similarity(summary_dict):
    embedding_dict = {}
    for model_name,summary in summary_dict.items():
        embeddings = sentence_model.encode(summary,convert_to_tensor=True)
        embeddings = embeddings.detach().cpu()
        embedding_dict[model_name] = embeddings.numpy()

    similarity_dict = {}
    for model_name_1,embeddings_1 in embedding_dict.items():
        similarity_list = []
        for model_name_2,embeddings_2 in embedding_dict.items():
            if model_name_1 != model_name_2:
                cosine_sim = util.cos_sim(embeddings_1,embeddings_2).item()
                similarity_list.append(cosine_sim)
        avg_cosine_sim = np.mean(similarity_list)
        similarity_dict[model_name_1] = avg_cosine_sim
    return similarity_dict


# %%
def main(filepath):
    if not filepath.endswith("csv"):
        raise Exception("File must be a csv file")    
    
    test_path = Path(filepath)
    test_df = pd.read_csv(test_path,index_col="ID")
    test_df["dialogue_wo_whitespaces"] = test_df["dialogue"].apply(lambda x: re.sub(r'[\r\n\s]+',' ',x))
    test_df.reset_index(inplace=True)
    test_df.rename(mapper={'index':'ID'},axis=1,inplace=True)

    test_df["section_header_desription"] = \
    test_df["section_header"].apply(lambda x: " and ".join(section_header_mapping[x.lower()]))
    test_df["section_header_desription"] = test_df["section_header_desription"].str.lower()
    
    test_df["SystemOutput"] = None
    test_df["model_name"] = None
    
    for idx in tqdm(test_df.index.tolist(),desc="Summary"):
        section_header_desription = test_df.loc[idx,"section_header_desription"]
        dialogue_wo_whitespaces = test_df.loc[idx,"dialogue_wo_whitespaces"]
        summary_dict = dict()
        for split in [0,1,2]:
            for model in TASKA_SUMMARY_MODEL_LIST:
                model_name = f"suryakiran786/{model}-{split}"
                summary = summary_generation(model_name,
                                            section_header_desription, \
                                            dialogue_wo_whitespaces)
                summary_dict[model_name] = summary
        similarity_dict = calculating_similarity(summary_dict)
            
        model_name_list,similarity_list = zip(*similarity_dict.items())
        
        best_similarity_index = np.argmax(similarity_list)
        best_model = model_name_list[best_similarity_index]
        best_summary = summary_dict[best_model]
        test_df.loc[idx,"SystemOutput"] = best_summary
        test_df.loc[idx,"model_name"] = best_model

    test_df = test_df[["ID","SystemOutput","model_name"]]
    test_df.rename(mapper={"ID":"TestID"}, \
                    axis=1, \
                    inplace=True)
    test_df.to_csv("taskB_SuryaKiran_run2_mediqaSum.csv",index=False)

# %%
if __name__ == "__main__":
    Fire(main)
