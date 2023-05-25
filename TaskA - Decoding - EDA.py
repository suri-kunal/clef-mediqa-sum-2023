# Imports
from pathlib import Path
import time
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
import plotly.express as px

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
from sentence_transformers import SentenceTransformer, util
import openai
from fire import Fire
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
warnings.filterwarnings("ignore")
tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = code_config.OPENAI_API
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["WANDB_MODE"] = "online"

api = wandb.Api()
entity, project = "kunal-suri-ml-experiments", "TaskA-metric-optimization-hpo-rouge-bertscore"  # set to your entity and project 
runs = api.runs(entity + "/" + project)
master_df = None
for run in runs:
    if "rouge_bertscore_bleurt_score_0" not in run.summary._json_dict:
        continue
    rouge1 = run.summary._json_dict["rouge_bertscore_bleurt_score_0"]
    rouge2 = run.summary._json_dict["rouge_bertscore_bleurt_score_1"]
    bert_score_f1 = run.summary._json_dict["rouge_bertscore_bleurt_score_2"]
    metric = (rouge1 + rouge2 + bert_score_f1)/3
    tag = run.tags[0]
    num_beams = run.config["num_beams"]
    early_stopping = run.config["early_stopping"]
    length_penalty = run.config["length_penalty"]
    no_repeat_ngram_size = run.config["no_repeat_ngram_size"]
    data_dict = \
    {
        "tag": [tag], \
        "metric":[metric], \
        "num_beams":[num_beams], \
        "early_stopping":[early_stopping], \
        "length_penalty":[length_penalty], \
        "no_repeat_ngram_size":[no_repeat_ngram_size]
    }
    tmp_df = \
    pd.DataFrame.from_dict(data_dict)
    if master_df is None:
        master_df = tmp_df
    else:
        master_df = pd.concat([master_df,tmp_df],axis=0,ignore_index=True)
score_df = master_df.groupby("tag")["metric"].max().reset_index()
score_df_2 = pd.merge(master_df,score_df,left_on=["tag","metric"],right_on=["tag","metric"])
ngram = score_df_2.groupby("tag")["no_repeat_ngram_size"].max().astype(np.int16).reset_index()
score_df_3 = \
pd.merge(score_df_2,ngram,left_on=["tag","no_repeat_ngram_size"],right_on=["tag","no_repeat_ngram_size"])
score_df_3.drop_duplicates(inplace=True)
beam = score_df_3.groupby("tag")["num_beams"].max().astype(np.int16).reset_index()
score_df_4 = \
pd.merge(score_df_3,beam,left_on=["tag","num_beams"],right_on=["tag","num_beams"])
length = score_df_4.groupby("tag")["length_penalty"].max().reset_index()
score_df_5 = \
pd.merge(score_df_4,length,left_on=["tag","length_penalty"],right_on=["tag","length_penalty"])
early_stopping = score_df_2[["tag","early_stopping"]].drop_duplicates()
score_df_6 = \
pd.merge(score_df_5,early_stopping,left_on="tag",right_on="tag")

score_df_6 = score_df_6.sort_values(by="metric",ascending=False).reset_index(drop=True)
score_df_6

px.line(data_frame=score_df_6,x="tag",y="metric")

score_df_6 = score_df_6.loc[score_df_6.index < 16]
score_df_6

hyper_param_json = dict()
for idx, row in score_df_6.iterrows():
    tag = row["tag"]
    print(tag)
    section = tag.split("-")[-2]
    if section not in hyper_param_json:
        hyper_param_json[section] = dict()
    hyper_param_json[section][tag] = \
    {
        "num_beams": row["num_beams"], \
        "early_stopping": row["early_stopping_x"], \
        "length_penalty": row["length_penalty"], \
        "no_repeat_ngram_size": row["no_repeat_ngram_size"]  
    }
with open("taska_summary_configuration_max.json","w") as f:
    json.dump(hyper_param_json,f,indent=2)
