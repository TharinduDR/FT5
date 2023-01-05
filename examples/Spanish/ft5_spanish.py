import math
import os

import argparse
import statistics

import numpy as np
import torch
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ft5.args import T5Args
from ft5.evaluation import sentence_label_evaluation
from ft5.t5_model import T5Model
import pandas as pd

FOLDS = 1
SEED = 777
macro_f1_scores = []
weighted_f1_scores = []

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="t5-base")
parser.add_argument('--model_type', required=False, help='model type', default="t5")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

model_type = arguments.model_type
model_name = arguments.model_name
cuda_device = int(arguments.cuda_device)

# os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_device)

for i in range(FOLDS):
    spanish_train = pd.read_csv("https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/training_set.tsv", sep="\t")
    spanish_test = pd.read_csv("https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/test_set.tsv", sep="\t")
    spanish_train["prefix"] = "olid_a"
    spanish_train['label'] = spanish_train['label'].replace(['OFP', 'OFG', 'NOE'], 'OFF')
    spanish_train['label'] = spanish_train['label'].replace(['NO'], 'NOT')

    spanish_test['label'] = spanish_test['label'].replace(['OFP', 'OFG', 'NOE'], 'OFF')
    spanish_test['label'] = spanish_test['label'].replace(['NO'], 'NOT')

    spanish_train = spanish_train.rename(columns={'comment': 'input_text', 'label': 'target_text'})
    spanish_train = spanish_train[["prefix", "input_text", "target_text"]]

    train_df, eval_df = train_test_split(spanish_train, test_size=0.2, random_state=SEED * i)

    model_args = T5Args()
    model_args.num_train_epochs = 5
    model_args.no_save = False
    model_args.fp16 = False
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 8
    model_args.max_length = 3
    model_args.max_seq_length = 128
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = int(
        math.floor(len(train_df) / (model_args.train_batch_size * 3) / 100.0)) * 100
    model_args.evaluate_during_training_verbose = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.use_multiprocessed_decoding = False
    model_args.overwrite_output_dir = True
    model_args.save_recent_only = True
    model_args.manual_seed = SEED
    model_args.early_stopping_patience = 25


    model_name_prefix = "Spanish" + model_name

    model_args.output_dir = os.path.join(model_name_prefix, "outputs")
    model_args.best_model_dir = os.path.join(model_name_prefix, "outputs", "best_model")
    model_args.cache_dir = os.path.join(model_name_prefix, "cache_dir")

    model = T5Model(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)

    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    result = model.eval_model(eval_df)

    test_list = []

    for index, row in spanish_test.iterrows():
        test_list.append("olid_a: " + row['comment'])

    model = T5Model(model_type, model_args.best_model_dir, args=model_args, use_cuda=torch.cuda.is_available(),cuda_device=cuda_device)

    preds = model.predict(test_list)
    macro_f1, weighted_f1 = sentence_label_evaluation(preds, spanish_test["label"].tolist())
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)

print("Weighted F1 scores ", weighted_f1_scores)
print("Mean weighted F1 scores", statistics.mean(weighted_f1_scores))
# print("STD weighted F1 scores", statistics.stdev(weighted_f1_scores))

print("Macro F1 scores ", macro_f1_scores)
print("Mean macro F1 scores", statistics.mean(macro_f1_scores))
# print("STD macro F1 scores", statistics.stdev(macro_f1_scores))