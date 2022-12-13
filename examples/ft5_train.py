import math
import os

import argparse
import numpy as np
import torch
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ft5.args import T5Args
from ft5.t5_model import T5Model
import pandas as pd

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="t5-base")
parser.add_argument('--model_type', required=False, help='model type', default="t5")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--strategy', required=False, help='datasets to use', default="both")
arguments = parser.parse_args()

model_type = arguments.model_type
model_name = arguments.model_name
strategy = arguments.strategy
cuda_device = int(arguments.cuda_device)
# offensive_thresholds = [0.8, 0.7, 0.6, 0.5]
SEED = 777

cctk = pd.read_csv("examples/CCTK/train.csv")

cctk['target_text'] = np.where(cctk['toxic'] == 1, 'TOX', 'NOT')
cctk = cctk[cctk['target_text'].notna()]
cctk["prefix"] = "cctk"
cctk = cctk.rename(columns={'comment_text': 'input_text'})
cctk = cctk[["prefix", "input_text", "target_text"]]

if strategy == "cctk":
    data = cctk
    data['target_text'] = np.where(data['toxic'] == 1, 'TOX', 'NOT')
    # data['target_text'] = np.where(data['toxic'] <= (1 - offensive_threshold), 'NOT', None)

    data = data[data['target_text'].notna()]

    data["prefix"] = "cctk"
    data = data.rename(columns={'comment_text': 'input_text'})
    data = data[["prefix", "input_text", "target_text"]]

    train_df, eval_df = train_test_split(data, test_size=0.2, random_state=SEED)

    model_args = T5Args()
    model_args.num_train_epochs = 25
    model_args.no_save = False
    model_args.fp16 = False
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 16
    model_args.max_length = 3
    model_args.max_seq_length = 256
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
    model_args.early_stopping_patience = 25
    model_args.manual_seed = SEED

    model_name_prefix = model_type + "_" + "cctk"

    model_args.output_dir = os.path.join(model_name_prefix, "outputs")
    model_args.best_model_dir = os.path.join(model_name_prefix, "outputs", "best_model")
    model_args.cache_dir = os.path.join(model_name_prefix, "cache_dir")

    model_args.wandb_project = "ft5"
    model_args.wandb_kwargs = {"name": model_name_prefix}

    model = T5Model(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)

    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    result = model.eval_model(eval_df)


elif strategy=="both" or strategy=="solid":

    thresholds = [0.05, 0.1, 0.15]

    for threshold in thresholds:

        solid = Dataset.to_pandas(load_dataset('tharindu/SOLID', split='train'))
        print("Building the model for ", threshold, " STD threshold ")

        data = solid.loc[solid['std'] < threshold]
        data['target_text'] = np.where(data['average'] >= 0.5, 'OFF', None)
        data['target_text'] = np.where(data['average'] < 0.5, 'NOT', None)

        data = data[data['target_text'].notna()]

        data["prefix"] = "olid_a"
        data = data.rename(columns={'text': 'input_text'})
        data = data[["prefix", "input_text", "target_text"]]

        if strategy == "both":

            full_data = pd.concat([data, cctk], ignore_index=True)
            full_data = full_data.sample(frac=1)

            model_name_prefix = model_type + "_" + str(threshold) + "SOLID_CCTK"

        elif strategy == "solid":
            full_data = data
            model_name_prefix = model_type + "_" + str(threshold) + "_SOLID"

        train_df, eval_df = train_test_split(full_data, test_size=0.2, random_state=SEED)

        model_args = T5Args()
        model_args.num_train_epochs = 25
        model_args.no_save = False
        model_args.fp16 = False
        model_args.learning_rate = 1e-5
        model_args.train_batch_size = 16
        model_args.max_length = 3
        model_args.max_seq_length = 256
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


        model_args.output_dir = os.path.join(model_name_prefix, "outputs")
        model_args.best_model_dir = os.path.join(model_name_prefix, "outputs", "best_model")
        model_args.cache_dir = os.path.join(model_name_prefix, "cache_dir")

        model_args.wandb_project = "ft5"
        model_args.wandb_kwargs = {"name": model_name_prefix}

        model = T5Model(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available(), cuda_device=cuda_device)

        # Train the model
        model.train_model(train_df, eval_data=eval_df)

        # Evaluate the model
        result = model.eval_model(eval_df)

        model = None
        del model

