import math
import os

import numpy as np
import torch
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ft5.args import T5Args
from ft5.t5_model import T5Model

thresholds = [0.05, 0.1, 0.15]
offensive_thresholds = [0.8, 0.7, 0.6]
SEED = 777

for threshold in thresholds:

    for offensive_threshold in offensive_thresholds:
        solid = Dataset.to_pandas(load_dataset('tharindu/SOLID', split='train', sep="\t"))
        print("Building the model for ", threshold, " STD threshold and ", offensive_threshold, " offensive threshold")

        data = solid.loc[solid['std'] < threshold]
        data['target_text'] = np.where(data['average'] >= offensive_threshold, 'OFF', None)
        data['target_text'] = np.where(data['average'] <= (1 - offensive_threshold), 'NOT', None)

        data = data[data['target_text'].notna()]

        data["prefix"] = "olid_a"
        data = data.rename(columns={'text': 'input_text'})
        data = data[["prefix", "input_text", "target_text"]]

        train_df, eval_df = train_test_split(data, test_size=0.2, random_state=SEED)

        model_args = T5Args()
        model_args.num_train_epochs = 5
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

        model_type = "t5"
        model_name = "t5-base"
        model_name_prefix = "ft5_" + str(threshold) + "_" + str(offensive_threshold)

        model_args.output_dir = os.path.join(model_name_prefix, "outputs")
        model_args.best_model_dir = os.path.join(model_name_prefix, "outputs", "best_model")
        model_args.cache_dir = os.path.join(model_name_prefix, "cache_dir")

        # model_args.wandb_project = "ft5"
        # model_args.wandb_kwargs = {"name": model_name_prefix}

        model = T5Model(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available(), cuda_device=3)

        # Train the model
        model.train_model(train_df, eval_data=eval_df)

        # Evaluate the model
        result = model.eval_model(eval_df)

        model = None
        del model

